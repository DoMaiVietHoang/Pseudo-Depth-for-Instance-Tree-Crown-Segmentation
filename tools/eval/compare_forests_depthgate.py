"""One PNG comparing 3 per-forest DepthGate models, each on its OWN val set.

For every forest we build the val dataset *from its config* (so the paired
depth map is loaded and fed to the model exactly as in training/eval -- plain
inference_detector cannot do this), run the DepthGate checkpoint on N sample
tiles, and render three columns per tile:

    [ RGB + GT masks ] [ DAv2 val depth ] [ RGB + DepthGate pred masks ]

Each instance gets its own colour and NO bounding box is drawn. The three
forests are stacked as row-groups.

Usage (defaults to the 3 trained DepthGate models):
    python tools/eval/compare_forests_depthgate.py --n 4 --score-thr 0.3
"""

from __future__ import annotations

import argparse
import colorsys
import os.path as osp
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO)
import depthgate  # noqa: F401  (registers DepthCocoDataset / transforms / model)

DEFAULT_MODELS = [
    ("QuebecTrees",
     "configs/mask2former_r50_depthgate_quebectree.py",
     "work_dirs/mask2former_r50_depthgate_quebectree/best_coco_segm_mAP_epoch_40.pth"),
    ("ForestSeg",
     "configs/mask2former_r50_depthgate_forestseg.py",
     "work_dirs/mask2former_r50_depthgate_forestseg/best_coco_segm_mAP_epoch_22.pth"),
    ("BAMFOREST",
     "configs/mask2former_r50_depthgate_bamforest.py",
     "work_dirs/mask2former_r50_depthgate_bamforest/epoch_46.pth"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None,
                   help='"Name:config.py:ckpt.pth" triples; default = 3 forests')
    p.add_argument("--n", type=int, default=4, help="example tiles per forest")
    p.add_argument("--select", default="most",
                   choices=["most", "first", "random"],
                   help="most = tiles with the most GT crowns")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.5, help="mask fill opacity")
    p.add_argument("--depth-cmap", default="magma")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=osp.join(
        REPO, "prediction_viz", "forests_depthgate_compare.png"))
    return p.parse_args()


def resolve_models(args):
    if not args.models:
        return [(n, osp.join(REPO, c), osp.join(REPO, k))
                for n, c, k in DEFAULT_MODELS]
    out = []
    for spec in args.models:
        name, cfg, ckpt = spec.split(":", 2)
        out.append((name, cfg if osp.isabs(cfg) else osp.join(REPO, cfg),
                    ckpt if osp.isabs(ckpt) else osp.join(REPO, ckpt)))
    return out


def distinct_colors(n, seed=0):
    """n visually distinct RGB colours in [0,1] (shuffled hues)."""
    rng = np.random.default_rng(seed)
    if n == 0:
        return []
    hues = (np.arange(n) / n + rng.random()) % 1.0
    rng.shuffle(hues)
    return [colorsys.hsv_to_rgb(float(h), 0.65, 1.0) for h in hues]


def overlay_instances(img_rgb, masks, seed=0, alpha=0.5):
    """Blend each instance mask in its own colour + a crisp contour. No bbox."""
    import cv2
    out = img_rgb.astype(np.float32).copy()
    colors = distinct_colors(len(masks), seed)
    for m, c in zip(masks, colors):
        if m.sum() == 0:
            continue
        col = np.array(c, np.float32) * 255.0
        out[m] = (1.0 - alpha) * out[m] + alpha * col
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col.tolist(), 2)
    return np.clip(out, 0, 255).astype(np.uint8)


def gt_masks(coco, img_id):
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    return [coco.annToMask(a).astype(bool) for a in anns]


def pred_masks(result, score_thr):
    inst = result.pred_instances
    if len(inst) == 0 or not hasattr(inst, "masks"):
        return []
    scores = inst.scores.cpu().numpy()
    keep = scores >= score_thr
    masks = inst.masks.cpu().numpy().astype(bool)
    return [masks[i] for i in np.where(keep)[0]]


def main():
    args = parse_args()
    import cv2
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS
    from pycocotools.coco import COCO

    init_default_scope("mmdet")
    rng = np.random.default_rng(args.seed)
    models = resolve_models(args)

    blocks = []  # (forest_name, [ (tile_id, gt_rgb, depth_arr, pred_rgb) ... ])
    for name, cfg_path, ckpt in models:
        if not (osp.isfile(cfg_path) and osp.isfile(ckpt)):
            print(f"[skip] {name}: missing config or ckpt\n  {cfg_path}\n  {ckpt}")
            continue
        print(f"[{name}] loading model + val dataset ...")
        cfg = Config.fromfile(cfg_path)
        model = init_detector(cfg, ckpt, device=args.device)

        ds_cfg = cfg.val_dataloader["dataset"]
        dataset = DATASETS.build(ds_cfg)
        ann = ds_cfg["ann_file"]
        data_root = ds_cfg.get("data_root", "")
        ann_path = ann if osp.isabs(ann) else osp.join(data_root, ann)
        coco = COCO(ann_path)

        # choose example tiles
        ids = list(range(len(dataset)))
        if args.select == "first":
            idxs = ids[:args.n]
        elif args.select == "random":
            idxs = list(rng.choice(ids, size=min(args.n, len(ids)), replace=False))
        else:  # most crowns
            cnt = [(len(coco.getAnnIds(imgIds=dataset.get_data_info(i)["img_id"])), i)
                   for i in ids]
            cnt.sort(reverse=True)
            idxs = [i for _, i in cnt[:args.n]]

        items = []
        for idx in idxs:
            info = dataset.get_data_info(idx)
            img_bgr = cv2.imread(info["img_path"], cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                continue
            if img_bgr.ndim == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            elif img_bgr.shape[2] == 4:
                img_bgr = img_bgr[:, :, :3]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            sample = dataset[idx]                       # runs depth pipeline
            with torch.no_grad():
                result = model.test_step(pseudo_collate([sample]))[0]

            gm = gt_masks(coco, info["img_id"])
            pm = pred_masks(result, args.score_thr)
            depth = np.load(info["depth_path"]).astype(np.float32)
            items.append((
                overlay_instances(img_rgb, gm, args.seed, args.alpha),
                depth,
                overlay_instances(img_rgb, pm, args.seed, args.alpha),
            ))
        blocks.append((name, items))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- pack rows: each row holds up to 2 examples from the SAME dataset ----
    row_specs = []  # (dataset_name, is_block_start, [example, example?])
    for name, items in blocks:
        for k in range(0, len(items), 2):
            row_specs.append((name, k == 0, items[k:k + 2]))
    if not row_specs:
        raise SystemExit("no panels produced")

    # ---- academic-style figure ----
    plt.rcParams.update({"font.family": "serif", "font.size": 12,
                         "axes.titlesize": 13})
    # 7 columns: [GT depth pred | spacer | GT depth pred]
    wr = [1, 1, 1, 0.18, 1, 1, 1]
    trip = [(0, 1, 2), (4, 5, 6)]
    headers = {0: "Ground truth", 1: "Pseudo-depth (DAv2)", 2: "DepthGate",
               4: "Ground truth", 5: "Pseudo-depth (DAv2)", 6: "DepthGate"}
    nrow = len(row_specs)
    fig, ax = plt.subplots(nrow, 7, figsize=(18.5, 3.05 * nrow),
                           squeeze=False,
                           gridspec_kw=dict(width_ratios=wr, wspace=0.04,
                                            hspace=0.06))
    for r, (name, is_start, pair) in enumerate(row_specs):
        for c in range(7):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[r, 3].axis("off")                       # spacer column
        for t, cols in enumerate(trip):
            if t < len(pair):
                gt_rgb, depth, pred_rgb = pair[t]
                ax[r, cols[0]].imshow(gt_rgb)
                lo, hi = np.percentile(depth, [2, 98])
                ax[r, cols[1]].imshow(depth, cmap=args.depth_cmap,
                                      vmin=lo, vmax=hi)
                ax[r, cols[2]].imshow(pred_rgb)
            else:
                for cc in cols:
                    ax[r, cc].axis("off")
        if r == 0:
            for c, txt in headers.items():
                ax[r, c].set_title(txt)
        # dataset label on the far left of every row (bold on block start)
        ax[r, 0].set_ylabel(name, style="italic", fontsize=14,
                            fontweight="bold" if is_start else "normal",
                            labelpad=8)

    fig.suptitle("Qualitative results: each dataset's DepthGate model on its own "
                 "validation tiles\n(two example tiles per row; instances coloured "
                 "individually, no bounding boxes)", fontsize=15, y=1.0)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
