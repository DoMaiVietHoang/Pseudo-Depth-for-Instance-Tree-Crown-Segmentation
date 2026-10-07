"""Spatial DepthGate gate maps across the three forest datasets.

The DepthGate module learns a per-pixel gate g in [0,1] (RGB-trust); the depth
reliance is (1 - g). Unlike a post-hoc attribution (e.g. Grad-CAM), this is an
explicit, learned reliability map. For each dataset's own DepthGate model we
forward-hook every fused level's gate to capture the full g tensor (works in
eval mode, where the module does not store its running mean) and visualise
(1 - g) over the RGB tile, with the ground-truth crowns overlaid.

Output:
    gate_maps.png   rows = tiles grouped by dataset; cols =
                    [RGB+GT | depth-reliance overlay | (1-g)@P3 | P4 | P5]

Usage (defaults to the 3 trained DepthGate models, 2 tiles each):
    python tools/eval/viz_gate_maps.py --n 2
"""

from __future__ import annotations

import argparse
import os.path as osp
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, osp.join(REPO, "tools", "eval"))
import depthgate  # noqa: F401
import compare_forests_depthgate as cf         # gt_masks helper

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
    p.add_argument("--n", type=int, default=2, help="tiles per dataset")
    p.add_argument("--select", default="most", choices=["most", "first"],
                   help="most = tiles with the most GT crowns")
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--alpha", type=float, default=0.55, help="overlay opacity")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=osp.join(REPO, "prediction_viz", "gate_maps.png"))
    return p.parse_args()


def abspath(p):
    return p if osp.isabs(p) else osp.join(REPO, p)


def resolve_models(args):
    if not args.models:
        return [(n, abspath(c), abspath(k)) for n, c, k in DEFAULT_MODELS]
    out = []
    for spec in args.models:
        name, cfg, ckpt = spec.split(":", 2)
        out.append((name, abspath(cfg), abspath(ckpt)))
    return out


def gt_outlines(img_rgb, inst, color=(255, 255, 0), thick=2):
    import cv2
    out = img_rgb.copy()
    for k in [int(i) for i in np.unique(inst) if i > 0]:
        cnts, _ = cv2.findContours((inst == k).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, thick)
    return out


def heat_overlay(img_rgb, heat, cmap, alpha):
    """Blend a scalar reliance map over a dimmed RGB image (contrast-stretched)."""
    lo, hi = np.percentile(heat, [2, 98])
    h = np.clip((heat - lo) / max(hi - lo, 1e-6), 0, 1)
    rgba = matplotlib.colormaps[cmap](h)[..., :3]
    base = img_rgb.astype(np.float32) / 255.0 * 0.45
    return np.clip(((1 - alpha) * base + alpha * rgba) * 255, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    import cv2
    import torch
    import torch.nn.functional as F
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS
    from pycocotools.coco import COCO

    init_default_scope("mmdet")
    names = None
    blocks = []                       # (dataset_name, [ (rgb_gt, overlay, rel) ])

    for ds_name, cfg_path, ckpt in resolve_models(args):
        if not (osp.isfile(cfg_path) and osp.isfile(ckpt)):
            print(f"[skip] {ds_name}: missing config or ckpt")
            continue
        print(f"[{ds_name}] loading model + val dataset ...")
        cfg = Config.fromfile(cfg_path)
        model = init_detector(cfg, ckpt, device=args.device)
        if not hasattr(model, "depth_gates"):
            print(f"[skip] {ds_name}: not a DepthGate model")
            del model
            continue

        captured = {}
        handles = []
        for key, mod in model.depth_gates.items():
            if hasattr(mod, "gate"):
                handles.append(mod.gate.register_forward_hook(
                    lambda m, i, o, k=key: captured.__setitem__(k, o.detach())))
        lvl_name = {k: f"P{int(k) + 2}" for k in model.depth_gates.keys()}
        if names is None:
            names = [lvl_name[k] for k in sorted(model.depth_gates.keys())]

        ds_cfg = cfg.val_dataloader["dataset"]
        dataset = DATASETS.build(ds_cfg)
        ann = ds_cfg["ann_file"]
        ann_path = ann if osp.isabs(ann) else osp.join(
            ds_cfg.get("data_root", ""), ann)
        coco = COCO(ann_path)

        order = list(range(len(dataset)))
        if args.select == "most":
            order.sort(key=lambda i: -len(coco.getAnnIds(
                imgIds=dataset.get_data_info(i)["img_id"])))
        idxs = order[:args.n]

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
            H, W = img_rgb.shape[:2]

            captured.clear()
            with torch.no_grad():
                model.test_step(pseudo_collate([dataset[idx]]))

            rel = {}
            for k in model.depth_gates.keys():
                if k not in captured:
                    continue
                g = captured[k].float()
                r = 1.0 - g.mean(dim=1, keepdim=True)
                r = F.interpolate(r, size=(H, W), mode="bilinear",
                                  align_corners=False)[0, 0].cpu().numpy()
                rel[lvl_name[k]] = r
            if not rel:
                continue
            agg = np.mean(list(rel.values()), axis=0)

            gm = cf.gt_masks(coco, info["img_id"])
            inst = np.zeros((H, W), np.int32)
            for i, m in enumerate(gm, start=1):
                inst[m] = i
            items.append((
                gt_outlines(img_rgb, inst),
                gt_outlines(heat_overlay(img_rgb, agg, args.cmap, args.alpha), inst),
                rel,
            ))
        blocks.append((ds_name, items))

        for h in handles:
            h.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = [(name, i == 0, it) for name, items in blocks
            for i, it in enumerate(items)]
    if not rows or names is None:
        raise SystemExit("no gate maps produced")

    ncol = 2 + len(names)
    heads = ["RGB + GT crowns", "Depth reliance (1-g)"] + \
            [f"(1-g) @ {n}" for n in names]
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(len(rows), ncol, figsize=(3.4 * ncol, 3.4 * len(rows)),
                           squeeze=False,
                           gridspec_kw=dict(wspace=0.05, hspace=0.05))
    for r, (name, is_start, (rgb_gt, overlay, rel)) in enumerate(rows):
        ax[r, 0].imshow(rgb_gt)
        ax[r, 1].imshow(overlay)
        for c, n in enumerate(names, start=2):
            ax[r, c].imshow(rel.get(n, np.zeros(rgb_gt.shape[:2])),
                            cmap=args.cmap, vmin=0, vmax=1)
        for c in range(ncol):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(heads[c], fontsize=12)
        ax[r, 0].set_ylabel(name, style="italic", fontsize=13,
                            fontweight="bold" if is_start else "normal",
                            labelpad=8)
    fig.suptitle("Learned DepthGate depth reliance (1-g): spatially structured and "
                 "level-dependent\n(high over canopy, low in inter-crown gaps; "
                 "each model on its own dataset)", y=1.0, fontsize=13)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)
fe

if __name__ == "__main__":
    main()
