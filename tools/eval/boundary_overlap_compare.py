"""Boundary/overlap qualitative figure (paper Sec. 5.4.3).

Contrasts the RGB-only Mask2Former baseline with DepthGate on *cropped regions
that contain touching crowns*, to show the target failure mode: the baseline
merges adjacent crowns of similar appearance, while DepthGate splits them using
the depth discontinuity.

One PNG:  rows = example crops,  cols = [ GT | Mask2Former (RGB) | DepthGate ]
Each instance gets its own colour; no bounding boxes.

The baseline is a plain RGB model (inference_detector). DepthGate is run through
its val dataset so the paired depth map is fed exactly as in eval.

Usage (defaults to BAMFOREST):
    python tools/eval/boundary_overlap_compare.py --n 4 --crop 420 --score-thr 0.3
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
# reuse the mask renderer / helpers from the per-forest comparison script
import compare_forests_depthgate as cf  # noqa: E402

DEF_BASE_CFG = "configs/mask2former_r50_bamforest.py"
DEF_BASE_CKPT = "work_dirs/mask2former_r50_bamforest/best_coco_segm_mAP_epoch_8.pth"
DEF_OUR_CFG = "configs/mask2former_r50_depthgate_bamforest.py"
DEF_OUR_CKPT = "work_dirs/mask2former_r50_depthgate_bamforest/epoch_46.pth"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-config", default=DEF_BASE_CFG)
    p.add_argument("--baseline-ckpt", default=DEF_BASE_CKPT)
    p.add_argument("--depthgate-config", default=DEF_OUR_CFG)
    p.add_argument("--depthgate-ckpt", default=DEF_OUR_CKPT)
    p.add_argument("--n", type=int, default=4, help="example crops")
    p.add_argument("--crop", type=int, default=420,
                   help="crop window size (px) centred on the touching cluster")
    p.add_argument("--dilate", type=int, default=10,
                   help="px tolerance for two GT crowns to count as touching")
    p.add_argument("--min-pairs", type=int, default=3,
                   help="min touching pairs for a tile to be a candidate")
    p.add_argument("--pool", type=int, default=80,
                   help="candidate tiles scanned to mine baseline/ours disagreement")
    p.add_argument("--iou-thr", type=float, default=0.5,
                   help="IoU to count a GT crown as recovered by a model")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default=osp.join(
        REPO, "prediction_viz", "boundary_overlap_compare.png"))
    return p.parse_args()


def abspath(p):
    return p if osp.isabs(p) else osp.join(REPO, p)


def touching_info(bboxes, dilate):
    """Return (num_touching_pairs, crop_center) from COCO [x,y,w,h] bboxes."""
    boxes = [(x, y, x + w, y + h) for x, y, w, h in bboxes]
    involved = []
    npair = 0
    for i in range(len(boxes)):
        a = boxes[i]
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if (a[0] - dilate < b[2] and b[0] - dilate < a[2] and
                    a[1] - dilate < b[3] and b[1] - dilate < a[3]):
                npair += 1
                involved.extend([i, j])
    if not involved:
        return 0, None
    cx = np.mean([(boxes[k][0] + boxes[k][2]) / 2 for k in involved])
    cy = np.mean([(boxes[k][1] + boxes[k][3]) / 2 for k in involved])
    return npair, (float(cx), float(cy))


def crop_window(center, size, H, W):
    cx, cy = center
    x0 = int(np.clip(cx - size / 2, 0, max(0, W - size)))
    y0 = int(np.clip(cy - size / 2, 0, max(0, H - size)))
    return x0, y0, min(x0 + size, W), min(y0 + size, H)


def gt_recovered(gt, pred, thr, size=160):
    """Bool per GT crown: recovered under greedy one-to-one matching at IoU>=thr
    (COCO-style, so one merged prediction can claim at most a single GT). Masks
    are downsampled for a fast ranking metric."""
    import cv2
    if not gt:
        return np.zeros(0, bool)
    matched = np.zeros(len(gt), bool)
    if not pred:
        return matched

    def ds(ms):
        return np.stack([
            cv2.resize(m.astype(np.uint8), (size, size),
                       interpolation=cv2.INTER_NEAREST).astype(bool).ravel()
            for m in ms]).astype(np.float32)

    G, P = ds(gt), ds(pred)
    inter = G @ P.T
    iou = inter / (G.sum(1)[:, None] + P.sum(1)[None, :] - inter + 1e-6)
    # greedy one-to-one assignment, highest IoU first
    order = np.argsort(iou, axis=None)[::-1]
    used_pred = set()
    for flat in order:
        g, p = divmod(int(flat), iou.shape[1])
        if iou[g, p] < thr:
            break
        if matched[g] or p in used_pred:
            continue
        matched[g] = True
        used_pred.add(p)
    return matched


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

    our_cfg = Config.fromfile(abspath(args.depthgate_config))
    base_cfg = Config.fromfile(abspath(args.baseline_config))
    our_model = init_detector(our_cfg, abspath(args.depthgate_ckpt),
                              device=args.device)
    base_model = init_detector(base_cfg, abspath(args.baseline_ckpt),
                               device=args.device)

    # Both configs share the depth-loading dataset pipeline, so run each model
    # through its OWN val dataset (which sets depth_path); the RGB baseline just
    # ignores the depth in metainfo. Match samples across datasets by img_id.
    def build_ds(cfg):
        ds = DATASETS.build(cfg.val_dataloader["dataset"])
        return ds, {ds.get_data_info(i)["img_id"]: i for i in range(len(ds))}

    our_ds, our_id2idx = build_ds(our_cfg)
    base_ds, base_id2idx = build_ds(base_cfg)

    ds_cfg = our_cfg.val_dataloader["dataset"]
    ann = ds_cfg["ann_file"]
    ann_path = ann if osp.isabs(ann) else osp.join(ds_cfg.get("data_root", ""), ann)
    coco = COCO(ann_path)

    def infer(model, ds, img_id):
        with torch.no_grad():
            res = model.test_step(pseudo_collate([ds[id2idx(ds, img_id)]]))[0]
        return cf.pred_masks(res, args.score_thr)

    def id2idx(ds, img_id):
        return our_id2idx[img_id] if ds is our_ds else base_id2idx[img_id]

    # candidate pool: tiles with touching GT crowns, ranked by touch density
    pool = []
    for img_id in our_id2idx:
        if img_id not in base_id2idx:
            continue
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        if len(anns) < 2:
            continue
        npair, center = touching_info([a["bbox"] for a in anns], args.dilate)
        if npair >= args.min_pairs and center is not None:
            pool.append((npair, img_id, center))
    pool.sort(reverse=True, key=lambda t: t[0])
    pool = pool[:args.pool]
    print(f"scanning {len(pool)} candidate tiles for baseline/DepthGate "
          f"disagreement ...")

    # score each pool tile by how many GT crowns DepthGate recovers that the
    # baseline misses (typically merged), and centre the crop on those crowns
    scored = []
    for npair, img_id, center in pool:
        gm = cf.gt_masks(coco, img_id)
        base_ok = gt_recovered(gm, infer(base_model, base_ds, img_id), args.iou_thr)
        our_ok = gt_recovered(gm, infer(our_model, our_ds, img_id), args.iou_thr)
        gain = int(our_ok.sum()) - int(base_ok.sum())
        fixed = (~base_ok) & our_ok            # DepthGate splits, baseline merged
        if fixed.any():
            cy = np.mean([np.nonzero(gm[k])[0].mean() for k in np.where(fixed)[0]])
            cx = np.mean([np.nonzero(gm[k])[1].mean() for k in np.where(fixed)[0]])
            center = (float(cx), float(cy))
        scored.append((gain, img_id, center))
    scored.sort(reverse=True, key=lambda t: t[0])
    sel = scored[:args.n]
    print(f"top disagreement gains (DepthGate − baseline recovered crowns): "
          f"{[g for g, _, _ in sel]}")
    if sel and sel[0][0] <= 0:
        print("WARNING: no tile where DepthGate beats the baseline was found; "
              "try raising --pool or lowering --iou-thr / --score-thr.")

    rows = []  # (gt_crop, base_crop, our_crop)
    for gain, img_id, center in sel:
        info = our_ds.get_data_info(our_id2idx[img_id])
        img_bgr = cv2.imread(info["img_path"], cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            continue
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = img_bgr[:, :, :3]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        x0, y0, x1, y1 = crop_window(center, args.crop, H, W)

        gm = cf.gt_masks(coco, img_id)
        bm = infer(base_model, base_ds, img_id)
        om = infer(our_model, our_ds, img_id)

        def crop_overlay(masks):
            cm = [m[y0:y1, x0:x1] for m in masks]
            cm = [m for m in cm if m.sum() > 0]
            return cf.overlay_instances(img_rgb[y0:y1, x0:x1], cm,
                                        args.seed, args.alpha)

        rows.append((crop_overlay(gm), crop_overlay(bm), crop_overlay(om)))

    del base_model, our_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not rows:
        raise SystemExit("no touching-crown crops found (try lowering --min-pairs)")

    # ---- academic figure: rows = crops, cols = [GT | baseline | ours] ----
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    headers = ["Ground truth", "Mask2Former (RGB)", "DepthGate (ours)"]
    nrow = len(rows)
    fig, ax = plt.subplots(nrow, 3, figsize=(11, 3.7 * nrow), squeeze=False,
                           gridspec_kw=dict(wspace=0.04, hspace=0.05))
    for r, (gt, base, ours) in enumerate(rows):
        for c, im in enumerate((gt, base, ours)):
            ax[r, c].imshow(im)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(headers[c], fontsize=14)
    fig.suptitle("Boundary and overlap analysis: the RGB baseline merges touching "
                 "crowns\nwhile DepthGate separates them via the depth "
                 "discontinuity", fontsize=14, y=1.0)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
