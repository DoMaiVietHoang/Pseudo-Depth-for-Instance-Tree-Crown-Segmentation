"""Boundary-AP computation for instance segmentation.

Implements Boundary IoU (Cheng et al., CVPR 2021):
    Boundary IoU = |B_gt ∩ B_pred| / |B_gt ∪ B_pred|
where B is the set of pixels within `dilation_ratio * diag(mask)` of the mask
boundary. AP is then computed against COCO ground truth using boundary IoU
as the matching criterion instead of mask IoU.

Usage:
    python tools/boundary_ap.py <gt_coco.json> <results.pkl> \
        [--dilation-ratio 0.02] [--iou-thr 0.5 0.75]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from scipy.ndimage import distance_transform_edt


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Return a binary band of pixels near the mask boundary."""
    if mask.sum() == 0:
        return mask.astype(bool)
    h, w = mask.shape
    diag = np.sqrt(h * h + w * w)
    d = max(1, int(round(dilation_ratio * diag)))
    pad = np.pad(mask.astype(np.uint8), 1, mode="constant")
    dist = distance_transform_edt(pad)
    dist = dist[1:-1, 1:-1]
    boundary = (mask > 0) & (dist <= d)
    return boundary


def boundary_iou(gt: np.ndarray, pred: np.ndarray, dilation_ratio: float = 0.02) -> float:
    bg = mask_to_boundary(gt, dilation_ratio)
    bp = mask_to_boundary(pred, dilation_ratio)
    inter = np.logical_and(bg, bp).sum()
    union = np.logical_or(bg, bp).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def compute_ap(matches: list[tuple[float, int]], n_gt: int) -> float:
    """COCO-style AP at a fixed IoU threshold from a sorted list of (score, tp)."""
    if n_gt == 0:
        return float("nan")
    matches.sort(key=lambda x: -x[0])
    tp = np.array([m[1] for m in matches])
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precision[recall >= t].max() if (recall >= t).any() else 0.0
        ap += p
    return ap / 101


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", help="COCO ground-truth json")
    ap.add_argument("results", help="MMDet results .pkl from tools/test.py --out")
    ap.add_argument("--dilation-ratio", type=float, default=0.02)
    ap.add_argument("--iou-thrs", type=float, nargs="+",
                    default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    args = ap.parse_args()

    coco = COCO(args.gt)
    results = pickle.loads(Path(args.results).read_bytes())

    img_id_to_gts: dict[int, list[np.ndarray]] = {}
    for img_id in coco.getImgIds():
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        masks = []
        for a in anns:
            if a.get("iscrowd", 0):
                continue
            masks.append(coco.annToMask(a).astype(bool))
        img_id_to_gts[img_id] = masks

    ap_per_thr: dict[float, float] = {}
    for thr in args.iou_thrs:
        all_matches: list[tuple[float, int]] = []
        total_gt = 0
        for res in results:
            img_id = res["img_id"]
            gts = img_id_to_gts.get(img_id, [])
            total_gt += len(gts)
            pred = res["pred_instances"]
            scores = pred["scores"].cpu().numpy() if hasattr(pred["scores"], "cpu") else np.asarray(pred["scores"])
            masks = pred["masks"]
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy().astype(bool)
            else:
                masks = np.asarray(masks).astype(bool)
            order = np.argsort(-scores)
            gt_used = [False] * len(gts)
            for i in order:
                m = masks[i]
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gts):
                    if gt_used[j]:
                        continue
                    if g.shape != m.shape:
                        continue
                    iou = boundary_iou(g, m, args.dilation_ratio)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= thr and best_j >= 0:
                    gt_used[best_j] = True
                    all_matches.append((float(scores[i]), 1))
                else:
                    all_matches.append((float(scores[i]), 0))
        ap_per_thr[thr] = compute_ap(all_matches, total_gt)
        print(f"Boundary-AP @ {thr:.2f} = {ap_per_thr[thr]:.4f}")

    mean_ap = float(np.nanmean(list(ap_per_thr.values())))
    print(f"\nBoundary-AP (mean, IoU=0.50:0.95) = {mean_ap:.4f}")
    print(f"Boundary-AP @ 0.50 = {ap_per_thr.get(0.5, float('nan')):.4f}")
    print(f"Boundary-AP @ 0.75 = {ap_per_thr.get(0.75, float('nan')):.4f}")


if __name__ == "__main__":
    main()
