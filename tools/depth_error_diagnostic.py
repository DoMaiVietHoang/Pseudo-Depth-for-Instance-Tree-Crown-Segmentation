"""Diagnostic: of baseline RGB-only errors, how many are depth-resolvable?

For each baseline error (merged or split crown), measure whether the DAv2
depth map shows a meaningful discontinuity at the location the model got
wrong. If a substantial fraction of errors have depth signal, DepthGate is
worth building.

Usage:
    python tools/depth_error_diagnostic.py \
        --gt data/bamforest/annotations/val.json \
        --results work_dirs/baseline/results.pkl \
        --depth-dir data/bamforest/depth/val \
        --iou-thr 0.5
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO


def load_depth(depth_dir: Path, stem: str) -> np.ndarray | None:
    for ext in (".npy", ".npz"):
        p = depth_dir / (stem + ext)
        if p.exists():
            if ext == ".npy":
                return np.load(p)
            data = np.load(p)
            key = "depth" if "depth" in data.files else data.files[0]
            return data[key]
    return None


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def depth_ridge_score(depth: np.ndarray, region: np.ndarray) -> float:
    """Maximum normalized depth gradient inside `region`. High = strong ridge."""
    if region.sum() == 0:
        return 0.0
    gy, gx = np.gradient(depth.astype(np.float32))
    grad = np.sqrt(gy * gy + gx * gx)
    g = grad[region]
    if g.size == 0:
        return 0.0
    denom = grad.std() + 1e-6
    return float(g.max() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--ridge-thr", type=float, default=2.0,
                    help="z-score threshold for 'depth-resolvable'")
    args = ap.parse_args()

    coco = COCO(args.gt)
    results = pickle.loads(Path(args.results).read_bytes())
    depth_dir = Path(args.depth_dir)

    merge_errors = 0          # one pred covers >=2 GT
    split_errors = 0          # >=2 preds cover one GT
    merge_depth_ok = 0
    split_depth_ok = 0
    total_gt = 0

    for res in results:
        img_id = res["img_id"]
        info = coco.loadImgs(img_id)[0]
        stem = Path(info["file_name"]).stem
        depth = load_depth(depth_dir, stem)
        if depth is None:
            continue

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        gts = [coco.annToMask(a).astype(bool) for a in anns if not a.get("iscrowd", 0)]
        total_gt += len(gts)

        pred = res["pred_instances"]
        scores = pred["scores"]
        masks = pred["masks"]
        if hasattr(scores, "cpu"):
            scores = scores.cpu().numpy()
            masks = masks.cpu().numpy().astype(bool)
        order = np.argsort(-scores)
        masks = masks[order]
        scores = scores[order]

        # Resize depth to mask shape if needed
        if masks.size and depth.shape != masks.shape[1:]:
            import cv2
            depth = cv2.resize(depth, (masks.shape[2], masks.shape[1]),
                               interpolation=cv2.INTER_LINEAR)

        # MERGE errors: one prediction overlaps multiple GTs above IoU/2
        for m in masks:
            covered = [j for j, g in enumerate(gts)
                       if g.shape == m.shape and mask_iou(m, g) > args.iou_thr / 2]
            if len(covered) >= 2:
                merge_errors += 1
                # Score depth ridge inside the predicted mask
                if depth_ridge_score(depth, m) > args.ridge_thr:
                    merge_depth_ok += 1

        # SPLIT errors: one GT overlapped by multiple predictions above IoU/2
        for g in gts:
            overlapping = [i for i, m in enumerate(masks)
                           if g.shape == m.shape and mask_iou(g, m) > args.iou_thr / 2]
            if len(overlapping) >= 2:
                split_errors += 1
                if depth_ridge_score(depth, g) > args.ridge_thr:
                    split_depth_ok += 1

    print(f"Total GT instances:       {total_gt}")
    print(f"Merge errors:             {merge_errors}")
    print(f"  with strong depth ridge: {merge_depth_ok}  "
          f"({merge_depth_ok / max(merge_errors,1):.1%})")
    print(f"Split errors:             {split_errors}")
    print(f"  with strong depth ridge: {split_depth_ok}  "
          f"({split_depth_ok / max(split_errors,1):.1%})")
    total_err = merge_errors + split_errors
    total_ok = merge_depth_ok + split_depth_ok
    if total_err:
        frac = total_ok / total_err
        print(f"\nOverall depth-resolvable error fraction: {frac:.1%}")
        if frac >= 0.30:
            print("=> Green light: DepthGate likely to give meaningful gains.")
        elif frac >= 0.10:
            print("=> Yellow: modest gains expected; proceed with tempered expectations.")
        else:
            print("=> Red: depth quality or error type is not depth-resolvable. Reconsider method.")


if __name__ == "__main__":
    main()
