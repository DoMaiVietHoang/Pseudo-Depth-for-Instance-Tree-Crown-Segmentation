"""Evaluate AP on the touching-crown subset of GT.

Builds a subset of validation GT where each instance has at least one other
instance whose mask-bbox overlaps (IoU>0 between dilated bboxes). This is
the subset DepthGate is supposed to help on.

Usage:
    python tools/eval/touching_crown_eval.py \
        --gt data/bamforest/annotations/val.json \
        --results work_dirs/depthgate/results.pkl \
        --dilate 8

Emits the COCO-style AP table on the touching subset only.
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import pickle
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True)
    p.add_argument("--results", required=True, help="results.pkl from tools/test.py")
    p.add_argument("--dilate", type=int, default=8,
                   help="pixels to dilate each bbox for touching test")
    p.add_argument("--out-gt", default="val_touching.json",
                   help="path to write subset GT json")
    p.add_argument("--out-dets", default="dets_coco.json",
                   help="path to write COCO-format detection json")
    return p.parse_args()


def bbox_dilated(b, d):
    x, y, w, h = b
    return [x - d, y - d, x + w + d, y + h + d]


def boxes_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def build_touching_subset(coco, dilate):
    keep_ann_ids = set()
    keep_img_ids = set()
    for img_id in coco.getImgIds():
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        if len(anns) < 2:
            continue
        bxs = [bbox_dilated(a["bbox"], dilate) for a in anns]
        # mark anns that touch >=1 other
        touched = [False] * len(anns)
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                if boxes_overlap(bxs[i], bxs[j]):
                    touched[i] = True
                    touched[j] = True
        for a, t in zip(anns, touched):
            if t:
                keep_ann_ids.add(a["id"])
                keep_img_ids.add(img_id)

    src = coco.dataset
    sub = {
        "info": src.get("info", {}),
        "licenses": src.get("licenses", []),
        "categories": src["categories"],
        "images": [im for im in src["images"] if im["id"] in keep_img_ids],
        "annotations": [a for a in src["annotations"] if a["id"] in keep_ann_ids],
    }
    return sub


def results_pkl_to_coco_dets(results, out_json):
    """Convert mmdet results pkl (list of DetDataSample dicts) to COCO json."""
    from pycocotools import mask as mask_utils
    dets = []
    for r in results:
        img_id = r["img_id"]
        pred = r["pred_instances"]
        scores = pred["scores"]
        labels = pred["labels"]
        masks = pred["masks"]
        if hasattr(scores, "cpu"):
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            masks = masks.cpu().numpy().astype(np.uint8)
        else:
            scores = np.asarray(scores)
            labels = np.asarray(labels)
            masks = np.asarray(masks).astype(np.uint8)
        # bbox from mask
        for i in range(len(scores)):
            m = masks[i]
            ys, xs = np.where(m)
            if ys.size == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            rle = mask_utils.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("ascii")
            dets.append({
                "image_id": int(img_id),
                "category_id": int(labels[i]) + 1,  # COCO is 1-indexed
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[i]),
                "segmentation": rle,
            })
    with open(out_json, "w") as f:
        json.dump(dets, f)
    return dets


def main():
    args = parse_args()
    coco = COCO(args.gt)
    sub = build_touching_subset(coco, args.dilate)
    print(f"touching subset: {len(sub['images'])} images / "
          f"{len(sub['annotations'])} annotations (out of "
          f"{len(coco.dataset['images'])} / {len(coco.dataset['annotations'])})")
    with open(args.out_gt, "w") as f:
        json.dump(sub, f)

    results = pickle.loads(Path(args.results).read_bytes())
    dets = results_pkl_to_coco_dets(results, args.out_dets)
    print(f"converted {len(dets)} predictions to COCO format")

    # Re-evaluate against subset
    coco_sub = COCO(args.out_gt)
    if not dets:
        print("no detections, skipping eval")
        return
    coco_dt = coco_sub.loadRes(args.out_dets)
    for iou_type in ("bbox", "segm"):
        print(f"\n=== TOUCHING SUBSET ({iou_type}) ===")
        e = COCOeval(coco_sub, coco_dt, iou_type)
        e.evaluate()
        e.accumulate()
        e.summarize()


if __name__ == "__main__":
    main()
