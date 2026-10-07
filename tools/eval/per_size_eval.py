"""COCOeval AP_S / AP_M / AP_L breakdown plus per-size recall.

Useful claim: 'DepthGate helps most on small / touching crowns'. This
script gives you the numbers to back that up.

Usage:
    python tools/eval/per_size_eval.py --gt val.json --results results.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out-dets", default="dets_coco.json")
    return p.parse_args()


def results_pkl_to_coco(results, out_json):
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
                "category_id": int(labels[i]) + 1,
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
    results = pickle.loads(Path(args.results).read_bytes())
    dets = results_pkl_to_coco(results, args.out_dets)
    if not dets:
        print("no detections, abort")
        return
    coco_dt = coco.loadRes(args.out_dets)
    for iou_type in ("bbox", "segm"):
        print(f"\n=== {iou_type} ===")
        e = COCOeval(coco, coco_dt, iou_type)
        e.evaluate(); e.accumulate(); e.summarize()


if __name__ == "__main__":
    main()
