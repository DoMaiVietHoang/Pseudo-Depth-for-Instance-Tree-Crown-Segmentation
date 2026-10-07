"""Side-by-side qualitative comparison: GT | baseline | DepthGate.

Usage:
    python tools/eval/compare_qualitative.py \
        --img-dir data/bamforest/images/val \
        --gt data/bamforest/annotations/val.json \
        --base-config configs/mask2former_r50_bamforest.py \
        --base-ckpt work_dirs/baseline/best.pth \
        --our-config configs/mask2former_r50_depthgate_bamforest.py \
        --our-ckpt work_dirs/depthgate/best.pth \
        --out-dir qual_vis --max-images 30 --score-thr 0.3
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from glob import glob

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
import depthgate  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir", required=True)
    p.add_argument("--gt", help="COCO annotations json for GT panel")
    p.add_argument("--base-config", required=True)
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--our-config", required=True)
    p.add_argument("--our-ckpt", required=True)
    p.add_argument("--out-dir", default="qual_vis")
    p.add_argument("--max-images", type=int, default=20)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def collect_images(d, n):
    files = []
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        files.extend(glob(osp.join(d, f"*{ext}")))
        files.extend(glob(osp.join(d, f"*{ext.upper()}")))
    files = sorted(set(files))
    return files[:n] if n > 0 else files


def draw_predictions(visualizer, model, img_rgb, name, score_thr):
    from mmdet.apis import inference_detector
    result = inference_detector(model, img_rgb[..., ::-1])  # mmdet expects BGR
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample(
        name=name, image=img_rgb, data_sample=result,
        draw_gt=False, show=False, pred_score_thr=score_thr, out_file=None,
    )
    return visualizer.get_image()


def draw_gt(visualizer, coco, img_id, img_rgb, name, dataset_meta):
    import torch
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    if not anns:
        return img_rgb
    masks = np.stack([coco.annToMask(a) for a in anns], axis=0).astype(bool)
    bboxes = np.array([
        [a["bbox"][0], a["bbox"][1],
         a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
        for a in anns
    ], dtype=np.float32)
    labels = np.zeros(len(anns), dtype=np.int64)

    ds = DetDataSample()
    gt = InstanceData()
    gt.bboxes = torch.from_numpy(bboxes)
    gt.labels = torch.from_numpy(labels)
    gt.masks = torch.from_numpy(masks)
    ds.gt_instances = gt

    visualizer.dataset_meta = dataset_meta
    visualizer.add_datasample(
        name=name, image=img_rgb, data_sample=ds,
        draw_pred=False, draw_gt=True, show=False, out_file=None,
    )
    return visualizer.get_image()


def main():
    args = parse_args()
    import cv2
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import VISUALIZERS

    cfg_base = Config.fromfile(args.base_config)
    cfg_our = Config.fromfile(args.our_config)
    base_model = init_detector(cfg_base, args.base_ckpt, device=args.device)
    our_model = init_detector(cfg_our, args.our_ckpt, device=args.device)

    visualizer = VISUALIZERS.build(
        cfg_our.get("visualizer", dict(type="DetLocalVisualizer", name="vis")))

    coco = None
    img_id_by_name = {}
    if args.gt:
        from pycocotools.coco import COCO
        coco = COCO(args.gt)
        for img_id in coco.getImgIds():
            info = coco.loadImgs(img_id)[0]
            img_id_by_name[info["file_name"]] = img_id

    os.makedirs(args.out_dir, exist_ok=True)
    images = collect_images(args.img_dir, args.max_images)
    print(f"comparing {len(images)} images")

    for path in images:
        img_bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            print(f"skip unreadable {path}")
            continue
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = img_bgr[:, :, :3]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        panels = []
        if coco is not None:
            name = osp.basename(path)
            img_id = img_id_by_name.get(name)
            if img_id is not None:
                panels.append(("GT", draw_gt(visualizer, coco, img_id, img_rgb,
                                             name, our_model.dataset_meta)))

        panels.append(("Baseline",
                       draw_predictions(visualizer, base_model, img_rgb,
                                        osp.basename(path), args.score_thr)))
        panels.append(("DepthGate",
                       draw_predictions(visualizer, our_model, img_rgb,
                                        osp.basename(path), args.score_thr)))

        # Add caption strip
        h = panels[0][1].shape[0]
        w = panels[0][1].shape[1]
        cap = 30
        cells = []
        for title, im in panels:
            strip = np.full((cap, w, 3), 255, dtype=np.uint8)
            cv2.putText(strip, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cells.append(np.vstack([strip, im]))
        canvas = np.concatenate(cells, axis=1)
        out_path = osp.join(args.out_dir,
                            osp.splitext(osp.basename(path))[0] + "_cmp.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
