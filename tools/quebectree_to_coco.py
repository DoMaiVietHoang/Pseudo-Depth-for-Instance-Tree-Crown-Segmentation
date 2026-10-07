"""Convert the QuebecTree YOLO-seg dataset to COCO instance-segmentation JSON.

The QuebecTree dataset ships as:

    <root>/images/{train,val}/<stem>.jpg          # RGB tiles
    <root>/labels/{train,val}/<stem>.txt          # YOLO-seg polygons

Each label line is::

    <class_id> x1 y1 x2 y2 ... xn yn       # all coords normalized to [0, 1]

DepthGate trains with ``depthgate.dataset.DepthCocoDataset`` (a COCO dataset),
so this script emits one COCO json per split::

    <root>/annotations/instances_train.json
    <root>/annotations/instances_val.json

Images are referenced in place by basename (no copying); point the config's
``data_prefix`` at ``images/<split>``.

After this, generate the paired depth maps that DepthGate fuses::

    python tools/generate_depth.py --model dav2_small \
        --img-dir dataset/QuebecTree/images/train \
        --out-dir dataset/QuebecTree/depth_train
    python tools/generate_depth.py --model dav2_small \
        --img-dir dataset/QuebecTree/images/val \
        --out-dir dataset/QuebecTree/depth_val

Usage::

    python tools/quebectree_to_coco.py --root dataset/QuebecTree
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from glob import glob

from PIL import Image


# Single foreground class — must match `classes=("tree",)` in the config so the
# COCO dataset can map this category to contiguous label 0.
CATEGORIES = [{"id": 1, "name": "tree", "supercategory": "tree"}]
TREE_CAT_ID = 1

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True,
                   help="QuebecTree root containing images/ and labels/")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--img-subdir", default="images")
    p.add_argument("--label-subdir", default="labels")
    p.add_argument("--out-subdir", default="annotations")
    p.add_argument("--min-points", type=int, default=3,
                   help="drop polygons with fewer than this many vertices")
    return p.parse_args()


def polygon_area(xs, ys):
    """Shoelace area for a single polygon ring (pixel coords)."""
    n = len(xs)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(s) * 0.5


def find_image(img_dir, stem):
    for ext in IMG_EXTS:
        for e in (ext, ext.upper()):
            cand = osp.join(img_dir, stem + e)
            if osp.isfile(cand):
                return cand
    return None


def convert_split(root, split, args):
    img_dir = osp.join(root, args.img_subdir, split)
    label_dir = osp.join(root, args.label_subdir, split)
    if not osp.isdir(img_dir):
        raise FileNotFoundError(f"missing image dir: {img_dir}")
    if not osp.isdir(label_dir):
        raise FileNotFoundError(f"missing label dir: {label_dir}")

    images, annotations = [], []
    img_id, ann_id = 0, 0
    n_no_label, n_bad_poly = 0, 0

    img_paths = []
    for ext in IMG_EXTS:
        img_paths.extend(glob(osp.join(img_dir, f"*{ext}")))
        img_paths.extend(glob(osp.join(img_dir, f"*{ext.upper()}")))
    img_paths = sorted(set(img_paths))

    for img_path in img_paths:
        stem = osp.splitext(osp.basename(img_path))[0]
        with Image.open(img_path) as im:
            W, H = im.size
        img_id += 1
        # file_name is the basename; data_prefix in the config supplies the dir.
        images.append({
            "id": img_id,
            "file_name": osp.basename(img_path),
            "width": W,
            "height": H,
        })

        label_path = osp.join(label_dir, stem + ".txt")
        if not osp.isfile(label_path):
            n_no_label += 1
            continue

        with open(label_path) as f:
            for line in f:
                vals = line.split()
                if len(vals) < 1 + 2 * args.min_points:
                    if vals:
                        n_bad_poly += 1
                    continue
                coords = list(map(float, vals[1:]))
                if len(coords) % 2 != 0:
                    n_bad_poly += 1
                    continue
                xs = [coords[i] * W for i in range(0, len(coords), 2)]
                ys = [coords[i] * H for i in range(1, len(coords), 2)]
                # Clamp into the image and flatten back to COCO [x,y,x,y,...].
                xs = [min(max(x, 0.0), W) for x in xs]
                ys = [min(max(y, 0.0), H) for y in ys]
                seg = []
                for x, y in zip(xs, ys):
                    seg.extend([round(x, 2), round(y, 2)])

                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                bw, bh = x1 - x0, y1 - y0
                if bw <= 0 or bh <= 0:
                    n_bad_poly += 1
                    continue
                area = polygon_area(xs, ys)
                if area <= 0:
                    n_bad_poly += 1
                    continue

                ann_id += 1
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": TREE_CAT_ID,
                    "segmentation": [seg],
                    "area": round(area, 2),
                    "bbox": [round(x0, 2), round(y0, 2), round(bw, 2), round(bh, 2)],
                    "iscrowd": 0,
                })

    coco = {
        "info": {"description": f"QuebecTree {split} (YOLO-seg -> COCO)"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }

    out_dir = osp.join(root, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f"instances_{split}.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)

    print(f"[{split}] images={len(images)} annotations={len(annotations)} "
          f"no_label={n_no_label} bad_polygons={n_bad_poly}")
    print(f"[{split}] wrote {out_path}")


def main():
    args = parse_args()
    root = osp.abspath(args.root)
    for split in args.splits:
        convert_split(root, split, args)
    print("done.")


if __name__ == "__main__":
    main()
