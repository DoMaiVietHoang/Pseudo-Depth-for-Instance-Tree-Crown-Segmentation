"""Convert the ForestSeg-T1 LabelMe dataset to COCO instance-segmentation JSON.

ForestSeg-T1 ships as::

    <root>/Train/images/<stem>.jpg     # RGB tiles (1024x1024)
    <root>/Train/labels/<stem>.json    # LabelMe polygons
    <root>/Train/masks/<stem>.png      # binary semantic mask (unused here)
    <root>/Val/...                      # same layout

Each LabelMe json holds ``shapes``: a list of ``shape_type="polygon"`` entries
whose ``points`` are absolute pixel coordinates [[x, y], ...]. There is a single
foreground class (label "0"), mapped to the COCO category ``tree`` so it matches
``classes=("tree",)`` / ``DepthCocoDataset.METAINFO`` in the configs.

DepthGate trains with ``depthgate.dataset.DepthCocoDataset`` (a COCO dataset),
so this script emits one COCO json per split::

    <root>/annotations/instances_train.json
    <root>/annotations/instances_val.json

Images are referenced in place by basename (no copying); point the config's
``data_prefix`` at ``Train/images`` / ``Val/images``.

After this, generate the paired depth maps that DepthGate fuses::

    python tools/generate_depth.py --model dav2_small \
        --img-dir dataset/ForestSeg-T1/Train/images \
        --out-dir dataset/ForestSeg-T1/Train/depth --device cuda:3
    python tools/generate_depth.py --model dav2_small \
        --img-dir dataset/ForestSeg-T1/Val/images \
        --out-dir dataset/ForestSeg-T1/Val/depth --device cuda:3

Usage::

    python tools/forestseg_to_coco.py --root dataset/ForestSeg-T1
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
                   help="ForestSeg-T1 root containing Train/ and Val/")
    p.add_argument("--splits", nargs="+", default=["Train", "Val"],
                   help="split sub-directories; output names are lowercased")
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


def find_label(label_dir, stem):
    cand = osp.join(label_dir, stem + ".json")
    return cand if osp.isfile(cand) else None


def convert_split(root, split, args, labels_seen):
    img_dir = osp.join(root, split, args.img_subdir)
    label_dir = osp.join(root, split, args.label_subdir)
    if not osp.isdir(img_dir):
        raise FileNotFoundError(f"missing image dir: {img_dir}")
    if not osp.isdir(label_dir):
        raise FileNotFoundError(f"missing label dir: {label_dir}")

    images, annotations = [], []
    img_id, ann_id = 0, 0
    n_no_label, n_bad_poly, n_non_polygon = 0, 0, 0

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

        label_path = find_label(label_dir, stem)
        if label_path is None:
            n_no_label += 1
            continue

        with open(label_path) as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            labels_seen.add(shape.get("label"))
            if shape.get("shape_type") != "polygon":
                n_non_polygon += 1
                continue
            pts = shape.get("points") or []
            if len(pts) < args.min_points:
                n_bad_poly += 1
                continue

            # Clamp into the image and flatten to COCO [x, y, x, y, ...].
            xs = [min(max(float(p[0]), 0.0), W) for p in pts]
            ys = [min(max(float(p[1]), 0.0), H) for p in pts]
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
        "info": {"description": f"ForestSeg-T1 {split} (LabelMe -> COCO)"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }

    out_dir = osp.join(root, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, f"instances_{split.lower()}.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)

    print(f"[{split}] images={len(images)} annotations={len(annotations)} "
          f"no_label={n_no_label} bad_polygons={n_bad_poly} "
          f"non_polygon_shapes={n_non_polygon}")
    print(f"[{split}] wrote {out_path}")


def main():
    args = parse_args()
    root = osp.abspath(args.root)
    labels_seen: set = set()
    for split in args.splits:
        convert_split(root, split, args, labels_seen)
    print(f"distinct LabelMe labels encountered: {sorted(labels_seen)} "
          f"(all mapped to category '{CATEGORIES[0]['name']}')")
    if len(labels_seen) > 1:
        print("WARNING: more than one distinct label found, but all were merged "
              "into a single 'tree' category. Edit CATEGORIES if you need "
              "multi-class output.")
    print("done.")


if __name__ == "__main__":
    main()
