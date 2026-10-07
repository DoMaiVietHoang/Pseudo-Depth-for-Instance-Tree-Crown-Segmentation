"""Visualize DepthGate predictions from a trained checkpoint.

Usage:
    # Single image
    python tools/visualize.py CONFIG CHECKPOINT \
        --img path/to/image.tif --depth path/to/image_depth.npy \
        --out-dir vis_out --score-thr 0.3

    # Whole folder (depth auto-resolved next to image using config rules)
    python tools/visualize.py CONFIG CHECKPOINT \
        --img-dir /mnt/hoangdmv/Research/Instance_segmentation/dataset/ForestSeg-T1/Val/images \
        --depth-dir /mnt/hoangdmv/Research/Instance_segmentation/dataset/ForestSeg-T1/Val/depth \
        --depth-suffix _depth --out-dir vis_out --score-thr 0.3
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from glob import glob

import numpy as np

# Make project root importable when running as `python tools/visualize.py`.
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import depthgate  # noqa: F401  (register custom modules)


IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize DepthGate predictions")
    p.add_argument("config", help="model config path")
    p.add_argument("checkpoint", help="trained checkpoint .pth")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--img", help="single image path")
    src.add_argument("--img-dir", help="directory of images to visualize")
    p.add_argument("--depth", help="single depth .npy/.npz (use with --img)")
    p.add_argument("--depth-dir", help="depth directory (use with --img-dir)")
    p.add_argument("--depth-suffix", default="_depth",
                   help="filename suffix added to stem to find depth, e.g. _depth")
    p.add_argument("--depth-ext", default=".npy", choices=[".npy", ".npz"])
    p.add_argument("--out-dir", default="vis_out", help="where to save outputs")
    p.add_argument("--score-thr", type=float, default=0.3,
                   help="instance score threshold")
    p.add_argument("--max-images", type=int, default=0,
                   help="limit number of images (0 = all)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-depth-panel", action="store_true",
                   help="skip the side-by-side depth visualization")
    p.add_argument("--gt", help="COCO annotations json; if given, prepends a GT panel")
    p.add_argument("--no-gt-panel", action="store_true",
                   help="skip the GT panel even if --gt is provided")
    return p.parse_args()


def resolve_depth_path(img_path: str, args) -> str | None:
    if args.depth:
        return args.depth
    stem = osp.splitext(osp.basename(img_path))[0]
    base_dir = args.depth_dir or osp.dirname(img_path)
    cand = osp.join(base_dir, stem + args.depth_suffix + args.depth_ext)
    return cand if osp.isfile(cand) else None


def collect_images(args) -> list[str]:
    if args.img:
        return [args.img]
    files = []
    for ext in IMG_EXTS:
        files.extend(glob(osp.join(args.img_dir, f"*{ext}")))
        files.extend(glob(osp.join(args.img_dir, f"*{ext.upper()}")))
    files = sorted(set(files))
    if args.max_images > 0:
        files = files[: args.max_images]
    return files


def load_depth_array(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        with np.load(path) as data:
            key = "depth" if "depth" in data.files else data.files[0]
            arr = data[key]
    else:
        arr = np.load(path)
    if arr.ndim == 3:
        arr = np.squeeze(arr)
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, (2.0, 98.0))
    if hi - lo > 1e-6:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    else:
        arr = np.zeros_like(arr)
    return arr


def depth_to_color(depth: np.ndarray) -> np.ndarray:
    import cv2
    d8 = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)


def load_coco_index(gt_path: str):
    """Return (coco, name_to_img_id) for GT lookup."""
    from pycocotools.coco import COCO
    coco = COCO(gt_path)
    name_to_id = {}
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        name_to_id[info["file_name"]] = img_id
        # also index by basename in case file_name has a subdir prefix
        name_to_id[osp.basename(info["file_name"])] = img_id
    return coco, name_to_id


def _instance_palette(n: int, seed: int | None = None):
    """n visually distinct RGB colors via golden-ratio hue spacing.

    The starting hue and the order the colors are handed out are drawn at
    random, so the same instances get recolored on every run; pass ``seed`` to
    reproduce one particular coloring."""
    import colorsys
    rng = np.random.default_rng(seed)
    offset = float(rng.random())
    colors = []
    for i in rng.permutation(max(n, 1)):
        h = (offset + i * 0.6180339887498949) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def draw_masks(img_rgb: np.ndarray, masks, alpha: float = 0.45,
               thickness: int = 2, seed: int | None = None) -> np.ndarray:
    """Overlay each instance mask in its own color (translucent fill + solid
    contour). No bounding boxes. Returns RGB uint8."""
    import cv2
    canvas = img_rgb.copy().astype(np.uint8)
    masks = [np.asarray(m, dtype=bool) for m in masks]
    if not masks:
        return canvas
    colors = _instance_palette(len(masks), seed)
    fill = canvas.copy()
    for m, col in zip(masks, colors):
        fill[m] = col
    canvas = cv2.addWeighted(fill, alpha, canvas, 1.0 - alpha, 0)
    for m, col in zip(masks, colors):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, col, thickness, cv2.LINE_AA)
    return canvas


def get_gt_masks(coco, img_id):
    """List of per-instance boolean GT masks for an image (crowds excluded)."""
    anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            if not a.get("iscrowd", 0)]
    return [coco.annToMask(a).astype(bool) for a in anns]


def render_gt(coco, img_id, img_rgb):
    """Draw GT instances (per-instance color, no bbox). Returns RGB uint8."""
    return draw_masks(img_rgb, get_gt_masks(coco, img_id))


def add_caption(im: np.ndarray, text: str, height: int = 30) -> np.ndarray:
    """Stack a white caption strip on top of im."""
    import cv2
    w = im.shape[1]
    strip = np.full((height, w, 3), 255, dtype=np.uint8)
    cv2.putText(strip, text, (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([strip, im])


def _build_test_pipeline(cfg):
    """Build the config's test_pipeline, but drop LoadAnnotations (no GT at
    inference) so it works on raw images."""
    from mmengine.dataset import Compose
    pipe_cfg = cfg.test_pipeline or cfg.val_dataloader.dataset.pipeline
    pipe_cfg = [t for t in pipe_cfg if t.get("type") != "LoadAnnotations"]
    return Compose(pipe_cfg)


def visualize_one(model, test_pipeline, img_path: str,
                  depth_path: str | None, out_dir: str, score_thr: float,
                  show_depth_panel: bool, coco=None, name_to_id=None,
                  show_gt_panel: bool = False) -> bool:
    """Returns True on success, False if skipped."""
    import cv2
    import mmcv
    import torch

    img = mmcv.imread(img_path, channel_order="rgb")
    if img is None:
        print(f"[visualize] failed to read {img_path}, skip")
        return False

    if not (depth_path and osp.isfile(depth_path)):
        print(f"[visualize] no depth for {img_path}, skip")
        return False

    # Build data dict the pipeline expects. LoadDepthFromFile reads
    # results["depth_path"], LoadImageFromFile reads results["img_path"].
    data = dict(
        img_path=img_path,
        depth_path=depth_path,
        img_id=0,
    )
    data = test_pipeline(data)
    if data is None:
        print(f"[visualize] pipeline returned None for {img_path}, skip")
        return False

    # Pack into a batch of 1 and run model through the data_preprocessor
    # (handles uint8 -> float, mean/std normalize, pad to size_divisor).
    device = next(model.parameters()).device
    batch = {
        "inputs": [data["inputs"].to(device)],   # list of (3, H, W) tensors
        "data_samples": [data["data_samples"]],
    }
    # move depth to device
    d = batch["data_samples"][0].metainfo.get("depth")
    if d is not None:
        batch["data_samples"][0].set_metainfo({"depth": d.to(device)})

    model.eval()
    with torch.no_grad():
        batch = model.data_preprocessor(batch, training=False)
        results = model.predict(batch["inputs"], batch["data_samples"])
    result = results[0]

    # Draw predictions: per-instance colored masks, no bbox.
    pi = result.pred_instances
    scores = (pi.scores.cpu().numpy() if "scores" in pi
              else np.ones(len(pi), dtype=np.float32))
    pred_masks = []
    if "masks" in pi and len(pi) > 0:
        pm = pi.masks
        if hasattr(pm, "cpu"):
            pm = pm.cpu().numpy()
        pm = np.asarray(pm).astype(bool)
        pred_masks = [m for m, s in zip(pm, scores) if s >= score_thr]
    pred_vis = draw_masks(img, pred_masks)  # RGB uint8

    # Compose final panel: [GT] | Pred | [Depth]
    panels: list[tuple[str, np.ndarray]] = []

    if show_gt_panel and coco is not None and name_to_id is not None:
        key = osp.basename(img_path)
        img_id = name_to_id.get(key) or name_to_id.get(img_path)
        if img_id is not None:
            try:
                gt_vis = render_gt(coco, img_id, img)
                # Safety: keep GT panel the same size as the prediction panel.
                if gt_vis.shape[:2] != pred_vis.shape[:2]:
                    import cv2
                    gt_vis = cv2.resize(
                        gt_vis, (pred_vis.shape[1], pred_vis.shape[0]),
                        interpolation=cv2.INTER_AREA)
                panels.append(("Ground Truth", gt_vis))
            except Exception as e:
                print(f"[visualize] GT render failed for {img_path}: {e}")
        else:
            print(f"[visualize] no GT match for {key} in annotations")

    panels.append(("Prediction", pred_vis))

    if show_depth_panel and depth_path and osp.isfile(depth_path):
        try:
            depth = load_depth_array(depth_path)
            if depth.shape[:2] != pred_vis.shape[:2]:
                depth = cv2.resize(
                    depth, (pred_vis.shape[1], pred_vis.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            depth_bgr = depth_to_color(depth)
            depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
            panels.append(("Depth", depth_rgb))
        except Exception as e:
            print(f"[visualize] depth panel failed for {img_path}: {e}")

    # Add caption strip above each panel, then concat horizontally
    captioned = [add_caption(im, title) for title, im in panels]
    canvas = np.concatenate(captioned, axis=1) if len(captioned) > 1 else captioned[0]

    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, osp.splitext(osp.basename(img_path))[0] + "_vis.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"[visualize] saved {out_path}")
    return True


def main():
    args = parse_args()

    from mmengine.config import Config
    from mmdet.apis import init_detector

    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    test_pipeline = _build_test_pipeline(cfg)

    coco = None
    name_to_id = None
    show_gt = bool(args.gt) and not args.no_gt_panel
    if show_gt:
        coco, name_to_id = load_coco_index(args.gt)
        print(f"[visualize] loaded {len(name_to_id)} GT entries from {args.gt}")

    images = collect_images(args)
    if not images:
        print(f"[visualize] no images found")
        return

    print(f"[visualize] processing {len(images)} image(s) -> {args.out_dir}")
    ok = 0
    for img_path in images:
        depth_path = resolve_depth_path(img_path, args)
        try:
            if visualize_one(model, test_pipeline, img_path,
                             depth_path, args.out_dir, args.score_thr,
                             not args.no_depth_panel,
                             coco=coco, name_to_id=name_to_id,
                             show_gt_panel=show_gt):
                ok += 1
        except Exception as e:
            print(f"[visualize] error on {img_path}: {type(e).__name__}: {e}")
    print(f"[visualize] done: {ok}/{len(images)} succeeded")


if __name__ == "__main__":
    main()
