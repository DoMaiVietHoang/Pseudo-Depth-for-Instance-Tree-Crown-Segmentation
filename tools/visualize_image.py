"""Visualize predictions of any trained model(s) on given images.

Each model is given as CONFIG:CHECKPOINT[:LABEL]. Whether a model needs a depth
map is auto-detected from its test_pipeline (RGB-only baselines have no
LoadDepthFromFile), so RGB, RGB-D concat and DepthGate checkpoints can all be
passed in the same run.

Saves the prediction only -- instance masks overlaid on the image, no GT panel,
no depth panel, no caption -- as one PNG per image per model:

    <out-dir>/<image stem>_<label>.png

Usage:
    python tools/visualize_image.py \
        --img dataset/ForestSeg-T1/Val/images/tile_012.jpg \
        --model configs/mask2former_r50_forestseg.py:work_dirs/mask2former_r50_forestseg/best_coco_segm_mAP_epoch_8.pth:RGB \
        --model configs/mask2former_r50_depthgate_forestseg.py:work_dirs/mask2former_r50_depthgate_forestseg/best_coco_segm_mAP_epoch_22.pth:DepthGate \
        --out-dir prediction_viz --score-thr 0.3
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import depthgate  # noqa: F401  (register custom modules)

from tools.visualize import draw_masks  # reuse the existing mask renderer


def parse_args():
    p = argparse.ArgumentParser(
        description="Save model predictions for given images")
    p.add_argument("--img", required=True, nargs="+",
                   help="one or more image paths")
    p.add_argument("--model", required=True, action="append", metavar="SPEC",
                   help="CONFIG:CHECKPOINT[:LABEL], repeatable")
    p.add_argument("--depth-dir",
                   help="depth directory; defaults to <image dir>/../depth "
                        "then <image dir>")
    p.add_argument("--depth-suffix", default="_depth")
    p.add_argument("--depth-ext", default=".npy", choices=[".npy", ".npz"])
    p.add_argument("--out-dir", default="prediction_viz")
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=None,
                   help="fix the instance colors; omit for new colors each run")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def parse_model_spec(spec: str) -> tuple[str, str, str]:
    """CONFIG:CHECKPOINT[:LABEL] -> (config, checkpoint, label)."""
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"[viz] bad --model {spec!r}: need CONFIG:CHECKPOINT[:LABEL]")
    config, checkpoint = parts[0], parts[1]
    label = ":".join(parts[2:]) if len(parts) > 2 else osp.splitext(
        osp.basename(config))[0]
    for path in (config, checkpoint):
        if not osp.isfile(path):
            raise SystemExit(f"[viz] not found: {path}")
    return config, checkpoint, label


def build_test_pipeline(cfg):
    """The config's test_pipeline without LoadAnnotations (no GT at inference).

    Also reports whether the pipeline reads a depth map.
    """
    from mmengine.dataset import Compose
    pipe_cfg = cfg.test_pipeline or cfg.val_dataloader.dataset.pipeline
    needs_depth = any(t.get("type") == "LoadDepthFromFile" for t in pipe_cfg)
    pipe_cfg = [t for t in pipe_cfg if t.get("type") != "LoadAnnotations"]
    return Compose(pipe_cfg), needs_depth


def load_models(specs: list[str], device: str):
    """[(label, model, pipeline, needs_depth), ...]"""
    from mmengine.config import Config
    from mmdet.apis import init_detector

    models = []
    for spec in specs:
        config, checkpoint, label = parse_model_spec(spec)
        cfg = Config.fromfile(config)
        model = init_detector(cfg, checkpoint, device=device)
        model.eval()
        pipeline, needs_depth = build_test_pipeline(cfg)
        kind = "RGB+depth" if needs_depth else "RGB only"
        print(f"[viz] loaded {label:<12} ({kind})  {osp.basename(checkpoint)}")
        models.append((label, model, pipeline, needs_depth))
    return models


def resolve_depth_path(img_path: str, args) -> str | None:
    """Depth for an image: <depth-dir>/<stem><suffix><ext>, with a sibling-dir
    fallback (images/<stem>.jpg -> depth/<stem>_depth.npy)."""
    stem = osp.splitext(osp.basename(img_path))[0]
    name = stem + args.depth_suffix + args.depth_ext
    img_dir = osp.dirname(osp.abspath(img_path))
    candidates = [args.depth_dir] if args.depth_dir else [
        osp.join(osp.dirname(img_dir), "depth"), img_dir]
    for d in candidates:
        cand = osp.join(d, name)
        if osp.isfile(cand):
            return cand
    return None


def predict_masks(model, pipeline, img_path: str, depth_path: str | None,
                  needs_depth: bool, score_thr: float) -> list[np.ndarray]:
    """Run one model on one image; return per-instance boolean masks above
    score_thr, in the image's original resolution."""
    import torch

    data = dict(img_path=img_path, img_id=0)
    if depth_path:
        data["depth_path"] = depth_path
    data = pipeline(data)
    if data is None:
        raise RuntimeError("test pipeline returned None")

    device = next(model.parameters()).device
    batch = {
        "inputs": [data["inputs"].to(device)],
        "data_samples": [data["data_samples"]],
    }
    depth = batch["data_samples"][0].metainfo.get("depth")
    # MaskFormerDepthGate.extract_feat silently falls back to RGB-only features
    # when no depth reaches it, so a depth model would still emit plausible but
    # wrong masks. Fail loudly instead.
    if needs_depth and depth is None:
        raise RuntimeError(
            "model expects depth but none survived the test pipeline "
            f"(depth_path={depth_path})")
    if depth is not None:
        batch["data_samples"][0].set_metainfo({"depth": depth.to(device)})

    with torch.no_grad():
        batch = model.data_preprocessor(batch, training=False)
        result = model.predict(batch["inputs"], batch["data_samples"])[0]

    pi = result.pred_instances
    if "masks" not in pi or len(pi) == 0:
        return []
    scores = (pi.scores.cpu().numpy() if "scores" in pi
              else np.ones(len(pi), dtype=np.float32))
    masks = pi.masks
    if hasattr(masks, "cpu"):
        masks = masks.cpu().numpy()
    masks = np.asarray(masks).astype(bool)
    return [m for m, s in zip(masks, scores) if s >= score_thr]


def slugify(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in label)


def visualize_one(img_path: str, models, args) -> list[str]:
    """Write one prediction image per model. Returns the paths written."""
    import cv2
    import mmcv

    img = mmcv.imread(img_path, channel_order="rgb")
    if img is None:
        print(f"[viz] cannot read {img_path}, skip")
        return []

    depth_path = resolve_depth_path(img_path, args)
    if depth_path is None and any(needs for _, _, _, needs in models):
        print(f"[viz] no depth map for {img_path} but a model needs one, skip")
        return []

    os.makedirs(args.out_dir, exist_ok=True)
    stem = osp.splitext(osp.basename(img_path))[0]

    written = []
    for label, model, pipeline, needs_depth in models:
        masks = predict_masks(model, pipeline, img_path,
                              depth_path if needs_depth else None,
                              needs_depth, args.score_thr)
        vis = draw_masks(img, masks, seed=args.seed)
        out_path = osp.join(args.out_dir, f"{stem}_{slugify(label)}.png")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"[viz] {label}: {len(masks)} instances -> {out_path}")
        written.append(out_path)
    return written


def main():
    args = parse_args()
    models = load_models(args.model, args.device)

    for img_path in args.img:
        try:
            visualize_one(img_path, models, args)
        except Exception as e:
            print(f"[viz] error on {img_path}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()