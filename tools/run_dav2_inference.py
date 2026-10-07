"""Run Depth Anything v2 over a folder of RGB tiles and save .npy depth maps.

Optional helper; skip if you already have depth maps. Requires the
Depth-Anything-V2 repo on the Python path.

Usage:
    python tools/run_dav2_inference.py \
        --images data/bamforest/images/train \
        --out    data/bamforest/depth/train \
        --encoder vitl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    p.add_argument("--checkpoint", default=None, help="path to DAv2 weights")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

    cfgs = {
        "vits": dict(encoder="vits", features=64,  out_channels=[48, 96, 192, 384]),
        "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
        "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]),
        "vitg": dict(encoder="vitg", features=384, out_channels=[1536, 1536, 1536, 1536]),
    }
    model = DepthAnythingV2(**cfgs[args.encoder])
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(args.device).eval()

    images = sorted(Path(args.images).glob("*"))
    for img_path in images:
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        with torch.no_grad():
            depth = model.infer_image(img)
        out_path = out_dir / (img_path.stem + ".npy")
        np.save(out_path, depth.astype(np.float32))
        print(f"{img_path.name} -> {out_path.name}  ({depth.shape}, "
              f"min={depth.min():.2f}, max={depth.max():.2f})")


if __name__ == "__main__":
    main()
