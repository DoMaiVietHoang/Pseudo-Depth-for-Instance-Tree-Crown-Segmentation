"""Generate depth maps for the dataset using a chosen monocular depth model.

Saves per-tile .npy files under <out-dir>/<stem>_depth.npy with shape (H, W)
float32, matching what depthgate.dataset.LoadDepthFromFile expects.

Supports:
    --model dav2_small | dav2_base | dav2_large
    --model midas
    --model zoe_n | zoe_k | zoe_nk

Usage:
    python tools/generate_depth.py --model dav2_small \
        --img-dir data/bamforest/images/train \
        --out-dir data/bamforest/depth_dav2_small/train

Requires the corresponding package installed (transformers for DAv2/Zoe,
or `pip install timm` for MiDaS via torch.hub).
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from glob import glob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["dav2_small", "dav2_base", "dav2_large",
                            "midas", "zoe_n", "zoe_k", "zoe_nk"])
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ext", default="_depth.npy",
                   help="output suffix; default matches train config")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def load_dav2(name, device):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    hub_id = {
        "dav2_small": "depth-anything/Depth-Anything-V2-Small-hf",
        "dav2_base":  "depth-anything/Depth-Anything-V2-Base-hf",
        "dav2_large": "depth-anything/Depth-Anything-V2-Large-hf",
    }[name]
    proc = AutoImageProcessor.from_pretrained(hub_id)
    model = AutoModelForDepthEstimation.from_pretrained(hub_id).to(device).eval()
    return proc, model, "hf"


def load_midas(device):
    import torch
    m = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device).eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return transform, m, "midas"


def load_zoe(name, device):
    import torch
    repo = "isl-org/ZoeDepth"
    code = {"zoe_n": "ZoeD_N", "zoe_k": "ZoeD_K", "zoe_nk": "ZoeD_NK"}[name]
    m = torch.hub.load(repo, code, pretrained=True, trust_repo=True).to(device).eval()
    return None, m, "zoe"


def infer(model_bundle, img):
    import torch
    proc, model, kind = model_bundle
    if kind == "hf":
        inputs = proc(images=img, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            out = model(**inputs).predicted_depth
        d = out[0].cpu().numpy()
        return d
    if kind == "midas":
        x = proc(img).to(next(model.parameters()).device)
        with torch.no_grad():
            d = model(x)
            d = torch.nn.functional.interpolate(
                d.unsqueeze(1), size=img.shape[:2],
                mode="bicubic", align_corners=False).squeeze().cpu().numpy()
        return d
    if kind == "zoe":
        from PIL import Image
        pil = Image.fromarray(img)
        d = model.infer_pil(pil)
        return d
    raise ValueError(kind)


def main():
    args = parse_args()
    import cv2
    import numpy as np

    os.makedirs(args.out_dir, exist_ok=True)

    if args.model.startswith("dav2_"):
        bundle = load_dav2(args.model, args.device)
    elif args.model == "midas":
        bundle = load_midas(args.device)
    else:
        bundle = load_zoe(args.model, args.device)

    files = []
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        files.extend(glob(osp.join(args.img_dir, f"*{ext}")))
        files.extend(glob(osp.join(args.img_dir, f"*{ext.upper()}")))
    files = sorted(set(files))
    print(f"processing {len(files)} images with {args.model}")

    ok, skip, fail = 0, 0, 0
    for p in files:
        stem = osp.splitext(osp.basename(p))[0]
        out = osp.join(args.out_dir, stem + args.ext)
        if args.skip_existing and osp.isfile(out):
            skip += 1
            continue
        img_bgr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            print(f"[fail-read] {p}")
            fail += 1
            continue
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = img_bgr[:, :, :3]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            d = infer(bundle, img_rgb).astype(np.float32)
            if d.shape != img_rgb.shape[:2]:
                d = cv2.resize(d, (img_rgb.shape[1], img_rgb.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
            np.save(out, d)
            ok += 1
        except Exception as e:
            print(f"[fail-infer] {p}: {type(e).__name__}: {e}")
            fail += 1
    print(f"done: ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
