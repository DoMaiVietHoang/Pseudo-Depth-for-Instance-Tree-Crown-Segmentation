"""Save per-pixel gate `g` heatmaps overlaid on the input image.

For interpretability: shows WHERE the model trusts RGB (g→1, bright) vs
depth (g→0, dark) for each fused FPN level.

Usage:
    python tools/eval/visualize_gate.py CONFIG CHECKPOINT \
        --img path/to/img.tif --depth path/to/img_depth.npy --out-dir gate_vis
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
import depthgate  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--img", required=True)
    p.add_argument("--depth", required=True)
    p.add_argument("--out-dir", default="gate_vis")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_depth(path):
    if path.endswith(".npz"):
        with np.load(path) as d:
            arr = d["depth"] if "depth" in d.files else d[d.files[0]]
    else:
        arr = np.load(path)
    if arr.ndim == 3:
        arr = arr.squeeze()
    arr = np.nan_to_num(arr.astype(np.float32))
    lo, hi = np.percentile(arr, (2.0, 98.0))
    if hi - lo > 1e-6:
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    else:
        arr = np.zeros_like(arr)
    return arr


def main():
    args = parse_args()
    import cv2
    import torch
    from mmengine.config import Config
    from mmdet.apis import init_detector

    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()

    # Patch DepthGate.forward to capture per-pixel gate maps
    from depthgate.depth_gate import DepthGate
    captured = {}
    orig_forward = DepthGate.forward

    def patched_forward(self, f_rgb, f_depth):
        if f_depth.shape[-2:] != f_rgb.shape[-2:]:
            f_depth_a = torch.nn.functional.interpolate(
                f_depth, size=f_rgb.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            f_depth_a = f_depth
        g = self.gate(torch.cat([f_rgb, f_depth_a], dim=1))
        a = self.depth_attn(f_depth_a)
        out = g * f_rgb + (1.0 - g) * (a * f_depth_a)
        # mean over channel dim → (H, W) spatial map
        captured.setdefault("gates", []).append(g.detach().cpu().mean(dim=1)[0].numpy())
        captured.setdefault("alphas", []).append(a.detach().cpu().mean(dim=1)[0].numpy())
        return out

    DepthGate.forward = patched_forward

    try:
        img_bgr = cv2.imread(args.img, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise RuntimeError(f"failed to read {args.img}")
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = img_bgr[:, :, :3]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        depth = load_depth(args.depth)

        # Pad/resize to pad_size_divisor=32
        h, w = img_rgb.shape[:2]
        nh = ((h + 31) // 32) * 32
        nw = ((w + 31) // 32) * 32
        img_pad = cv2.resize(img_rgb, (nw, nh))
        depth_pad = cv2.resize(depth, (nw, nh))

        x = torch.from_numpy(img_pad.transpose(2, 0, 1)).float()[None].to(args.device)
        mean = torch.tensor([123.675, 116.28, 103.53], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([58.395, 57.12, 57.375], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        d = torch.from_numpy(depth_pad).float()[None, None].to(args.device)

        # Build a minimal DataSample so _collect_depth works
        from mmengine.structures import InstanceData
        from mmdet.structures import DetDataSample
        ds = DetDataSample()
        ds.set_metainfo({"depth": d[0]})

        with torch.no_grad():
            model.extract_feat(x, [ds])

        gates  = captured.get("gates",  [])
        alphas = captured.get("alphas", [])
        if not gates:
            print("no gates captured (model has no DepthGate fusion?)")
            return

        os.makedirs(args.out_dir, exist_ok=True)
        stem = osp.splitext(osp.basename(args.img))[0]

        def to_heatmap(arr, colormap):
            up   = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
            norm = np.clip(up, 0, 1)
            bgr  = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        def blend(base, heat, alpha=0.5):
            return (alpha * base + (1 - alpha) * heat).astype(np.uint8)

        depth_heat = to_heatmap(depth_pad, cv2.COLORMAP_INFERNO)

        # Print stats table
        print(f"\n{'Level':<8} {'g mean':>8} {'g std':>8} {'a mean':>8} {'a std':>8}")
        print("-" * 44)
        for i, g_map in enumerate(gates):
            a_map  = alphas[i] if i < len(alphas) else None
            a_mean = a_map.mean() if a_map is not None else float("nan")
            a_std  = a_map.std()  if a_map is not None else float("nan")
            print(f"L{i+1:<7} {g_map.mean():>8.3f} {g_map.std():>8.3f}"
                  f" {a_mean:>8.3f} {a_std:>8.3f}")

        # Row 1 – gate g  (bright = trust RGB, dark = trust Depth)
        row_g = [img_pad, depth_heat]
        for g_map in gates:
            row_g.append(blend(img_pad, to_heatmap(g_map, cv2.COLORMAP_VIRIDIS)))

        # Row 2 – depth attention a  (bright = depth edge/ridge highlighted)
        row_a = [img_pad, depth_heat]
        for a_map in alphas:
            row_a.append(blend(img_pad, to_heatmap(a_map, cv2.COLORMAP_HOT)))

        # Row 3 – effective depth contribution (1-g)*a
        row_contrib = [img_pad, depth_heat]
        for g_map, a_map in zip(gates, alphas):
            contrib = (1.0 - g_map) * a_map
            row_contrib.append(blend(img_pad, to_heatmap(contrib, cv2.COLORMAP_JET)))

        # Pad all rows to same length
        blank = np.zeros_like(img_pad)
        n_cols = max(len(row_g), len(row_a), len(row_contrib))
        for row in (row_g, row_a, row_contrib):
            while len(row) < n_cols:
                row.append(blank)

        # Label column on the left
        def label_panel(text):
            p = np.full((nh, 80, 3), 240, dtype=np.uint8)
            cv2.putText(p, text, (4, nh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
            return p

        canvas = np.concatenate([
            np.concatenate([label_panel("g")]        + row_g,       axis=1),
            np.concatenate([label_panel("alpha")]    + row_a,       axis=1),
            np.concatenate([label_panel("(1-g)*a")]  + row_contrib, axis=1),
        ], axis=0)

        out_path = osp.join(args.out_dir, f"{stem}_gate.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"\nsaved → {out_path}")

        # Save raw arrays for quantitative analysis.
        # Per-level shapes differ (different FPN strides), so save under
        # per-level keys instead of stacking into a single array.
        npz_path = osp.join(args.out_dir, f"{stem}_maps.npz")
        npz_payload = {}
        for lvl, g_map in enumerate(gates):
            npz_payload[f"gate_L{lvl + 1}"] = g_map
        for lvl, a_map in enumerate(alphas):
            npz_payload[f"alpha_L{lvl + 1}"] = a_map
        np.savez_compressed(npz_path, **npz_payload)
        shapes = [g.shape for g in gates]
        print(f"saved → {npz_path}  (per-level gate shapes: {shapes})")
    finally:
        DepthGate.forward = orig_forward


if __name__ == "__main__":
    main()
