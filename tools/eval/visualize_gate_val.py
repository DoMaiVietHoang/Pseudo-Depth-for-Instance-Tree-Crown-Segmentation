"""Save per-pixel gate `g` (+ alpha + contrib) heatmaps for the ENTIRE val set.

Walks the val dataloader built from CONFIG, runs `extract_feat` on each
sample with `DepthGate.forward` patched to capture gate / alpha maps, and
writes one `<stem>_gate.jpg` + `<stem>_maps.npz` per image into --out-dir.

Also writes `<out-dir>/summary.csv` with per-image gate/alpha mean+std per level.

Usage:
    PYTHONPATH=. python tools/eval/visualize_gate_val.py CONFIG CHECKPOINT \
        [--out-dir gate_vis_val] [--device cuda:0] [--limit N] \
        [--no-overlay]   # skip jpg, write only .npz + csv (faster)
"""

from __future__ import annotations

import argparse
import csv
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
    p.add_argument("--out-dir", default="gate_vis_val")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=0,
                   help="Only process the first N samples (0 = all).")
    p.add_argument("--no-overlay", action="store_true",
                   help="Skip jpg overlays; only dump .npz + csv.")
    return p.parse_args()


def main():
    args = parse_args()
    import cv2
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.apis import init_detector

    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()

    # Build val dataloader directly from the config so the pipeline (depth
    # load + resize + normalize) matches what the model was evaluated with.
    val_loader = Runner.build_dataloader(cfg.val_dataloader)

    # Patch DepthGate.forward to capture per-pixel gate maps for the current sample.
    from depthgate.depth_gate import DepthGate
    captured: dict[str, list] = {}
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
        captured.setdefault("gates", []).append(g.detach().cpu().mean(dim=1)[0].numpy())
        captured.setdefault("alphas", []).append(a.detach().cpu().mean(dim=1)[0].numpy())
        return out

    DepthGate.forward = patched_forward

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = osp.join(args.out_dir, "summary.csv")
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["stem", "level", "g_mean", "g_std", "a_mean", "a_std"])

    # ---------- Paper-style rendering helpers ----------
    PANEL_BG     = (252, 252, 252)
    BORDER_COLOR = (200, 200, 200)
    TEXT_COLOR   = (35, 35, 35)
    SUBTLE_COLOR = (110, 110, 110)
    HEADER_BG    = (235, 235, 235)

    HEADER_H = 34
    FOOTER_H = 26
    SIDE_W   = 56
    GAP      = 6
    BORDER   = 1
    COLORBAR_W = 28

    def _coolwarm_lut():
        # Diverging colormap: blue (low / trust RGB) ↔ white (0.5) ↔ red (high / trust depth).
        # Built once; values 0..255 → BGR triplets matching cv2 convention.
        xs = np.linspace(0, 1, 256, dtype=np.float32)
        # Cubic-ease toward extremes so mid range (~0.5) stays visibly distinct.
        def lerp(a, b, t):
            return a + (b - a) * t
        r = np.where(xs < 0.5,
                     lerp(59,  247, xs * 2),
                     lerp(247, 178, (xs - 0.5) * 2)) / 255.0
        g = np.where(xs < 0.5,
                     lerp(76,  247, xs * 2),
                     lerp(247, 24,  (xs - 0.5) * 2)) / 255.0
        b = np.where(xs < 0.5,
                     lerp(192, 247, xs * 2),
                     lerp(247, 43,  (xs - 0.5) * 2)) / 255.0
        rgb = np.stack([r, g, b], axis=1)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        bgr = rgb[:, ::-1]
        return bgr.reshape(256, 1, 3)

    COOLWARM_LUT = _coolwarm_lut()

    def to_heatmap(arr, colormap, size):
        up = cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)
        norm = np.clip(up, 0, 1)
        idx = (norm * 255).astype(np.uint8)
        if colormap == "coolwarm":
            bgr = COOLWARM_LUT[idx, 0]
        else:
            bgr = cv2.applyColorMap(idx, colormap)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def blend(base, heat, alpha=0.55):
        return (alpha * heat + (1 - alpha) * base).astype(np.uint8)

    def add_border(panel):
        return cv2.copyMakeBorder(panel, BORDER, BORDER, BORDER, BORDER,
                                  cv2.BORDER_CONSTANT, value=BORDER_COLOR)

    def header_strip(width, text, sub=None):
        strip = np.full((HEADER_H, width, 3), HEADER_BG, dtype=np.uint8)
        cv2.putText(strip, text, (10, 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, TEXT_COLOR, 1, cv2.LINE_AA)
        if sub:
            (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(strip, sub, (width - tw - 10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, SUBTLE_COLOR, 1, cv2.LINE_AA)
        return strip

    def footer_strip(width, text):
        strip = np.full((FOOTER_H, width, 3), PANEL_BG, dtype=np.uint8)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(strip, text, ((width - tw) // 2, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
        return strip

    def side_label(height, text):
        strip = np.full((height, SIDE_W, 3), PANEL_BG, dtype=np.uint8)
        # Vertical text — render horizontally on a tall canvas then rotate.
        tmp = np.full((SIDE_W, height, 3), PANEL_BG, dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
        cv2.putText(tmp, text, ((height - tw) // 2, SIDE_W // 2 + th // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, TEXT_COLOR, 1, cv2.LINE_AA)
        rotated = cv2.rotate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
        strip[:rotated.shape[0], :rotated.shape[1]] = rotated
        return strip

    def make_panel(content, title, subtitle, footer):
        h, w = content.shape[:2]
        body = add_border(content)
        bw = body.shape[1]
        head = header_strip(bw, title, subtitle)
        foot = footer_strip(bw, footer)
        return np.concatenate([head, body, foot], axis=0)

    def vertical_colorbar(total_height, colormap, vmin, vmax, label):
        # Build a colorbar block whose final height exactly matches `total_height`
        # (the height of a sibling panel: header + bordered body + footer).
        # Layout mirrors make_panel: [header_strip | (bordered bar+ticks) | footer_strip].
        body_h = total_height - HEADER_H - FOOTER_H
        bar_h = max(body_h - 2 * BORDER, 1)

        grad = np.linspace(1.0, 0.0, bar_h, dtype=np.float32).reshape(-1, 1)
        grad = np.tile(grad, (1, COLORBAR_W))
        bar_rgb = to_heatmap(grad, colormap, (COLORBAR_W, bar_h))

        tick_w = 52
        ticks = np.full((bar_h, tick_w, 3), PANEL_BG, dtype=np.uint8)
        for frac, val in [(0.0, vmax), (0.5, (vmin + vmax) / 2), (1.0, vmin)]:
            y = int(frac * (bar_h - 1))
            y = max(12, min(bar_h - 4, y))
            cv2.putText(ticks, f"{val:.2f}", (4, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_COLOR, 1, cv2.LINE_AA)

        bar_with_ticks = np.concatenate([bar_rgb, ticks], axis=1)
        bar_with_ticks = add_border(bar_with_ticks)

        width = bar_with_ticks.shape[1]
        head = header_strip(width, label)
        foot = footer_strip(width, "")
        block = np.concatenate([head, bar_with_ticks, foot], axis=0)

        # Guard against off-by-one rounding so concatenate never errors.
        if block.shape[0] != total_height:
            if block.shape[0] > total_height:
                block = block[:total_height]
            else:
                pad_h = total_height - block.shape[0]
                pad = np.full((pad_h, width, 3), PANEL_BG, dtype=np.uint8)
                block = np.concatenate([block, pad], axis=0)
        return block

    n_total = len(val_loader)
    n_done = 0

    try:
        with torch.no_grad():
            for i, data_batch in enumerate(val_loader):
                if args.limit and i >= args.limit:
                    break

                # data_batch is a dict from PackDetInputs:
                # {'inputs': [tensor(C,H,W)], 'data_samples': [DetDataSample]}
                data_batch = model.data_preprocessor(data_batch, training=False)
                inputs = data_batch["inputs"]
                data_samples = data_batch["data_samples"]

                captured.clear()
                model.extract_feat(inputs, data_samples)

                gates = captured.get("gates", [])
                alphas = captured.get("alphas", [])
                if not gates:
                    print(f"[{i+1}/{n_total}] no gates captured — model has no DepthGate?")
                    continue

                ds = data_samples[0]
                img_path = ds.metainfo.get("img_path", f"sample_{i:06d}")
                stem = osp.splitext(osp.basename(img_path))[0]

                # ---- summary csv ----
                for lvl, (g_map, a_map) in enumerate(zip(gates, alphas)):
                    writer.writerow([
                        stem, lvl + 1,
                        f"{g_map.mean():.4f}", f"{g_map.std():.4f}",
                        f"{a_map.mean():.4f}", f"{a_map.std():.4f}",
                    ])

                # ---- raw arrays ----
                # Gate/alpha maps live at different FPN strides so their
                # spatial shapes differ per level (e.g. 96x96, 48x48, 24x24
                # for stride 8/16/32). Store them under per-level keys
                # instead of stacking, which requires identical shapes.
                npz_path = osp.join(args.out_dir, f"{stem}_maps.npz")
                npz_payload = {}
                for lvl, (g_map, a_map) in enumerate(zip(gates, alphas)):
                    npz_payload[f"gate_L{lvl + 1}"] = g_map
                    npz_payload[f"alpha_L{lvl + 1}"] = a_map
                np.savez_compressed(npz_path, **npz_payload)

                if not args.no_overlay:
                    # Recover the (padded) RGB tensor as uint8 for overlay.
                    x = inputs[0].detach().cpu()
                    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
                    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
                    img_pad = (x * std + mean).clamp(0, 255).byte().permute(1, 2, 0).numpy()
                    nh, nw = img_pad.shape[:2]

                    # Recover depth (after ResizeDepth, before model perturb).
                    depth_tensor = ds.metainfo.get("depth", None)
                    if depth_tensor is not None:
                        depth_pad = depth_tensor.detach().cpu().numpy()
                        if depth_pad.ndim == 3:
                            depth_pad = depth_pad[0]
                        if depth_pad.shape != (nh, nw):
                            depth_pad = cv2.resize(depth_pad, (nw, nh),
                                                   interpolation=cv2.INTER_LINEAR)
                    else:
                        depth_pad = np.zeros((nh, nw), dtype=np.float32)

                    depth_heat = to_heatmap(depth_pad, cv2.COLORMAP_INFERNO, (nw, nh))

                    # ---------- Build paper-style canvas ----------
                    # Row 1: [RGB] [Depth] [g overlay L1] [g overlay L2] [g overlay L3]
                    # Row 2: [RGB] [Depth] [(1-g)a L1]    [(1-g)a L2]    [(1-g)a L3]
                    # Colorbar on the far right of each row.
                    row1_panels = [
                        make_panel(img_pad,
                                   "RGB", "input", "image"),
                        make_panel(depth_heat,
                                   "Depth", "DAv2", "normalized"),
                    ]
                    for lvl, g_map in enumerate(gates, start=1):
                        heat = to_heatmap(g_map, "coolwarm", (nw, nh))
                        overlay = blend(img_pad, heat, alpha=0.55)
                        title = f"Gate g  L{lvl}"
                        sub = f"stride {2 ** (lvl + 2)}"
                        foot = f"mean={g_map.mean():.3f}   std={g_map.std():.3f}"
                        row1_panels.append(make_panel(overlay, title, sub, foot))

                    row2_panels = [
                        make_panel(img_pad,
                                   "RGB", "input", "image"),
                        make_panel(depth_heat,
                                   "Depth", "DAv2", "normalized"),
                    ]
                    for lvl, (g_map, a_map) in enumerate(zip(gates, alphas), start=1):
                        contrib = (1.0 - g_map) * a_map
                        heat = to_heatmap(contrib, cv2.COLORMAP_INFERNO, (nw, nh))
                        overlay = blend(img_pad, heat, alpha=0.6)
                        title = f"(1-g)·a  L{lvl}"
                        sub = "depth contribution"
                        foot = f"mean={contrib.mean():.3f}   max={contrib.max():.3f}"
                        row2_panels.append(make_panel(overlay, title, sub, foot))

                    # Stitch rows horizontally with small spacers between panels.
                    def hstitch(panels):
                        h = panels[0].shape[0]
                        spacer = np.full((h, GAP, 3), PANEL_BG, dtype=np.uint8)
                        out = panels[0]
                        for p in panels[1:]:
                            out = np.concatenate([out, spacer, p], axis=1)
                        return out

                    row1 = hstitch(row1_panels)
                    row2 = hstitch(row2_panels)

                    # Colorbars matched to row content (gate=coolwarm, contrib=inferno).
                    cb_g = vertical_colorbar(row1.shape[0], "coolwarm",
                                             vmin=0.0, vmax=1.0, label="g")
                    cb_c = vertical_colorbar(row2.shape[0], cv2.COLORMAP_INFERNO,
                                             vmin=0.0, vmax=1.0, label="(1-g)a")

                    cb_gap = np.full((row1.shape[0], GAP, 3), PANEL_BG, dtype=np.uint8)
                    row1_full = np.concatenate([row1, cb_gap, cb_g], axis=1)
                    cb_gap2 = np.full((row2.shape[0], GAP, 3), PANEL_BG, dtype=np.uint8)
                    row2_full = np.concatenate([row2, cb_gap2, cb_c], axis=1)

                    # Pad shorter row to width of the longer one.
                    target_w = max(row1_full.shape[1], row2_full.shape[1])
                    def pad_right(arr, w):
                        if arr.shape[1] >= w:
                            return arr
                        pad = np.full((arr.shape[0], w - arr.shape[1], 3),
                                      PANEL_BG, dtype=np.uint8)
                        return np.concatenate([arr, pad], axis=1)
                    row1_full = pad_right(row1_full, target_w)
                    row2_full = pad_right(row2_full, target_w)

                    # Vertical gap between rows + title strip on top.
                    v_gap = np.full((GAP * 2, target_w, 3), PANEL_BG, dtype=np.uint8)
                    title_h = 44
                    title_strip = np.full((title_h, target_w, 3), PANEL_BG, dtype=np.uint8)
                    title_text = f"DepthGate visualization — {stem}"
                    cv2.putText(title_strip, title_text, (16, 30),
                                cv2.FONT_HERSHEY_DUPLEX, 0.78, TEXT_COLOR, 1, cv2.LINE_AA)
                    sub_text = ("Row 1: per-level gate g  (blue = trust RGB, "
                                "red = trust depth).   "
                                "Row 2: effective depth contribution (1-g)·a")
                    sub_strip = np.full((24, target_w, 3), PANEL_BG, dtype=np.uint8)
                    cv2.putText(sub_strip, sub_text, (16, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, SUBTLE_COLOR, 1, cv2.LINE_AA)

                    canvas = np.concatenate(
                        [title_strip, sub_strip, row1_full, v_gap, row2_full],
                        axis=0,
                    )

                    # Outer margin for a clean printed look.
                    margin = 18
                    canvas = cv2.copyMakeBorder(canvas, margin, margin, margin, margin,
                                                cv2.BORDER_CONSTANT, value=PANEL_BG)

                    out_path = osp.join(args.out_dir, f"{stem}_gate.jpg")
                    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 95])

                n_done += 1
                if n_done % 10 == 0 or n_done == n_total:
                    print(f"[{n_done}/{n_total}] last: {stem}")
    finally:
        DepthGate.forward = orig_forward
        csv_f.close()

    print(f"\nfinished — {n_done} samples")
    print(f"  per-image jpg + npz: {args.out_dir}")
    print(f"  summary csv:         {csv_path}")


if __name__ == "__main__":
    main()
