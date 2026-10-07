"""Benchmark Depth-Anything-V2 depth-generation throughput (FPS).

DAv2 is the preprocessing stage of the DepthGate pipeline: tools/generate_depth.py
runs it offline and writes .npy maps that training/eval later read from disk. This
script measures that stage's cost on its own -- how fast depth maps can be produced,
independent of any Mask2Former.

Weights are read from the local HF cache (dav_base_pretrained/hub) by default, so
no download is attempted. Only the sizes actually present in that cache can run.

Timing method: CUDA-synchronised wall clock around the model forward, after warmup
iterations that absorb cuDNN autotuning and lazy CUDA init. Reports median (robust
to scheduler hiccups) alongside mean/std.

Usage:
    python tools/bench_dav2_fps.py
    python tools/bench_dav2_fps.py --shape 1024 1024 --batch-size 4
    python tools/bench_dav2_fps.py --model dav2_base --iters 100 --include-preprocess
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import statistics
import sys
import time

_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Point HF at the vendored cache before transformers is imported, so a missing
# model fails loudly offline instead of silently pulling from the hub.
os.environ.setdefault("HF_HOME", osp.join(_ROOT, "dav_base_pretrained"))

import numpy as np
import torch

_HUB_IDS = {
    "dav2_small": "depth-anything/Depth-Anything-V2-Small-hf",
    "dav2_base": "depth-anything/Depth-Anything-V2-Base-hf",
    "dav2_large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", default=["dav2_base"],
                   choices=list(_HUB_IDS), help="DAv2 size(s) to benchmark")
    p.add_argument("--shape", type=int, nargs=2, default=[768, 768],
                   metavar=("H", "W"), help="input resolution (default 768 768)")
    p.add_argument("--batch-size", type=int, default=1,
                   help="images per forward (default 1; FPS is per image)")
    p.add_argument("--iters", type=int, default=50, help="timed iterations")
    p.add_argument("--warmup", type=int, default=10, help="untimed warmup iterations")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp16", action="store_true", help="autocast to float16")
    p.add_argument("--include-preprocess", action="store_true",
                   help="also time the HF image processor (resize/normalise on CPU), "
                        "which generate_depth.py pays per image in practice")
    return p.parse_args()


def load(name, device):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    hub_id = _HUB_IDS[name]
    proc = AutoImageProcessor.from_pretrained(hub_id)
    model = AutoModelForDepthEstimation.from_pretrained(hub_id).to(device).eval()
    return proc, model


def bench(name, h, w, args):
    device = torch.device(args.device)
    try:
        proc, model = load(name, device)
    except Exception as e:  # noqa: BLE001 - report and skip, don't kill the sweep
        print(f"[skip] {name}: {type(e).__name__}: {str(e)[:160]}")
        return None

    n_params = sum(p.numel() for p in model.parameters())
    bs = args.batch_size

    # Random uint8 frames stand in for tiles; content does not affect timing.
    raw = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(bs)]
    with torch.no_grad():
        pixel_values = proc(images=raw, return_tensors="pt")["pixel_values"].to(device)
    in_hw = tuple(pixel_values.shape[-2:])

    def once():
        if args.include_preprocess:
            pv = proc(images=raw, return_tensors="pt")["pixel_values"].to(device)
        else:
            pv = pixel_values
        with torch.autocast("cuda", torch.float16, enabled=args.fp16):
            model(pixel_values=pv)

    with torch.no_grad():
        for _ in range(args.warmup):
            once()
        torch.cuda.synchronize(device)

        times = []
        for _ in range(args.iters):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            once()
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)

    per_img = [t / bs for t in times]
    med = statistics.median(per_img)
    mean = statistics.mean(per_img)
    std = statistics.stdev(per_img) if len(per_img) > 1 else 0.0
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    torch.cuda.reset_peak_memory_stats(device)

    print("=" * 72)
    print(f"Model      : {name}  ({_HUB_IDS[name]})")
    print(f"Params     : {n_params:,}")
    print(f"Input      : {bs}x3x{h}x{w}  -> processor resized to {in_hw[0]}x{in_hw[1]}")
    print(f"Precision  : {'fp16 autocast' if args.fp16 else 'fp32'}"
          f"   preprocess timed: {args.include_preprocess}")
    print(f"Latency    : {med*1000:.2f} ms/img (median)"
          f"  | mean {mean*1000:.2f} +/- {std*1000:.2f}")
    print(f"FPS        : {1.0/med:.2f}  (median)")
    print(f"Peak VRAM  : {peak:.2f} GiB")
    return name, n_params, med, 1.0 / med, in_hw


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA not available; FPS on CPU would not be meaningful here.")

    rows = [r for r in (bench(m, *args.shape, args) for m in args.model) if r]
    if len(rows) > 1:
        print("\n" + "=" * 72)
        print(f"{'model':<14} {'params':>14} {'ms/img':>9} {'FPS':>8}  {'proc. input':>12}")
        for name, n, med, fps, in_hw in rows:
            print(f"{name:<14} {n:>14,} {med*1000:>9.2f} {fps:>8.2f}"
                  f"  {in_hw[0]}x{in_hw[1]:>7}")


if __name__ == "__main__":
    main()
