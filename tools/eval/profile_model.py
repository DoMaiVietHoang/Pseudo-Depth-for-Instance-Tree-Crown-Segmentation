"""Report params, FLOPs, and latency for a config — for the efficiency table.

Usage:
    python tools/eval/profile_model.py CONFIG [--device cuda:0] [--repeats 50] [--input-size 768]
"""

from __future__ import annotations

import argparse
import os.path as osp
import sys
import time

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
import depthgate  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--input-size", type=int, default=768)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    return p.parse_args()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def try_flops(model, h, w, device):
    try:
        from mmengine.analysis import get_model_complexity_info
        try:
            res = get_model_complexity_info(model, input_shape=(3, h, w))
            return res.get("flops"), res.get("params")
        except Exception:
            pass
    except Exception:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        import torch
        x = torch.randn(1, 3, h, w, device=device)
        fca = FlopCountAnalysis(model, x)
        return fca.total(), None
    except Exception as e:
        return None, str(e)


def measure_latency(model, h, w, device, warmup, repeats):
    import torch
    x = torch.randn(1, 3, h, w, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            try:
                _ = model.extract_feat(x)
            except Exception:
                _ = model.backbone(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repeats):
            try:
                _ = model.extract_feat(x)
            except Exception:
                _ = model.backbone(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = (time.time() - t0) / repeats
    return dt


def main():
    args = parse_args()
    from mmengine.config import Config
    from mmengine.registry import MODELS

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    model.to(args.device)

    total, trainable = count_params(model)
    print(f"# {args.config}")
    print(f"params_total      = {total / 1e6:.2f} M")
    print(f"params_trainable  = {trainable / 1e6:.2f} M")

    flops, _ = try_flops(model, args.input_size, args.input_size, args.device)
    if isinstance(flops, (int, float)):
        print(f"flops             = {flops / 1e9:.2f} G  (input {args.input_size}x{args.input_size})")
    else:
        print(f"flops             = N/A ({flops})")

    try:
        dt = measure_latency(model, args.input_size, args.input_size,
                             args.device, args.warmup, args.repeats)
        print(f"latency (backbone+neck+fusion) = {dt * 1000:.2f} ms / sample")
    except Exception as e:
        print(f"latency           = N/A ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
