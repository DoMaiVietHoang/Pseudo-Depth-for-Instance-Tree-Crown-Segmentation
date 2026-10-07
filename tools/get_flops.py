"""Measure parameter count and FLOPs of a (fusion-ablation) config.

The DepthGate models receive depth through `batch_data_samples` metainfo,
so the stock mmdet get_flops path (which calls `extract_feat(batch_inputs)`
alone) would skip the depth encoder / projection / fusion modules entirely.
Here we build a dummy DetDataSample carrying a depth map and route the
forward through the same fused path used at train/test time.

RGB baselines (plain `Mask2Former`) inherit `SingleStageDetector.extract_feat`,
which takes `batch_inputs` only. We inspect the signature and pass the samples
just to the models that accept them, so RGB and DepthGate configs can be
measured side by side in one run.

Usage:
    python tools/get_flops.py configs/ablations/fusion_concat.py
    python tools/get_flops.py configs/ablations/*.py --shape 768 768
    python tools/get_flops.py configs/*.py configs/ablations/*.py
"""

from __future__ import annotations

import argparse
import inspect
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
from mmengine.analysis import get_model_complexity_info
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample

import depthgate  # noqa: F401  (registers Mask2FormerDepthGate)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("configs", nargs="+", help="config file(s)")
    p.add_argument("--shape", type=int, nargs=2, default=[768, 768],
                   metavar=("H", "W"), help="input resolution (default 768 768)")
    p.add_argument("--table", action="store_true",
                   help="also print the per-module complexity table")
    return p.parse_args()


def measure(cfg_path: str, h: int, w: int, show_table: bool):
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get("default_scope", "mmdet"))

    model = MODELS.build(cfg.model)
    model.eval()

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    batch_inputs = torch.zeros(1, 3, h, w)
    sample = DetDataSample()
    sample.set_metainfo({
        "img_shape": (h, w),
        "ori_shape": (h, w),
        "batch_input_shape": (h, w),
        "depth": torch.zeros(1, h, w),
    })
    samples = [sample]

    # Depth reaches the model only via batch_data_samples, and only the
    # DepthGate subclass accepts them; stock Mask2Former is RGB-only.
    takes_samples = len(
        inspect.signature(model.extract_feat).parameters) > 1

    def forward(inputs):
        if takes_samples:
            feats = model.extract_feat(inputs, samples)
        else:
            feats = model.extract_feat(inputs)
        return model.panoptic_head.forward(feats, samples)

    original_forward = model.forward
    model.forward = forward
    try:
        analysis = get_model_complexity_info(
            model, inputs=batch_inputs,
            show_table=show_table, show_arch=False,
        )
    finally:
        model.forward = original_forward

    print("=" * 72)
    print(f"Config     : {cfg_path}")
    if takes_samples:
        print(f"Model      : {cfg.model['type']}"
              f"  (fusion_type={cfg.model.get('fusion_type', 'depthgate')})")
        print(f"Input      : rgb 1x3x{h}x{w} + depth 1x1x{h}x{w}")
    else:
        print(f"Model      : {cfg.model['type']}  (RGB only)")
        print(f"Input      : rgb 1x3x{h}x{w}")
    print(f"FLOPs      : {analysis['flops_str']}")
    print(f"Params     : {analysis['params_str']}"
          f"  (total {n_total:,} | trainable {n_train:,})")
    if show_table:
        print(analysis["out_table"])
    return cfg_path, analysis["flops_str"], n_total, n_train


def main():
    args = parse_args()
    rows = [measure(c, *args.shape, args.table) for c in args.configs]
    if len(rows) > 1:
        print("\n" + "=" * 72)
        print(f"{'config':<48} {'FLOPs':>10} {'params':>13} {'trainable':>13}")
        for cfg_path, flops, total, train in rows:
            print(f"{cfg_path:<48} {flops:>10} {total:>13,} {train:>13,}")


if __name__ == "__main__":
    main()
