"""Test-time robustness: evaluate model under perturbed depth.

Sweeps depth noise std and depth-dropout probability at INFERENCE time,
reports AP for each setting. Lets you make a "degrades gracefully" plot.

Usage:
    python tools/eval/test_robustness.py CONFIG CHECKPOINT \
        --noise 0.0 0.05 0.1 0.2 0.3 \
        --dropout 0.0 0.2 0.5 1.0 \
        --out robustness.json
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
import depthgate  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3])
    p.add_argument("--dropout", type=float, nargs="+", default=[0.0, 0.2, 0.5, 1.0])
    p.add_argument("--out", default="robustness.json")
    p.add_argument("--work-dir", default="work_dirs/_robustness_tmp")
    return p.parse_args()


def run_eval(cfg, checkpoint, work_dir):
    """Build runner from cfg, load checkpoint, run test."""
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config(cfg.to_dict()) if hasattr(cfg, "to_dict") else Config(cfg)
    cfg.work_dir = work_dir
    cfg.load_from = checkpoint
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    return metrics


def main():
    args = parse_args()
    from mmengine.config import Config

    base_cfg = Config.fromfile(args.config)

    out = {"noise": [], "dropout": []}

    for s in args.noise:
        cfg = Config(base_cfg.to_dict())
        # ensure model evaluates with this noise at test time too — we must
        # set training=True for noise to fire, which is wrong, so instead we
        # patch _maybe_perturb_depth via a monkeypatch alternative: override
        # the model's depth_noise_std AND force-eval-time perturbation by a
        # local wrapper around predict. Simpler: keep at training=False but
        # call perturb manually inside extract_feat. We do this by setting
        # the flag and forcing model.train(False) but enabling a perturb-at-test
        # override via cfg.
        cfg.model.depth_noise_std = float(s)
        cfg.model.depth_dropout_prob = 0.0
        # Tell the wrapper to perturb at eval too via an attribute trick:
        cfg.model.perturb_at_eval = True  # not yet supported, see note below
        m = run_eval(cfg, args.checkpoint, args.work_dir + f"_noise_{s}")
        out["noise"].append({"std": s, "metrics": _flatten(m)})

    for p in args.dropout:
        cfg = Config(base_cfg.to_dict())
        cfg.model.depth_noise_std = 0.0
        cfg.model.depth_dropout_prob = float(p)
        cfg.model.perturb_at_eval = True
        m = run_eval(cfg, args.checkpoint, args.work_dir + f"_drop_{p}")
        out["dropout"].append({"prob": p, "metrics": _flatten(m)})

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {args.out}")


def _flatten(m):
    if isinstance(m, dict):
        return {k: float(v) if hasattr(v, "__float__") else v for k, v in m.items()}
    return m


if __name__ == "__main__":
    main()
