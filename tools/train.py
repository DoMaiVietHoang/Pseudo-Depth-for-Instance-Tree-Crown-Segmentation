"""Train entrypoint. Thin wrapper around mmdet's train so our custom_imports load."""

import argparse
import os
import os.path as osp
import sys

# Make the project root importable when running `python tools/train.py ...`
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# Force-register custom modules
import depthgate  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description="Train DepthGate / baseline on BAMFOREST")
    p.add_argument("config", help="path to config file")
    p.add_argument("--work-dir", help="override work directory")
    p.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    p.add_argument("--amp", action="store_true", help="enable mixed precision training")
    p.add_argument("--cfg-options", nargs="+", action=DictAction,
                   help="override config values, e.g. train_dataloader.batch_size=4")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join("work_dirs", osp.splitext(osp.basename(args.config))[0])
    if args.amp:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"
    if args.resume:
        cfg.resume = True

    os.makedirs(cfg.work_dir, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
