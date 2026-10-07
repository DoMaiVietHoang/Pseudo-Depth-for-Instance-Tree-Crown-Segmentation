"""Evaluation entrypoint."""

import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

import depthgate  # noqa: F401


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model on BAMFOREST")
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--work-dir")
    p.add_argument("--out", help="save predictions to this pkl path")
    p.add_argument("--cfg-options", nargs="+", action=DictAction)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.work_dir = args.work_dir or osp.join("work_dirs",
                                             osp.splitext(osp.basename(args.config))[0])
    cfg.load_from = args.checkpoint
    if args.out:
        cfg.test_evaluator = [cfg.test_evaluator,
                              dict(type="DumpDetResults", out_file_path=args.out)]
    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == "__main__":
    main()
