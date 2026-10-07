"""Plot robustness curves from tools/eval/test_robustness.py output JSON.

Produces robustness_noise.png and robustness_dropout.png.

Usage:
    python tools/eval/plot_robustness.py robustness.json --out-dir figs/
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("json", help="robustness.json from test_robustness.py")
    p.add_argument("--out-dir", default="figs")
    p.add_argument("--metric", default="coco/segm_mAP")
    return p.parse_args()


def main():
    args = parse_args()
    import matplotlib.pyplot as plt
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.json) as f:
        data = json.load(f)

    for k, xlabel in [("noise", "depth noise std"),
                      ("dropout", "depth dropout prob")]:
        if k not in data or not data[k]:
            continue
        xs = [e["std"] if k == "noise" else e["prob"] for e in data[k]]
        ys = [e["metrics"].get(args.metric, float("nan")) for e in data[k]]
        plt.figure(figsize=(4.5, 3.2))
        plt.plot(xs, ys, marker="o")
        plt.xlabel(xlabel)
        plt.ylabel(args.metric)
        plt.title(f"Robustness to {k}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = osp.join(args.out_dir, f"robustness_{k}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"saved {out}")


if __name__ == "__main__":
    main()
