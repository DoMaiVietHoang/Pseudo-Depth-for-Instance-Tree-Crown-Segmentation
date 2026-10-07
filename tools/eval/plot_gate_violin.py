"""Statistical summary of the pooled DepthGate distributions (paper Fig.).

Replaces the 9-overlapping-KDE plot with a clean dot-and-whisker (point-range)
figure: one panel per quantity (g, alpha, effective depth), x = FPN level, one
series per dataset showing the median (marker) and inter-quartile range
(whiskers), connected across levels to expose the level-dependent trend.

Reads the per-image gate maps written during eval:
    <vis_dir>/*_maps.npz   with keys gate_L{1,2,3}, alpha_L{1,2,3}

Usage:
    python tools/eval/plot_gate_violin.py
"""

from __future__ import annotations

import argparse
import glob
import os.path as osp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))

DEFAULT_DIRS = [
    ("QuebecTrees", "evaluate_gate_val/gate_vis_val_quebec"),
    ("ForestSeg",   "evaluate_gate_val/gate_vis_val_forestSeg"),
    ("BAMFOREST",   "evaluate_gate_val/gate_vis_val_bam"),
]
PALETTE = {"QuebecTrees": "#e07b39", "ForestSeg": "#2a9d4a",
           "BAMFOREST": "#2c6fbb"}
MARKER = {"QuebecTrees": "o", "ForestSeg": "s", "BAMFOREST": "^"}
LVL_NAME = {1: "P3\n(stride 8)", 2: "P4\n(stride 16)", 3: "P5\n(stride 32)"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", action="append", default=None,
                   metavar="NAME:PATH", help="dataset name:vis_dir (repeatable)")
    p.add_argument("--stride", type=int, default=4,
                   help="pixel subsample stride when pooling (memory)")
    p.add_argument("--band", default="iqr", choices=["iqr", "std"],
                   help="whiskers = inter-quartile range (default) or +-1 std")
    p.add_argument("--out", default=osp.join(REPO, "evaluate_gate_val",
                                             "gate_summary_ALL.png"))
    return p.parse_args()


def resolve_dirs(args):
    if not args.dir:
        return [(n, osp.join(REPO, p)) for n, p in DEFAULT_DIRS]
    out = []
    for spec in args.dir:
        name, path = spec.split(":", 1)
        out.append((name, path if osp.isabs(path) else osp.join(REPO, path)))
    return out


def pool_stats(vis_dir, stride, band):
    """Return {level: {metric: (center, lo, hi)}} pooled over all images."""
    files = sorted(glob.glob(osp.join(vis_dir, "*_maps.npz")))
    if not files:
        return None
    levels = sorted(int(k.split("_L")[1]) for k in np.load(files[0]).files
                    if k.startswith("gate_L"))
    acc = {(lvl, m): [] for lvl in levels for m in ("g", "a", "c")}
    for f in files:
        d = np.load(f)
        for lvl in levels:
            g = d[f"gate_L{lvl}"].ravel()[::stride]
            a = d[f"alpha_L{lvl}"].ravel()[::stride]
            acc[(lvl, "g")].append(g)
            acc[(lvl, "a")].append(a)
            acc[(lvl, "c")].append((1.0 - g) * a)
    stats = {lvl: {} for lvl in levels}
    for (lvl, m), chunks in acc.items():
        v = np.concatenate(chunks)
        if band == "iqr":
            q25, med, q75 = np.percentile(v, [25, 50, 75])
            stats[lvl][m] = (med, med - q25, q75 - med)
        else:
            mu, sd = float(v.mean()), float(v.std())
            stats[lvl][m] = (mu, sd, sd)
    return levels, stats


def main():
    args = parse_args()
    data = {}          # name -> (levels, stats)
    for name, vis_dir in resolve_dirs(args):
        res = pool_stats(vis_dir, args.stride, args.band)
        if res is None:
            print(f"[skip] {name}: no *_maps.npz in {vis_dir}")
            continue
        data[name] = res
        print(f"[{name}] pooled {len(glob.glob(osp.join(vis_dir, '*_maps.npz')))} images")
    if not data:
        raise SystemExit("no data pooled")

    levels = sorted(next(iter(data.values()))[0])
    ds_names = [n for n, _ in resolve_dirs(args) if n in data]
    x = np.arange(len(levels))
    dodge = (np.arange(len(ds_names)) - (len(ds_names) - 1) / 2) * 0.16

    plt.rcParams.update({
        "font.family": "serif", "font.size": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.35, "grid.linestyle": "--",
    })
    metrics = [("g", r"$g$   (RGB trust)"),
               ("a", r"$\alpha$   (depth attention)"),
               ("c", r"$(1-g)\,\alpha$   (effective depth use)")]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    for ax, (m, title) in zip(axes, metrics):
        for di, name in enumerate(ds_names):
            _, stats = data[name]
            cen = np.array([stats[l][m][0] for l in levels])
            lo = np.array([stats[l][m][1] for l in levels])
            hi = np.array([stats[l][m][2] for l in levels])
            ax.errorbar(
                x + dodge[di], cen, yerr=np.vstack([lo, hi]),
                fmt="-", marker=MARKER[name], ms=9, mfc=PALETTE[name],
                mec="white", mew=1.2, color=PALETTE[name], lw=2.0,
                elinewidth=1.6, capsize=5, capthick=1.6, label=name, zorder=3)
        if m == "g":
            ax.axhline(0.5, color="#666", ls=":", lw=1.4, zorder=1)
            ax.text(x[-1] + dodge[-1], 0.505, "balanced ($g{=}0.5$)",
                    ha="right", va="bottom", fontsize=11, color="#666")
        ax.set_title(title, fontsize=16, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([LVL_NAME.get(l, f"L{l}") for l in levels])
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_xlabel("feature-pyramid level")
    axes[0].set_ylabel("gate value  (median $\\pm$ IQR)"
                       if args.band == "iqr" else "value  (mean $\\pm$ s.d.)")
    axes[0].legend(title="dataset", frameon=True, framealpha=0.95,
                   loc="best", fontsize=11, title_fontsize=12)
    fig.suptitle("DepthGate gate statistics across feature-pyramid levels "
                 "(pooled per-pixel over each validation set)",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    fig.savefig(osp.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
