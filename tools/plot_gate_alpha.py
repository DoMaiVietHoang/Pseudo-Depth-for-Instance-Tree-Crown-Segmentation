"""Plot DepthGate gate/alpha training curves for one or more datasets.

Parses the lines emitted by ``LogDepthGateHook`` in an mmengine training log::

    ... [LogDepthGateHook] iter=39 gate/L1=0.500 | alpha/L1=0.506 | ...
        | gate/mean=0.500 | alpha/mean=0.503

and renders publication-style figures (serif fonts, muted colorblind-safe
palette, thin grid). Two outputs are produced:

    gate_alpha_per_dataset.{pdf,png}  -- grid: rows=datasets, cols=[gate, alpha],
                                          each cell showing the per-FPN-level
                                          curves L1/L2/L3 and their mean.
    gate_alpha_mean_compare.{pdf,png} -- gate/mean and alpha/mean overlaid
                                          across datasets vs training progress.

Usage::

    # auto-discover Log_gate_alpha_* files in a directory
    python tools/plot_gate_alpha.py --log-dir log_gate_alpha

    # or list logs explicitly with labels
    python tools/plot_gate_alpha.py \
        --log log_gate_alpha/Log_gate_alpha_forestseg.log:ForestSeg-T1 \
        --log log_gate_alpha/Log_gate_alpha_quebec.log:QuebecTree \
        --log log_gate_alpha/Log_gate_alpha_BAMFOREST:BAMFOREST
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
from glob import glob

import numpy as np

# Headless backend so it runs over SSH / without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_LINE_RE = re.compile(r"\[LogDepthGateHook\]\s+iter=(\d+)\s+(.*)")
_KV_RE = re.compile(r"([a-zA-Z]+/[A-Za-z0-9]+)=([-+0-9.eE]+)")

# Categorical palette: pre-validated slot order (CVD-safe as a set) from the
# dataviz design system -- blue / aqua / yellow / violet / red.
_DATASET_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]
_LEVEL_STYLES = {
    # level_key -> (color, linestyle); first three categorical slots, fixed order.
    "L1": ("#2a78d6", "-"),   # blue
    "L2": ("#1baf7a", "-"),   # aqua
    "L3": ("#eda100", "-"),   # yellow
}
_MEAN_STYLE = dict(color="black", linestyle="--", linewidth=1.6)
_INK = {"primary": "#0b0b0b", "secondary": "#52514e"}

# Default label mapping for the known logs; falls back to the cleaned stem.
_KNOWN_LABELS = {
    "forestseg": "ForestSeg-T1",
    "quebec": "QuebecTree",
    "bamforest": "BAMFOREST",
}


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "lines.linewidth": 1.3,
    })


def parse_log(path: str):
    """Return (iters: np.ndarray, data: dict[str, np.ndarray])."""
    iters: list[int] = []
    cols: dict[str, list[float]] = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = _LINE_RE.search(line)
            if not m:
                continue
            it = int(m.group(1))
            pairs = dict(_KV_RE.findall(m.group(2)))
            if not pairs:
                continue
            iters.append(it)
            for k, v in pairs.items():
                cols.setdefault(k, []).append(float(v))
    n = len(iters)
    # Right-pad any key that started appearing late so all arrays align by index.
    data = {}
    for k, vals in cols.items():
        if len(vals) == n:
            data[k] = np.asarray(vals, dtype=float)
    return np.asarray(iters, dtype=float), data


def _moving_average(x: np.ndarray, y: np.ndarray, window: int):
    if window <= 1 or len(y) < window:
        return x, y
    kernel = np.ones(window) / window
    ys = np.convolve(y, kernel, mode="valid")
    half = window - 1
    return x[half // 2: half // 2 + len(ys)], ys


def _auto_window(n: int, override: int) -> int:
    """Odd smoothing window: honour override>1, else ~1.5% of the series."""
    if override and override > 1:
        return override
    w = max(11, n // 60)
    return w + 1 if w % 2 == 0 else w


def _levels(data: dict, quantity: str) -> list[str]:
    """Sorted level keys (L1, L2, ...) present for 'gate' or 'alpha'."""
    levels = [k.split("/")[1] for k in data
              if k.startswith(f"{quantity}/") and k.split("/")[1] != "mean"]
    return sorted(levels, key=lambda s: (len(s), s))


def label_from_path(path: str) -> str:
    stem = osp.splitext(osp.basename(path))[0]
    key = stem.replace("Log_gate_alpha_", "").replace("log_gate_alpha_", "")
    return _KNOWN_LABELS.get(key.lower(), key)


def discover_logs(log_dir: str):
    files = sorted(glob(osp.join(log_dir, "Log_gate_alpha_*")))
    return [(p, label_from_path(p)) for p in files]


def plot_per_dataset(parsed, out_base: str, smooth: int):
    """Grid: rows = datasets, cols = [gate, alpha]; cells show per-level curves.

    Raw curves are drawn faint underneath a smoothed line so the trend is
    legible without hiding the underlying noise. One shared legend; a faint
    g=0.5 reference in the gate column marks the balanced initialisation.
    """
    nd = len(parsed)
    col_titles = (r"Gate $g$  (RGB trust)", r"Residual scale $\alpha$")
    fig, axes = plt.subplots(nd, 2, figsize=(8.6, 2.35 * nd + 0.7),
                             squeeze=False, sharex="row")
    seen = {}
    for r, (label, iters, data) in enumerate(parsed):
        for c, quantity in enumerate(("gate", "alpha")):
            ax = axes[r][c]
            if quantity == "gate":
                ax.axhline(0.5, color="#9a9a9a", ls=":", lw=0.9, zorder=0)
            for lv in _levels(data, quantity):
                color, _ = _LEVEL_STYLES.get(lv, ("#555555", "-"))
                y = data[f"{quantity}/{lv}"]
                w = _auto_window(len(y), smooth)
                xs, ys = _moving_average(iters, y, w)
                ax.plot(iters, y, color=color, lw=0.6, alpha=0.16, zorder=1)
                h, = ax.plot(xs, ys, color=color, lw=1.9, zorder=2,
                             solid_capstyle="round", label=lv)
                seen[lv] = h
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12, color=_INK["primary"])
            if c == 0:
                ax.set_ylabel(label, fontsize=12, fontweight="bold",
                              color=_INK["primary"], labelpad=8)
            if r == nd - 1:
                ax.set_xlabel("Training iteration", color=_INK["secondary"])
            ax.tick_params(colors=_INK["secondary"], labelsize=9)
            ax.margins(x=0.01)
    # single shared legend across the whole figure
    order = sorted(seen, key=lambda s: (len(s), s))
    fig.legend([seen[k] for k in order], order, title="FPN level",
               loc="upper center", ncol=len(order), frameon=False,
               bbox_to_anchor=(0.5, 1.005), fontsize=10, title_fontsize=10,
               columnspacing=1.6, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}")
    plt.close(fig)
    print(f"[plot] wrote {out_base}.pdf / .png")


def plot_mean_compare(parsed, out_base: str, smooth: int):
    """Overlay gate/mean and alpha/mean across datasets vs training progress."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), squeeze=False)
    titles = {"gate": r"Gate mean $\bar{g}$ (RGB$\rightarrow$1, depth$\rightarrow$0)",
              "alpha": r"Residual scale mean $\bar{\alpha}$"}
    for c, quantity in enumerate(("gate", "alpha")):
        ax = axes[0][c]
        mkey = f"{quantity}/mean"
        for i, (label, iters, data) in enumerate(parsed):
            if mkey not in data:
                continue
            color = _DATASET_COLORS[i % len(_DATASET_COLORS)]
            # normalize x to [0, 1] so datasets of different length align
            prog = iters / iters.max() if iters.max() > 0 else iters
            xs, ys = _moving_average(prog, data[mkey], _auto_window(len(iters), smooth))
            ax.plot(prog, data[mkey], color=color, lw=0.6, alpha=0.16, zorder=1)
            ax.plot(xs, ys, color=color, lw=1.9, zorder=2,
                    solid_capstyle="round", label=label)
        if quantity == "gate":
            ax.axhline(0.5, color="#9a9a9a", ls=":", lw=0.9, zorder=0)
        ax.set_title(titles[quantity], fontsize=11, color=_INK["primary"])
        ax.set_xlabel("Training progress", color=_INK["secondary"])
        ax.set_ylabel(mkey, color=_INK["secondary"])
        ax.set_xlim(0, 1)
        ax.tick_params(colors=_INK["secondary"], labelsize=9)
        ax.margins(x=0.01)
    axes[0][0].legend(loc="best", fontsize=9, title="dataset", title_fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}")
    plt.close(fig)
    print(f"[plot] wrote {out_base}.pdf / .png")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--log-dir", help="dir of Log_gate_alpha_* files to auto-load")
    src.add_argument("--log", action="append",
                     help="PATH or PATH:LABEL (repeatable)")
    p.add_argument("--out-dir", default=None,
                   help="where to write figures (default: alongside the logs)")
    p.add_argument("--smooth", type=int, default=1,
                   help="moving-average window over iters (1 = raw)")
    return p.parse_args()


def main():
    args = parse_args()
    _apply_style()

    if args.log_dir:
        logs = discover_logs(args.log_dir)
        default_out = args.log_dir
    else:
        logs = []
        for spec in args.log:
            if ":" in spec and not osp.exists(spec):
                path, label = spec.rsplit(":", 1)
            else:
                path, label = spec, label_from_path(spec)
            logs.append((path, label))
        default_out = osp.dirname(logs[0][0]) or "."

    if not logs:
        print("[plot] no logs found")
        return

    parsed = []
    for path, label in logs:
        iters, data = parse_log(path)
        if len(iters) == 0:
            print(f"[plot] WARNING: no LogDepthGateHook lines in {path}, skipping")
            continue
        print(f"[plot] {label:14s} {len(iters):5d} points, "
              f"iters {int(iters.min())}..{int(iters.max())}, "
              f"keys: {sorted(data)}")
        parsed.append((label, iters, data))

    if not parsed:
        print("[plot] nothing to plot")
        return

    out_dir = args.out_dir or default_out
    os.makedirs(out_dir, exist_ok=True)
    plot_per_dataset(parsed, osp.join(out_dir, "gate_alpha_per_dataset"), args.smooth)
    plot_mean_compare(parsed, osp.join(out_dir, "gate_alpha_mean_compare"), args.smooth)
    print("[plot] done.")


if __name__ == "__main__":
    main()
