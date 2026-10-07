"""Ridgeline ("joyplot") of the pooled per-pixel gate distributions —
CVPR / NeurIPS figure template (line + marker + confidence band, shared legend).

Three columns  =  three gate quantities
    g            : RGB-branch gate confidence
    alpha        : depth-branch attention
    (1-g)·alpha  : effective depth contribution

Each panel shows one distribution curve per (dataset × FPN-level) combination,
with a ±1 std confidence band, plotted on a light-grey grid background that
matches the reference figure style.

Auto-discovers every sub-directory  gate_vis_val_*/  that contains *_maps.npz.

Usage
-----
    python compare_gate_distributions.py evaluate_gate_val
    python compare_gate_distributions.py evaluate_gate_val \\
        --out evaluate_gate_val/gate_distributions.pdf \\
        --hist-stride 2 --bw 0.025
"""



import argparse
import glob
import os.path as osp

import numpy as np


# ── Palette (Okabe–Ito, print-safe) ──────────────────────────────────────────
# line colour  / fill colour (confidence band)
DATASET_STYLES: list[dict] = [
    {"color": "#3274A1", "fill": "#AEC6DE", "marker": "o",  "ms": 5.5},
    {"color": "#2D9B4E", "fill": "#A4D4B0", "marker": "s",  "ms": 5.0},
    {"color": "#C8511B", "fill": "#F0B89A", "marker": "^",  "ms": 5.5},
    {"color": "#8B4FA8", "fill": "#CBA8D8", "marker": "D",  "ms": 4.5},
    {"color": "#B8860B", "fill": "#E8D08A", "marker": "v",  "ms": 5.0},
]

# FPN level → linestyle
LEVEL_STYLE: dict[int, str] = {
    1: "-",
    2: "--",
    3: ":",
}


# ── rcParams (template style) ─────────────────────────────────────────────────
def _setup_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        # font
        "font.family":              "sans-serif",
        "font.sans-serif":          ["Helvetica Neue", "Helvetica", "Arial",
                                     "DejaVu Sans"],
        "font.size":                9.5,
        "mathtext.fontset":         "stixsans",

        # axes — all four spines on, matching reference
        "axes.linewidth":           0.8,
        "axes.edgecolor":           "#888888",
        "axes.facecolor":           "#EAEAF4",   # seaborn-style light-grey bg
        "axes.spines.top":          True,
        "axes.spines.right":        True,
        "axes.spines.left":         True,
        "axes.spines.bottom":       True,
        "axes.titlesize":           10.5,
        "axes.titleweight":         "bold",
        "axes.titlepad":            6,
        "axes.labelsize":           9.5,
        "axes.labelpad":            3,
        "axes.labelcolor":          "#222222",

        # grid — white lines on grey background (reference style)
        "axes.grid":                True,
        "grid.color":               "white",
        "grid.linewidth":           0.8,
        "grid.alpha":               1.0,
        "axes.axisbelow":           True,

        # ticks — outward, both axes
        "xtick.direction":          "out",
        "ytick.direction":          "out",
        "xtick.labelsize":          9.0,
        "ytick.labelsize":          9.0,
        "xtick.major.size":         4.0,
        "ytick.major.size":         4.0,
        "xtick.major.width":        0.8,
        "ytick.major.width":        0.8,
        "xtick.color":              "#444444",
        "ytick.color":              "#444444",

        # legend
        "legend.fontsize":          9.0,
        "legend.frameon":           False,
        "legend.handlelength":      2.0,
        "legend.handleheight":      1.2,
        "legend.handletextpad":     0.5,
        "legend.columnspacing":     1.4,
        "legend.borderpad":         0.0,

        # lines / markers
        "lines.linewidth":          1.6,
        "lines.markersize":         5.5,
        "lines.markeredgewidth":    1.0,

        # figure
        "figure.facecolor":         "white",
        "savefig.facecolor":        "white",
        "savefig.dpi":              300,
        "savefig.bbox":             "tight",
        "pdf.fonttype":             42,
        "ps.fonttype":              42,
    })


# ── Helpers ───────────────────────────────────────────────────────────────────
def pretty_name(dirname: str) -> str:
    name = dirname.replace("gate_vis_val_", "").replace("gate_vis_", "")
    overrides = {"forestseg": "ForestSeg", "quebec": "Quebec", "bam": "BAM"}
    return overrides.get(name.lower(), name)


def discover_datasets(root: str) -> list[tuple[str, list[str]]]:
    datasets = []
    for d in sorted(glob.glob(osp.join(root, "gate_vis_val_*"))):
        if not osp.isdir(d):
            continue
        npz = sorted(glob.glob(osp.join(d, "*_maps.npz")))
        if npz:
            datasets.append((pretty_name(osp.basename(d)), npz))
    return datasets


def accumulate_densities(
    npz_paths: list[str],
    bins: int,
    stride: int,
    bw: float,
    edges: np.ndarray,
) -> tuple[list[int], dict]:
    """Return smoothed PDFs and per-bin std for each (level, metric)."""
    from scipy.ndimage import gaussian_filter1d

    first  = np.load(npz_paths[0])
    levels = sorted(
        int(k.split("_L")[1])
        for k in first.files
        if k.startswith("gate_L")
    )

    binw  = edges[1] - edges[0]
    # accumulate histogram counts and squared-counts for variance
    hist  = {lv: {k: np.zeros(bins) for k in "gac"} for lv in levels}
    hist2 = {lv: {k: np.zeros(bins) for k in "gac"} for lv in levels}
    cnt   = {lv: 0 for lv in levels}

    for p in npz_paths:
        d = np.load(p)
        for lv in levels:
            g = d[f"gate_L{lv}"].ravel()[::stride]
            a = d[f"alpha_L{lv}"].ravel()[::stride]
            c = (1.0 - g) * a
            for key, arr in (("g", g), ("a", a), ("c", c)):
                h, _ = np.histogram(arr, bins=bins, range=(0, 1))
                hist [lv][key] += h
                hist2[lv][key] += h * h
            cnt[lv] += g.size

    sigma = max(bw / binw, 0.5)
    out: dict = {}
    for lv in levels:
        out[lv] = {}
        for key in "gac":
            h   = hist[lv][key].astype(float)
            tot = h.sum()
            pdf = h / (tot * binw) if tot > 0 else h

            # approximate per-bin std from file-to-file variation
            if len(npz_paths) > 1:
                h2  = hist2[lv][key].astype(float)
                n   = len(npz_paths)
                var = np.maximum((h2 / n) - (h / n) ** 2, 0.0)
                std = np.sqrt(var) / (tot / n * binw + 1e-12)
            else:
                std = pdf * 0.08   # fallback: 8 % envelope

            pdf = gaussian_filter1d(pdf, sigma=sigma, mode="nearest")
            std = gaussian_filter1d(std, sigma=sigma, mode="nearest")
            out[lv][key] = (pdf, std)

    return levels, out


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root",          nargs="?", default="evaluate_gate_val")
    ap.add_argument("--out",         default=None)
    ap.add_argument("--hist-stride", type=int,   default=2)
    ap.add_argument("--bins",        type=int,   default=400)
    ap.add_argument("--bw",          type=float, default=0.025)
    args = ap.parse_args()

    out_path = args.out or osp.join(args.root, "gate_distributions.pdf")

    datasets = discover_datasets(args.root)
    if not datasets:
        raise SystemExit(
            f"No gate_vis_val_*/*_maps.npz found under '{args.root}'"
        )

    _setup_style()

    import matplotlib.pyplot as plt
    from matplotlib.lines   import Line2D
    from matplotlib.patches import Patch

    styles = {
        name: DATASET_STYLES[i % len(DATASET_STYLES)]
        for i, (name, _) in enumerate(datasets)
    }

    edges   = np.linspace(0, 1, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # ── Accumulate ───────────────────────────────────────────────────────────
    per_ds: dict    = {}
    all_levels: set = set()
    for name, npz in datasets:
        print(f"  pooling '{name}'  ({len(npz)} files) …")
        lvls, dens = accumulate_densities(
            npz, args.bins, args.hist_stride, args.bw, edges
        )
        per_ds[name]  = dens
        all_levels   |= set(lvls)

    levels = sorted(all_levels)
    n_lvl  = len(levels)

    # ── Column definitions ───────────────────────────────────────────────────
    metrics = [
        ("g",  r"$g$  (RGB gate)",                    "Gate value"),
        ("a",  r"$\alpha$  (Depth attention)",         "Gate value"),
        ("c",  r"$(1{-}g)\cdot\alpha$  (Eff. depth)", "Gate value"),
    ]

    # ── Figure ───────────────────────────────────────────────────────────────
    # Wide figure: ~14 in = good for 3 large panels side-by-side
    FIG_W = 14.0
    FIG_H =  4.2
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), sharey=False)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.20,
                        top=0.82,  wspace=0.28)

    # ── Plot each panel ───────────────────────────────────────────────────────
    for col, (key, title, xlabel) in enumerate(metrics):
        ax = axes[col]

        for name, _ in datasets:
            st = styles[name]
            for lv in levels:
                if lv not in per_ds[name]:
                    continue
                pdf, std = per_ds[name][lv][key]
                ls       = LEVEL_STYLE.get(lv, "-")

                # confidence band ± 1 std
                ax.fill_between(
                    centers,
                    np.maximum(pdf - std, 0),
                    pdf + std,
                    color=st["fill"],
                    alpha=0.40,
                    linewidth=0,
                    zorder=2,
                )

                # main density line with markers every N bins
                marker_step = max(1, len(centers) // 10)
                ax.plot(
                    centers, pdf,
                    color     = st["color"],
                    lw        = 1.6,
                    ls        = ls,
                    marker    = st["marker"],
                    markersize= st["ms"],
                    markevery = marker_step,
                    markeredgecolor = st["color"],
                    markeredgewidth = 0.8,
                    markerfacecolor = "white" if ls == "--" else st["color"],
                    zorder    = 3,
                    clip_on   = True,
                )

        # reference line g = 0.5 only for the g panel
        if key == "g":
            ax.axvline(0.5, color="#666666", lw=0.9,
                       ls=(0, (4, 3)), zorder=1, label=r"$g=0.5$")

        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel, fontsize=9.5, labelpad=4)
        ax.set_xticks(np.linspace(0, 1, 6))

        if col == 0:
            ax.set_ylabel("Probability density", fontsize=9.5, labelpad=4)
        else:
            ax.set_ylabel("")

        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=7)

        # panel tag
        ax.text(
            0.01, 0.98, f"({'abc'[col]})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11, fontweight="bold", color="#111111",
        )

    # ── Shared legend (top-centre, above all panels) ──────────────────────────
    legend_handles = []

    for name, _ in datasets:
        st = styles[name]
        legend_handles.append(
            Line2D(
                [0], [0],
                color      = st["color"],
                lw         = 1.6,
                marker     = st["marker"],
                markersize = st["ms"],
                markeredgecolor = st["color"],
                markeredgewidth = 0.8,
                markerfacecolor = st["color"],
                label      = name,
            )
        )

    # Linestyle entries for FPN levels
    ls_labels = {"-": "solid (L1)", "--": "dashed (L2)", ":": "dotted (L3)"}
    for lv in levels:
        ls = LEVEL_STYLE.get(lv, "-")
        legend_handles.append(
            Line2D(
                [0], [0],
                color="#444444", lw=1.6, ls=ls,
                label=rf"$L_{lv}$  (stride {2**(lv+2)})",
            )
        )

    legend_handles.append(
        Line2D([0], [0], color="#666666", lw=0.9, ls=(0, (4, 3)),
               label=r"$g = 0.5$ (balanced)")
    )

    fig.legend(
        handles        = legend_handles,
        loc            = "upper center",
        ncol           = len(legend_handles),
        bbox_to_anchor = (0.52, 1.00),
        frameon        = False,
        fontsize       = 9.0,
        handlelength   = 2.0,
        handletextpad  = 0.5,
        columnspacing  = 1.6,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(out_path)
    if out_path.lower().endswith(".pdf"):
        png_path = out_path[:-4] + ".png"
        fig.savefig(png_path, dpi=300)
        print(f"Saved PNG : {png_path}")
    elif out_path.lower().endswith(".png"):
        pdf_path = out_path[:-4] + ".pdf"
        fig.savefig(pdf_path)
        print(f"Saved PDF : {pdf_path}")

    plt.close(fig)
    print(f"Saved     : {out_path}")
    print(f"Datasets  : {', '.join(n for n, _ in datasets)}")
    print(f"FPN levels: {levels}")


if __name__ == "__main__":
    main()