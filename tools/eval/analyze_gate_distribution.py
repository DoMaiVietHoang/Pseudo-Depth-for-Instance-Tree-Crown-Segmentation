"""Dataset-level visualization of DepthGate g, alpha, and their effect.

Run AFTER `visualize_gate_val.py` has produced:
    <vis_dir>/summary.csv
    <vis_dir>/<stem>_maps.npz          (per-image, per-level g & alpha)
    <vis_dir>/<stem>_gate.jpg          (optional, only for hero strips)

Produces, into <out_dir>:
    1_violin_per_level.png      g & alpha distributions per FPN level
    2_scatter_g_vs_contrib.png  per-image gate vs depth-contribution
    3_hero_strip_top.jpg        top-5 depth-heavy images
    3_hero_strip_bot.jpg        bottom-5 depth-light images
    4_hist_pixel_pooled.png     pooled per-pixel g histogram (all images)
    5_spatial_mean_maps.png     mean gate / contrib maps over the dataset
    summary_extra.csv           per-image aggregate metrics
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import os.path as osp
import sys

import numpy as np


# -----------------------------------------------------------------------------
# Global seaborn / matplotlib style — paper-ready defaults
# -----------------------------------------------------------------------------

def _setup_style():
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        font_scale=1.05,
        rc={
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.9,
            "axes.titlepad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "grid.color": "#e8e8e8",
            "grid.linewidth": 0.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        },
    )

    # Vibrant palette for FPN levels (electric blue → magenta → tangerine).
    return sns.color_palette(["#1f77ff", "#d633ff", "#ff8a00"])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("vis_dir",
                   help="Directory written by visualize_gate_val.py "
                        "(contains summary.csv + *_maps.npz [+ *_gate.jpg]).")
    p.add_argument("--out-dir", default=None,
                   help="Where to write plots (default: <vis_dir>/_dataset_viz).")
    p.add_argument("--hist-stride", type=int, default=4,
                   help="Subsample stride for pooled pixel histogram (memory).")
    p.add_argument("--spatial-size", type=int, default=128,
                   help="Resize each gate map to this size before averaging.")
    p.add_argument("--hero-k", type=int, default=5,
                   help="How many top/bottom images for hero strips.")
    p.add_argument("--rank-level", type=int, default=1,
                   help="Which level (1/2/3) to use for hero ranking.")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def load_summary(vis_dir):
    """Parse summary.csv -> dict[level] = list of dicts."""
    path = osp.join(vis_dir, "summary.csv")
    if not osp.isfile(path):
        raise SystemExit(f"missing {path}; run visualize_gate_val.py first")
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "stem":   r["stem"],
                "level":  int(r["level"]),
                "g_mean": float(r["g_mean"]),
                "g_std":  float(r["g_std"]),
                "a_mean": float(r["a_mean"]),
                "a_std":  float(r["a_std"]),
            })
    return rows


def list_npz(vis_dir):
    return sorted(glob.glob(osp.join(vis_dir, "*_maps.npz")))


# -----------------------------------------------------------------------------
# Plot 1: violin / box plot of g and alpha per level
# -----------------------------------------------------------------------------

def plot_violin(rows, out_path):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(rows)
    df["level_label"] = df["level"].map(lambda l: f"L{l}")
    df_long = pd.melt(
        df,
        id_vars=["stem", "level_label"],
        value_vars=["g_mean", "a_mean"],
        var_name="metric",
        value_name="value",
    )
    df_long["metric"] = df_long["metric"].map({
        "g_mean": "gate  g",
        "a_mean": "depth attention  α",
    })

    levels = sorted(df["level_label"].unique())
    palette = {
        "gate  g":            "#00b4d8",   # electric cyan
        "depth attention  α": "#ff5d8f",   # vivid pink
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, metric in zip(axes, ["gate  g", "depth attention  α"]):
        sub = df_long[df_long["metric"] == metric]
        sns.violinplot(
            data=sub, x="value", y="level_label", order=levels,
            ax=ax, inner=None, linewidth=1.1,
            color=palette[metric], saturation=1.0, cut=0,
        )
        # Strip overlay (per-image points)
        sns.stripplot(
            data=sub, x="value", y="level_label", order=levels,
            ax=ax, color="#1a1a1a", size=1.9, alpha=0.45, jitter=0.18,
        )
        # Mean marker
        means = sub.groupby("level_label")["value"].mean().reindex(levels)
        ax.scatter(means.values, range(len(levels)),
                   marker="D", s=42, color="white",
                   edgecolor="black", linewidth=1.0, zorder=5,
                   label="mean")

        ax.axvline(0.5, color="#888", ls="--", lw=1, alpha=0.7,
                   label="balanced (0.5)")
        ax.set_xlim(0, 1)
        ax.set_xlabel("value")
        ax.set_ylabel("FPN level" if ax is axes[0] else "")
        ax.set_title(metric + "   (per-image mean)")
        # Stats annotation per level
        for i, lvl in enumerate(levels):
            vals = sub[sub["level_label"] == lvl]["value"].values
            ax.text(0.985, i - 0.35,
                    f"μ={vals.mean():.3f}   σ={vals.std():.3f}   n={len(vals)}",
                    fontsize=8, color="#555", ha="right",
                    transform=ax.get_yaxis_transform())
        ax.legend(loc="lower right", fontsize=8)

    n_imgs = len(df) // len(levels)
    fig.suptitle(
        f"Per-level distribution of gate g and depth-attention α  ·  N={n_imgs} val images",
        y=1.02, fontsize=13, fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# -----------------------------------------------------------------------------
# Plot 2: scatter g_mean vs (1-g)*a per image (with annotations)
# -----------------------------------------------------------------------------

def compute_per_image_aggregates(npz_paths, rank_level):
    """For each image, compute g_mean, contrib_mean, contrib_max at rank_level."""
    out = []
    for p in npz_paths:
        stem = osp.basename(p).replace("_maps.npz", "")
        d = np.load(p)
        gk = f"gate_L{rank_level}"
        ak = f"alpha_L{rank_level}"
        if gk not in d.files or ak not in d.files:
            continue
        g = d[gk]
        a = d[ak]
        contrib = (1.0 - g) * a
        out.append({
            "stem":          stem,
            "g_mean":        float(g.mean()),
            "g_std":         float(g.std()),
            "a_mean":        float(a.mean()),
            "contrib_mean":  float(contrib.mean()),
            "contrib_max":   float(contrib.max()),
            "contrib_p90":   float(np.percentile(contrib, 90)),
        })
    return out


def plot_scatter(per_img, out_path, hero_k):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(per_img)
    df = df.rename(columns={"g_mean": "g", "contrib_mean": "contrib"})

    # JointGrid: scatter + marginal KDE/hist
    g_grid = sns.JointGrid(
        data=df, x="contrib", y="g", height=7, ratio=5, space=0.05,
    )
    g_grid.plot_joint(
        sns.scatterplot,
        s=32, hue=df["g"], palette="turbo",
        hue_norm=(0, 1), edgecolor="#111", linewidth=0.45,
        alpha=0.92, legend=False,
    )
    g_grid.plot_marginals(
        sns.histplot, bins=40, color="#7209b7",
        edgecolor="white", linewidth=0.5, alpha=0.85,
    )
    ax = g_grid.ax_joint

    # Reference line: g = 0.5 (balanced)
    ax.axhline(0.5, color="#666", ls="--", lw=1, alpha=0.7, zorder=1)
    ax.text(ax.get_xlim()[1] * 0.99, 0.51, "balanced (g = 0.5)",
            ha="right", va="bottom", fontsize=8, color="#666")

    # Pearson r as a stats badge
    if len(df) > 2:
        r = float(np.corrcoef(df["contrib"], df["g"])[0, 1])
        ax.text(0.02, 0.98, f"Pearson r = {r:+.3f}\nN = {len(df)}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color="#222",
                bbox=dict(facecolor="white", edgecolor="#ccc",
                          boxstyle="round,pad=0.35"))

    # Annotate hero points
    c_arr = df["contrib"].values
    g_arr = df["g"].values
    top_idx = np.argsort(c_arr)[-hero_k:][::-1]
    bot_idx = np.argsort(c_arr)[:hero_k]
    for i in top_idx:
        ax.annotate(df["stem"].iloc[i], (c_arr[i], g_arr[i]),
                    fontsize=7.5, color="#e63946", fontweight="bold",
                    xytext=(4, 4), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#e63946", lw=0.7))
    for i in bot_idx:
        ax.annotate(df["stem"].iloc[i], (c_arr[i], g_arr[i]),
                    fontsize=7.5, color="#0077b6", fontweight="bold",
                    xytext=(4, -10), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="#0077b6", lw=0.7))

    ax.set_xlabel("(1 − g) · α   ·   effective depth contribution (per-image mean)")
    ax.set_ylabel("g   ·   RGB trust (per-image mean)")
    ax.set_ylim(0, 1)
    ax.set_xlim(left=max(0.0, c_arr.min() * 0.95))
    g_grid.figure.suptitle(
        f"Gate trust vs effective depth use  ·  N={len(df)} val images",
        y=1.02, fontsize=13, fontweight="semibold",
    )
    g_grid.figure.savefig(out_path)
    plt.close(g_grid.figure)
    print(f"  wrote {out_path}")
    return list(top_idx), list(bot_idx)


# -----------------------------------------------------------------------------
# Plot 3: hero strips - top-K and bottom-K images
# -----------------------------------------------------------------------------

def make_hero_strip(per_img, idxs, vis_dir, out_path, title):
    import cv2

    THUMB = 320
    PAD = 8
    HEADER_H = 36
    rows = []
    for i in idxs:
        stem = per_img[i]["stem"]
        jpg = osp.join(vis_dir, f"{stem}_gate.jpg")
        if not osp.isfile(jpg):
            continue
        full = cv2.imread(jpg)
        if full is None:
            continue
        # Resize to a thin strip (keep aspect, target height = THUMB)
        h, w = full.shape[:2]
        scale = THUMB / h
        thumb = cv2.resize(full, (int(w * scale), THUMB),
                           interpolation=cv2.INTER_AREA)
        label = np.full((HEADER_H, thumb.shape[1], 3), 240, dtype=np.uint8)
        txt = f"{stem}   contrib={per_img[i]['contrib_mean']:.3f}   g={per_img[i]['g_mean']:.3f}"
        cv2.putText(label, txt, (8, 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        rows.append(np.concatenate([label, thumb], axis=0))

    if not rows:
        print(f"  skipped {out_path} (no source jpgs found)")
        return

    target_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < target_w:
            pad = np.full((r.shape[0], target_w - r.shape[1], 3), 240,
                          dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
        padded.append(np.full((PAD, target_w, 3), 245, dtype=np.uint8))
    padded.pop()

    canvas = np.concatenate(padded, axis=0)
    title_strip = np.full((40, target_w, 3), 240, dtype=np.uint8)
    cv2.putText(title_strip, title, (12, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, (35, 35, 35), 1, cv2.LINE_AA)
    canvas = np.concatenate([title_strip, canvas], axis=0)
    canvas = cv2.copyMakeBorder(canvas, 14, 14, 14, 14,
                                cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  wrote {out_path}")


# -----------------------------------------------------------------------------
# Plot 4: pooled pixel-level histogram of g (all images, all pixels)
# -----------------------------------------------------------------------------

def plot_pooled_hist(npz_paths, out_path, stride):
    import matplotlib.pyplot as plt
    import seaborn as sns

    first = np.load(npz_paths[0])
    levels = sorted(int(k.split("_L")[1]) for k in first.files if k.startswith("gate_L"))

    pooled_g = {lvl: [] for lvl in levels}
    pooled_a = {lvl: [] for lvl in levels}
    pooled_c = {lvl: [] for lvl in levels}

    for p in npz_paths:
        d = np.load(p)
        for lvl in levels:
            g = d[f"gate_L{lvl}"].ravel()[::stride]
            a = d[f"alpha_L{lvl}"].ravel()[::stride]
            c = ((1.0 - d[f"gate_L{lvl}"]) * d[f"alpha_L{lvl}"]).ravel()[::stride]
            pooled_g[lvl].append(g)
            pooled_a[lvl].append(a)
            pooled_c[lvl].append(c)

    for lvl in levels:
        pooled_g[lvl] = np.concatenate(pooled_g[lvl])
        pooled_a[lvl] = np.concatenate(pooled_a[lvl])
        pooled_c[lvl] = np.concatenate(pooled_c[lvl])

    fig, axes = plt.subplots(
        3, len(levels),
        figsize=(4.4 * len(levels), 9.5),
        sharex=True, sharey="row",
    )
    if len(levels) == 1:
        axes = axes[:, None]

    metrics = [
        ("g",        "#00b4d8", "RGB trust"),            # electric cyan
        ("α",        "#ff006e", "depth attention"),      # hot magenta
        ("(1−g)·α",  "#9d4edd", "effective depth use"),  # vivid violet
    ]
    pooled_by_metric = [pooled_g, pooled_a, pooled_c]

    for col, lvl in enumerate(levels):
        for row, ((name, color, ylabel), pooled) in enumerate(
                zip(metrics, pooled_by_metric)):
            ax = axes[row, col]
            data = pooled[lvl]

            sns.histplot(
                data, bins=80, binrange=(0, 1),
                color=color, edgecolor="white", linewidth=0.35,
                alpha=0.92, stat="density", ax=ax, kde=True,
                kde_kws={"bw_adjust": 0.6, "cut": 0},
                line_kws={"color": "#111", "linewidth": 1.7, "alpha": 0.95},
            )

            mu = float(np.mean(data))
            med = float(np.median(data))
            ax.axvline(0.5, color="#888", ls=":", lw=0.9, alpha=0.7)
            ax.axvline(mu, color="#222", ls="--", lw=1.1,
                       label=f"mean={mu:.3f}")
            ax.axvline(med, color=color, ls="-", lw=1.1, alpha=0.9,
                       label=f"median={med:.3f}")

            ax.text(0.98, 0.96,
                    f"μ = {mu:.3f}\nσ = {np.std(data):.3f}\nN = {len(data)/1e6:.1f} M",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, color="#222",
                    bbox=dict(facecolor="white", edgecolor="#cccccc",
                              boxstyle="round,pad=0.35", alpha=0.95))

            if row == 0:
                ax.set_title(f"Level L{lvl}  ·  stride {2 ** (lvl + 2)}",
                             fontsize=11.5)
            if col == 0:
                ax.set_ylabel(f"{name}\n{ylabel}\n· density ·",
                              fontsize=10)
            else:
                ax.set_ylabel("")
            if row == len(metrics) - 1:
                ax.set_xlabel("value")
            else:
                ax.set_xlabel("")
            ax.set_xlim(0, 1)
            ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)

    fig.suptitle(
        "Pooled per-pixel distributions across the validation set",
        y=1.00, fontsize=13.5, fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# -----------------------------------------------------------------------------
# Plot 5: spatial mean of gate / contrib maps across the dataset
# -----------------------------------------------------------------------------

def plot_spatial_mean(npz_paths, out_path, size):
    import cv2
    import matplotlib.pyplot as plt
    import seaborn as sns

    first = np.load(npz_paths[0])
    levels = sorted(int(k.split("_L")[1]) for k in first.files if k.startswith("gate_L"))

    mean_g = {lvl: np.zeros((size, size), dtype=np.float64) for lvl in levels}
    mean_c = {lvl: np.zeros((size, size), dtype=np.float64) for lvl in levels}
    n = 0

    for p in npz_paths:
        d = np.load(p)
        for lvl in levels:
            g = d[f"gate_L{lvl}"]
            a = d[f"alpha_L{lvl}"]
            c = (1.0 - g) * a
            g_r = cv2.resize(g.astype(np.float32), (size, size),
                             interpolation=cv2.INTER_LINEAR)
            c_r = cv2.resize(c.astype(np.float32), (size, size),
                             interpolation=cv2.INTER_LINEAR)
            mean_g[lvl] += g_r
            mean_c[lvl] += c_r
        n += 1

    for lvl in levels:
        mean_g[lvl] /= n
        mean_c[lvl] /= n

    fig, axes = plt.subplots(
        2, len(levels),
        figsize=(3.9 * len(levels), 7.6),
        constrained_layout=True,
    )
    if len(levels) == 1:
        axes = axes[:, None]

    # Vibrant divergent for g (blue ↔ yellow ↔ red), neon-ish magma for contrib.
    cmap_g = plt.get_cmap("turbo")
    cmap_c = sns.color_palette("magma", as_cmap=True)

    for col, lvl in enumerate(levels):
        # Row 0: mean gate g
        ax_g = axes[0, col]
        im_g = ax_g.imshow(mean_g[lvl], cmap=cmap_g, vmin=0, vmax=1,
                           interpolation="bilinear")
        ax_g.set_title(f"L{lvl}   ⟨ g ⟩   ·   stride {2 ** (lvl + 2)}",
                       fontsize=11)
        ax_g.set_xticks([]); ax_g.set_yticks([])
        for spine in ax_g.spines.values():
            spine.set_edgecolor("#888"); spine.set_linewidth(0.6)
        cbar = fig.colorbar(im_g, ax=ax_g, fraction=0.046, pad=0.03)
        cbar.outline.set_visible(False)
        cbar.set_label("g", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        # Stats annotation
        ax_g.text(0.02, 0.96,
                  f"μ={mean_g[lvl].mean():.3f}  σ={mean_g[lvl].std():.3f}",
                  transform=ax_g.transAxes, va="top", ha="left",
                  fontsize=8, color="white",
                  bbox=dict(facecolor="#0a0a0a", edgecolor="#ffffff",
                            boxstyle="round,pad=0.3", alpha=0.82,
                            linewidth=0.6))

        # Row 1: mean contrib (1-g)·a
        ax_c = axes[1, col]
        vmax_c = max(float(mean_c[lvl].max()), 1e-6)
        im_c = ax_c.imshow(mean_c[lvl], cmap=cmap_c, vmin=0, vmax=vmax_c,
                           interpolation="bilinear")
        ax_c.set_title(f"L{lvl}   ⟨ (1−g)·α ⟩", fontsize=11)
        ax_c.set_xticks([]); ax_c.set_yticks([])
        for spine in ax_c.spines.values():
            spine.set_edgecolor("#888"); spine.set_linewidth(0.6)
        cbar = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.03)
        cbar.outline.set_visible(False)
        cbar.set_label("contrib", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        ax_c.text(0.02, 0.96,
                  f"μ={mean_c[lvl].mean():.3f}  max={mean_c[lvl].max():.3f}",
                  transform=ax_c.transAxes, va="top", ha="left",
                  fontsize=8, color="white",
                  bbox=dict(facecolor="#0a0a0a", edgecolor="#ffffff",
                            boxstyle="round,pad=0.3", alpha=0.82,
                            linewidth=0.6))

    fig.suptitle(
        f"Spatial mean maps over {n} validation images   "
        f"(resampled to {size}×{size})",
        fontsize=13.5, fontweight="semibold",
    )
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# -----------------------------------------------------------------------------
# Extra CSV for downstream analysis (per image, per level)
# -----------------------------------------------------------------------------

def write_summary_extra(per_img_by_lvl, out_path):
    levels = sorted(per_img_by_lvl.keys())
    fields = ["stem"]
    for lvl in levels:
        fields += [f"L{lvl}_g_mean", f"L{lvl}_g_std", f"L{lvl}_a_mean",
                   f"L{lvl}_contrib_mean", f"L{lvl}_contrib_p90"]
    # Build per-stem row.
    by_stem = {}
    for lvl, rows in per_img_by_lvl.items():
        for r in rows:
            by_stem.setdefault(r["stem"], {})[lvl] = r
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for stem, by_lvl in sorted(by_stem.items()):
            row = [stem]
            for lvl in levels:
                r = by_lvl.get(lvl, {})
                row += [
                    f"{r.get('g_mean', float('nan')):.4f}",
                    f"{r.get('g_std', float('nan')):.4f}",
                    f"{r.get('a_mean', float('nan')):.4f}",
                    f"{r.get('contrib_mean', float('nan')):.4f}",
                    f"{r.get('contrib_p90', float('nan')):.4f}",
                ]
            w.writerow(row)
    print(f"  wrote {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = args.out_dir or osp.join(args.vis_dir, "_dataset_viz")
    os.makedirs(out_dir, exist_ok=True)

    _setup_style()

    rows = load_summary(args.vis_dir)
    npz_paths = list_npz(args.vis_dir)
    if not npz_paths:
        raise SystemExit(f"no *_maps.npz under {args.vis_dir}")
    print(f"loaded {len(rows)} summary rows, {len(npz_paths)} npz files")

    # ---- Plot 1
    print("[1/5] violin per-level …")
    plot_violin(rows, osp.join(out_dir, "1_violin_per_level.png"))

    # ---- Plot 2 + 3 (need per-image aggregates at rank_level)
    print(f"[2/5] per-image aggregates at L{args.rank_level} …")
    per_img = compute_per_image_aggregates(npz_paths, args.rank_level)

    print("[3/5] scatter g vs contrib …")
    top, bot = plot_scatter(per_img,
                            osp.join(out_dir, "2_scatter_g_vs_contrib.png"),
                            args.hero_k)

    print("[4/5] hero strips …")
    make_hero_strip(per_img, top, args.vis_dir,
                    osp.join(out_dir, "3_hero_strip_top.jpg"),
                    title=f"Top-{args.hero_k} depth-heavy images  "
                          f"(by L{args.rank_level} contrib mean)")
    make_hero_strip(per_img, bot, args.vis_dir,
                    osp.join(out_dir, "3_hero_strip_bot.jpg"),
                    title=f"Bottom-{args.hero_k} depth-light images  "
                          f"(by L{args.rank_level} contrib mean)")

    # ---- Plot 4 pooled pixel histogram
    print("[5/5] pooled pixel histograms …")
    plot_pooled_hist(npz_paths,
                     osp.join(out_dir, "4_hist_pixel_pooled.png"),
                     stride=args.hist_stride)

    # ---- Plot 5 spatial mean
    print("[+ ] spatial mean maps …")
    plot_spatial_mean(npz_paths,
                      osp.join(out_dir, "5_spatial_mean_maps.png"),
                      size=args.spatial_size)

    # ---- Extra CSV (per-image, all levels)
    print("[+ ] summary_extra.csv (all levels per image) …")
    per_img_by_lvl = {}
    for lvl in sorted({r["level"] for r in rows}):
        per_img_by_lvl[lvl] = compute_per_image_aggregates(npz_paths, lvl)
    write_summary_extra(per_img_by_lvl,
                        osp.join(out_dir, "summary_extra.csv"))

    print(f"\ndone. outputs under {out_dir}")


if __name__ == "__main__":
    main()
