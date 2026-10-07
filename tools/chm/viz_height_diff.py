"""Height-difference visualisation: does DAv2 monocular depth reproduce
*differences of height* between trees?

For each selected tile draws
    RGB+crowns | DAv2 raw | DAv2->m | CHM (m) | ΔH map (DAv2->m − CHM) | crown height ladder
and one aggregate figure
    pooled crown scatter | neighbour ΔH quadrants | ordering accuracy vs |ΔCHM|.

The "height ladder" sorts the tile's crowns by true CHM height and overlays
the depth-derived height of each crown — if monocular depth carries a height
signal the blue markers climb with the green ones. The neighbour panels show
the overlap-relevant factor: for two *touching* crowns, does the depth get
the sign (and margin) of their height difference right?

Usage:
    python tools/chm/viz_height_diff.py --zone zone3 --n-tiles 4
    # -> dataset/QuebecTree/chm/viz/zone3_height_diff_tiles.png
    # -> dataset/QuebecTree/chm/viz/zone3_height_diff_agg.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import os.path as osp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.stats import pearsonr, spearmanr
from skimage.segmentation import find_boundaries


def repo_root():
    return osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


def load_ce():
    spec = importlib.util.spec_from_file_location(
        "correlation_experiments",
        osp.join(osp.dirname(osp.abspath(__file__)), "correlation_experiments.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CE = load_ce()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zone", default="zone3")
    p.add_argument("--n-tiles", type=int, default=4,
                   help="tiles shown in the per-tile figure "
                        "(best crown-rho tiles + the median one)")
    p.add_argument("--min-cov", type=float, default=0.6)
    p.add_argument("--min-crowns", type=int, default=10)
    p.add_argument("--crown-min-px", type=int, default=200)
    p.add_argument("--neighbor-dilate", type=int, default=6)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


# ---------------------------------------------------------------- per tile
def analyse_tile(dep_path, chm_path, inst_path, crown_min_px, dilate, rng):
    d = np.load(dep_path).astype(np.float64)
    h = np.load(chm_path).astype(np.float64)
    inst = np.load(inst_path)
    valid = np.isfinite(h) & np.isfinite(d)
    if valid.sum() < 5000:
        return None

    # per-tile robust scale-shift (depth is affine-invariant, CHM is metres)
    idx = np.flatnonzero(valid)
    if idx.size > 30000:
        idx = rng.choice(idx, 30000, replace=False)
    s, t, _ = CE.robust_scale_shift(d.ravel()[idx], h.ravel()[idx])
    dm = s * d + t

    # per-crown canopy-top heights (p90)
    crowns = {}
    for cid in np.unique(inst):
        if cid <= 0:
            continue
        m = (inst == cid)
        if m.sum() < crown_min_px:
            continue
        mv = m & valid
        if mv.sum() < 100:
            continue
        crowns[int(cid)] = (float(np.percentile(dm[mv], 90)),
                            float(np.percentile(h[mv], 90)))
    if len(crowns) < 3:
        return None

    dep_p90 = np.array([v[0] for v in crowns.values()])
    chm_p90 = np.array([v[1] for v in crowns.values()])
    rho = spearmanr(dep_p90, chm_p90)[0]

    # touching-crown pairs: sign of height difference
    pairs = []
    for a, b in CE.adjacency_pairs(inst, dilate):
        if a in crowns and b in crowns:
            dd = crowns[a][0] - crowns[b][0]
            dh = crowns[a][1] - crowns[b][1]
            pairs.append((dh, dd))
    pairs = np.array(pairs) if pairs else np.zeros((0, 2))
    acc = (float(np.mean(np.sign(pairs[:, 0]) == np.sign(pairs[:, 1])))
           if len(pairs) else np.nan)

    return dict(d=d, dm=dm, h=h, inst=inst, valid=valid, crowns=crowns,
                rho=float(rho), pairs=pairs, acc=acc, s=s, t=t)


def crown_overlay(inst, width=1, color=(1.0, 1.0, 0.0, 1.0)):
    """RGBA overlay of tight per-instance outlines. `mode='inner'` marks the
    ring of pixels *inside* each crown that touch a different label, so every
    instance keeps its own boundary even where crowns touch. `width` dilates
    the 1-px ring for visibility while staying glued to each crown edge."""
    bnd = find_boundaries(inst, mode="inner", background=0)
    if width > 1:
        bnd = binary_dilation(bnd, iterations=width - 1)
    overlay = np.zeros((*inst.shape, 4), np.float32)
    overlay[bnd] = color
    return overlay


# ---------------------------------------------------------------- figures
def draw_tiles(sel, rgb_dir, out_png):
    n = len(sel)
    fig, axes = plt.subplots(n, 6, figsize=(26, 4.4 * n))
    axes = np.atleast_2d(axes)
    for r, (tid, R) in enumerate(sel):
        rgb = np.asarray(Image.open(osp.join(rgb_dir, tid + ".jpg")))
        h_m = np.ma.masked_invalid(R["h"])
        dm_m = np.ma.masked_where(~R["valid"], R["dm"])
        vmin = float(np.nanpercentile(R["h"], 2))
        vmax = float(np.nanpercentile(R["h"], 98))

        ax = axes[r, 0]
        ax.imshow(rgb)
        ax.imshow(crown_overlay(R["inst"], width=6))
        ax.set_ylabel(tid, fontsize=9)
        if r == 0:
            ax.set_title("RGB + GT crowns")

        ax = axes[r, 1]
        ax.imshow(R["d"], cmap="magma")
        if r == 0:
            ax.set_title("DAv2 raw (relative)")

        ax = axes[r, 2]
        im = ax.imshow(dm_m, cmap="viridis", vmin=vmin, vmax=vmax)
        if r == 0:
            ax.set_title("DAv2 → metres (scale-shift)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[r, 3]
        im = ax.imshow(h_m, cmap="viridis", vmin=vmin, vmax=vmax)
        if r == 0:
            ax.set_title("CHM (m)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[r, 4]
        diff = np.where(R["valid"], R["dm"] - R["h"], np.nan)
        lim = float(np.nanpercentile(np.abs(diff), 95))
        im = ax.imshow(np.ma.masked_invalid(diff), cmap="RdBu_r",
                       vmin=-lim, vmax=lim)
        mae = float(np.nanmean(np.abs(diff)))
        ax.set_xlabel(f"MAE={mae:.2f} m", fontsize=9)
        if r == 0:
            ax.set_title("ΔH = DAv2→m − CHM")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[r, 5]
        dep = np.array([v[0] for v in R["crowns"].values()])
        chm = np.array([v[1] for v in R["crowns"].values()])
        order = np.argsort(chm)
        x = np.arange(len(order))
        ax.plot(x, chm[order], "-o", c="forestgreen", ms=3, lw=1.5,
                label="CHM p90")
        ax.plot(x, dep[order], "o", c="steelblue", ms=3, alpha=0.85,
                label="DAv2 p90")
        ax.set_xlabel("crowns sorted by true height", fontsize=9)
        ax.set_ylabel("crown height (m)", fontsize=9)
        npair = len(R["pairs"])
        ax.text(0.03, 0.97,
                f"crown ρ={R['rho']:.2f}  (n={len(dep)})\n"
                f"neighbour ordering acc={R['acc']:.0%} ({npair} pairs)",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(fc="white", alpha=0.8, ec="none"))
        if r == 0:
            ax.set_title("crown height ladder")
            ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    for ax in axes[:, :5].ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "DAv2 monocular depth reproduces between-tree height differences "
        "(per-tile scale-shift vs LiDAR-style CHM)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def draw_agg(results, out_png):
    dep = np.concatenate([[v[0] for v in R["crowns"].values()]
                          for _, R in results])
    chm = np.concatenate([[v[1] for v in R["crowns"].values()]
                          for _, R in results])
    pairs = np.concatenate([R["pairs"] for _, R in results if len(R["pairs"])])

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    ax = axes[0]
    hb = ax.hexbin(dep, chm, gridsize=45, cmap="viridis", mincnt=1)
    lo = min(np.percentile(dep, 1), np.percentile(chm, 1))
    hi = max(np.percentile(dep, 99), np.percentile(chm, 99))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    r = pearsonr(dep, chm)[0]
    rho = spearmanr(dep, chm)[0]
    ax.set_xlabel("DAv2 crown height p90 (m, per-tile scale-shift)")
    ax.set_ylabel("CHM crown height p90 (m)")
    ax.set_title(f"per-crown height, all tiles  "
                 f"(n={len(dep)}, r={r:.2f}, ρ={rho:.2f})")
    plt.colorbar(hb, ax=ax, label="crowns")

    ax = axes[1]
    dh, dd = pairs[:, 0], pairs[:, 1]
    ok = np.sign(dh) == np.sign(dd)
    ax.axhline(0, c="grey", lw=0.8)
    ax.axvline(0, c="grey", lw=0.8)
    ax.scatter(dh[ok], dd[ok], s=6, c="forestgreen", alpha=0.5,
               label=f"ordering correct ({ok.mean():.0%})")
    ax.scatter(dh[~ok], dd[~ok], s=6, c="crimson", alpha=0.5,
               label="ordering wrong")
    ax.set_xlabel("ΔCHM between touching crowns (m)")
    ax.set_ylabel("ΔDAv2 height (m)")
    ax.set_title(f"touching-crown height differences (n={len(dh)} pairs)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    bins = np.array([0, 0.5, 1, 2, 3, 5, 8, np.inf])
    mids, accs, ns = [], [], []
    a = np.abs(dh)
    for i in range(len(bins) - 1):
        m = (a >= bins[i]) & (a < bins[i + 1])
        if m.sum() < 15:
            continue
        mids.append(a[m].mean())
        accs.append(ok[m].mean())
        ns.append(int(m.sum()))
    ax.axhline(0.5, c="grey", ls="--", lw=1, label="chance")
    ax.plot(mids, accs, "-o", c="steelblue")
    for x, y, n in zip(mids, accs, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("|ΔCHM| between touching crowns (m)")
    ax.set_ylabel("ordering accuracy")
    ax.set_title("bigger true height gap → more reliable monocular ordering")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("DAv2 depth as a relative-height signal (crown level)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    root = repo_root()
    base = osp.join(root, "dataset", "QuebecTree")
    dep_dir = osp.join(base, "depth_tiles", args.zone)
    chm_dir = osp.join(base, "chm_tiles", args.zone)
    inst_dir = osp.join(base, "crown_tiles", args.zone)
    rgb_dir = osp.join(base, "tiles", args.zone)
    out_dir = args.out_dir or osp.join(base, "chm", "viz")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    cov = json.load(open(osp.join(chm_dir, "coverage.json")))["finite_frac"]
    tids = sorted(t for t, f in cov.items() if f >= args.min_cov)

    results = []
    for tid in tids:
        dp = osp.join(dep_dir, tid + "_depth.npy")
        cp = osp.join(chm_dir, tid + "_chm.npy")
        ip = osp.join(inst_dir, tid + "_inst.npy")
        if not (osp.exists(dp) and osp.exists(cp) and osp.exists(ip)):
            continue
        R = analyse_tile(dp, cp, ip, args.crown_min_px,
                         args.neighbor_dilate, rng)
        if R is not None:
            results.append((tid, R))
    if not results:
        raise SystemExit("no usable tiles found")
    print(f"analysed {len(results)} tiles")

    rich = [(tid, R) for tid, R in results
            if len(R["crowns"]) >= args.min_crowns and len(R["pairs"]) >= 5]
    rich.sort(key=lambda x: -x[1]["rho"])
    k = max(1, args.n_tiles - 1)
    sel = rich[:k] + [rich[len(rich) // 2]]     # best + one median (honesty)
    seen: set[str] = set()
    sel = [x for x in sel if not (x[0] in seen or seen.add(x[0]))]
    sel = sel[:args.n_tiles]

    tiles_png = osp.join(out_dir, f"{args.zone}_height_diff_tiles.png")
    agg_png = osp.join(out_dir, f"{args.zone}_height_diff_agg.png")
    draw_tiles(sel, rgb_dir, tiles_png)
    draw_agg(results, agg_png)

    rhos = np.array([R["rho"] for _, R in results])
    accs = np.array([R["acc"] for _, R in results if np.isfinite(R["acc"])])
    print(f"crown-level Spearman ρ: mean {np.nanmean(rhos):.3f} "
          f"± {np.nanstd(rhos):.3f}")
    print(f"neighbour ordering accuracy: mean {accs.mean():.3f}")
    print("wrote", tiles_png)
    print("wrote", agg_png)


if __name__ == "__main__":
    main()
