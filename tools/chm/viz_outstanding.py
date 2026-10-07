"""Publication-grade figures for the DAv2-depth <-> CHM study (zone-level).

Produces five figures under dataset/QuebecTree/correlation_results/figs/<zone>/:

  fig1_crown_joint   : joint density (hexbin + marginals) of crown p90 depth vs
                       CHM, with 1:1 + regression -- the headline correlation.
  fig2_overlap_curve : Spearman rho / AbsRel as a *continuous* function of
                       distance to the nearest crown-crown boundary. The key
                       overlap evidence: rho stays flat into the boundary.
  fig3_neighbor_pairs: Delta-depth vs Delta-CHM for touching crown pairs, coded
                       correct/incorrect, + ordering accuracy vs |Delta CHM|.
                       Visualises the crown-separation mechanism.
  fig4_spatial_rho   : per-tile agreement (rho) painted over the RGB ortho --
                       shows where depth tracks CHM and where it fails (water).
  fig5_hero          : best touching-crown tiles -- RGB+outlines | depth | CHM |
                       a transect across two crowns where both signals agree.

Reuses the alignment / geometry helpers from correlation_experiments.py.

Usage::

    python tools/chm/viz_outstanding.py --zone zone3
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
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import spearmanr
from scipy.ndimage import distance_transform_edt


def repo_root():
    return osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


def load_ce():
    p = osp.join(osp.dirname(osp.abspath(__file__)), "correlation_experiments.py")
    spec = importlib.util.spec_from_file_location("ce", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DEFAULT_COG = ("/mnt/hoangdmv/Research/Instance_segmentation/dataset/"
               "quebec_trees_dataset_2021-09-02/2021-09-02/{z}/"
               "2021-09-02-sbl-z{n}-rgb-cog.tif")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zone", default="zone3",
                   choices=["zone1", "zone2", "zone3"])
    p.add_argument("--min-cov", type=float, default=0.6)
    p.add_argument("--trim", type=float, default=0.2)
    p.add_argument("--crown-min-px", type=int, default=200)
    p.add_argument("--neighbor-dilate", type=int, default=6)
    p.add_argument("--boundary-r", type=int, default=12)
    p.add_argument("--crown-agg", default="p90", choices=["p90", "max", "mean"])
    p.add_argument("--vmax", type=float, default=24.0)
    p.add_argument("--n-hero", type=int, default=2)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    ce = load_ce()
    import rasterio
    from rasterio.windows import Window
    from rasterio.enums import Resampling

    root = repo_root()
    z, n = args.zone, args.zone[-1]
    base = osp.join(root, "dataset", "QuebecTree")
    dd = osp.join(base, "depth_tiles", z)
    cd = osp.join(base, "chm_tiles", z)
    rd = osp.join(base, "crown_tiles", z)
    man = json.load(open(osp.join(base, f"tiles_manifest_{z}.json")))
    tiles_by_id = {r["tile_id"]: r for r in man["tiles"]}
    res = man["res"]
    cov = json.load(open(osp.join(cd, "coverage.json")))["finite_frac"]
    out = args.out_dir or osp.join(base, "correlation_results", "figs", z)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)

    usable = sorted(t for t, f in cov.items()
                    if f >= args.min_cov
                    and osp.isfile(osp.join(dd, f"{t}_depth.npy"))
                    and osp.isfile(osp.join(rd, f"{t}_inst.npy")))
    print(f"[{z}] usable tiles: {len(usable)}")

    # ---- accumulators ----
    crown_pts = {}                       # oid -> (dbar, hbar, npix)
    pair_dd, pair_dh, pair_ok = [], [], []
    # distance-to-boundary bins (px), per-bin reservoirs of (d_al, h)
    dist_edges = np.arange(0, 216, 8)
    nb = len(dist_edges) - 1
    bin_d = [[] for _ in range(nb)]
    bin_h = [[] for _ in range(nb)]
    tile_rho = {}                        # tile_id -> pixel rho
    hero_score = {}                      # tile_id -> score

    for tid in usable:
        d = np.load(osp.join(dd, f"{tid}_depth.npy")).astype(np.float32)
        h = np.load(osp.join(cd, f"{tid}_chm.npy")).astype(np.float32)
        if d.shape != h.shape:
            import cv2
            d = cv2.resize(d, (h.shape[1], h.shape[0]))
        m = np.isfinite(d) & np.isfinite(h)
        if m.sum() < 1000:
            continue
        s, t, _ = ce.robust_scale_shift(d[m], h[m], args.trim)
        d_al = s * d + t

        # tile pixel rho (subsample)
        dv, hv = d_al[m], h[m]
        k = min(20000, dv.size)
        ii = rng.choice(dv.size, k, replace=False)
        tile_rho[tid] = float(spearmanr(dv[ii], hv[ii])[0])

        inst = np.load(osp.join(rd, f"{tid}_inst.npy"))
        local = {}
        for oid in [int(i) for i in np.unique(inst) if i > 0]:
            cm = (inst == oid) & m
            if cm.sum() < args.crown_min_px:
                continue
            db = ce.agg_stat(d_al[cm], args.crown_agg)
            hb = ce.agg_stat(h[cm], args.crown_agg)
            local[oid] = (db, hb, int(cm.sum()))
            if oid not in crown_pts or cm.sum() > crown_pts[oid][2]:
                crown_pts[oid] = (db, hb, int(cm.sum()))

        # neighbour pairs
        pairs = ce.adjacency_pairs(inst, args.neighbor_dilate)
        big_dh = 0
        for a, b in pairs:
            if a in local and b in local:
                dh = local[a][1] - local[b][1]
                if abs(dh) < 0.5:
                    continue
                dd_ = local[a][0] - local[b][0]
                pair_dd.append(dd_); pair_dh.append(dh)
                pair_ok.append(np.sign(dd_) == np.sign(dh))
                if abs(dh) > 2:
                    big_dh += 1
        # heroes = tiles that both agree well (high tile rho) and contain
        # touching crowns with real height contrast
        hero_score[tid] = tile_rho[tid] * (1 + big_dh) + 0.001 * len(local)

        # distance-to-boundary curve
        seed = ce.crown_crown_boundary(inst, 1)
        if seed.any():
            dist = distance_transform_edt(~seed)
            crown_m = (inst > 0) & m
            db_idx = np.digitize(dist[crown_m], dist_edges) - 1
            dvals = d_al[crown_m]; hvals = h[crown_m]
            for bi in range(nb):
                sel = db_idx == bi
                c = int(sel.sum())
                if c == 0:
                    continue
                take = min(1500, c)
                jj = rng.choice(c, take, replace=False)
                bin_d[bi].append(dvals[sel][jj])
                bin_h[bi].append(hvals[sel][jj])

    sns.set_theme(style="whitegrid", context="talk")

    # ================= fig1: crown joint density =================
    oids = sorted(crown_pts)
    cdv = np.array([crown_pts[o][0] for o in oids])
    chv = np.array([crown_pts[o][1] for o in oids])
    g_r = float(np.corrcoef(cdv, chv)[0, 1])
    g_rho = float(spearmanr(cdv, chv)[0])
    jg = sns.jointplot(x=cdv, y=chv, kind="hex", height=8,
                       joint_kws=dict(gridsize=38, mincnt=1, cmap="turbo"),
                       marginal_kws=dict(bins=40, color="#2ecc71", kde=True))
    lim = float(np.percentile(np.concatenate([cdv, chv]), 99.5)) + 1
    jg.ax_joint.plot([0, lim], [0, lim], "k--", lw=1.2, alpha=.7, label="1:1")
    b1, b0 = np.polyfit(cdv, chv, 1)
    xs = np.array([0, lim])
    jg.ax_joint.plot(xs, b1 * xs + b0, "r-", lw=2, label="OLS fit")
    jg.ax_joint.set_xlim(0, lim); jg.ax_joint.set_ylim(0, lim)
    jg.set_axis_labels(f"DAv2 {args.crown_agg} → metres (per crown)",
                       f"CHM {args.crown_agg} (m, per crown)")
    jg.ax_joint.legend(loc="lower right", fontsize=12)
    jg.ax_joint.text(0.04, 0.93,
                     f"ρ = {g_rho:.2f}\nr = {g_r:.2f}\nn = {len(oids)} crowns",
                     transform=jg.ax_joint.transAxes, fontsize=15,
                     va="top", bbox=dict(boxstyle="round", fc="white", alpha=.8))
    jg.figure.suptitle("Crown-level canopy height — DAv2 vs CHM", y=1.02)
    jg.figure.savefig(osp.join(out, "fig1_crown_joint.png"), dpi=140,
                      bbox_inches="tight")
    plt.close(jg.figure)

    # ================= fig2: overlap distance curve =================
    centers_m = (0.5 * (dist_edges[:-1] + dist_edges[1:])) * res
    rho_bin, n_bin, absrel_bin = [], [], []
    for bi in range(nb):
        if bin_d[bi]:
            xd = np.concatenate(bin_d[bi]); xh = np.concatenate(bin_h[bi])
            rho_bin.append(spearmanr(xd, xh)[0] if xd.size > 10 else np.nan)
            pos = xh > 0.5
            absrel_bin.append(float(np.mean(np.abs(xd[pos] - xh[pos]) / xh[pos]))
                              if pos.any() else np.nan)
            n_bin.append(xd.size)
        else:
            rho_bin.append(np.nan); absrel_bin.append(np.nan); n_bin.append(0)
    rho_bin = np.array(rho_bin); absrel_bin = np.array(absrel_bin)
    fig, ax = plt.subplots(figsize=(10, 6))
    band_m = args.boundary_r * res
    ax.axvspan(0, band_m, color="#dd8452", alpha=.18,
               label=f"boundary band (≤{band_m:.2f} m)")
    ax.plot(centers_m, rho_bin, "-o", color="#1f77b4", lw=2.5, ms=7,
            label="Spearman ρ")
    interior = np.nanmean(rho_bin[centers_m > 1.0])
    ax.axhline(interior, color="#1f77b4", ls=":", alpha=.6)
    ax.set_xlabel("distance to nearest crown–crown boundary (m)")
    ax.set_ylabel("Spearman ρ (DAv2 vs CHM)", color="#1f77b4")
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(centers_m, absrel_bin, "-s", color="#d62728", lw=1.6, ms=5,
             alpha=.7, label="AbsRel")
    ax2.set_ylabel("AbsRel", color="#d62728")
    ax2.set_ylim(0, max(0.6, float(np.nanmax(absrel_bin)) * 1.3))
    ax2.grid(False)
    ax.set_title("Depth–CHM agreement vs distance to crown boundary\n"
                 "(ρ flat into the boundary ⇒ depth holds where RGB is ambiguous)",
                 fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(osp.join(out, "fig2_overlap_curve.png"), dpi=140)
    plt.close(fig)

    # ================= fig3: neighbour-pair ordering =================
    pd_ = np.array(pair_dd); ph_ = np.array(pair_dh); pk_ = np.array(pair_ok)
    acc = float(pk_.mean()) if pk_.size else float("nan")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.4),
                           gridspec_kw=dict(width_ratios=[1.25, 1]))
    L = float(np.percentile(np.abs(np.concatenate([pd_, ph_])), 99)) if pd_.size else 10
    ax[0].axhspan(0, L, xmin=0.5, xmax=1, color="green", alpha=.05)
    ax[0].axhspan(-L, 0, xmin=0, xmax=0.5, color="green", alpha=.05)
    ax[0].scatter(pd_[pk_], ph_[pk_], s=14, c="#2ca02c", alpha=.4,
                  label=f"correct ({pk_.mean()*100:.0f}%)", edgecolors="none")
    ax[0].scatter(pd_[~pk_], ph_[~pk_], s=14, c="#d62728", alpha=.5,
                  label="wrong", edgecolors="none")
    ax[0].axhline(0, color="k", lw=.8); ax[0].axvline(0, color="k", lw=.8)
    ax[0].set_xlim(-L, L); ax[0].set_ylim(-L, L)
    ax[0].set_xlabel("Δ DAv2 height  (crown A − B, m)")
    ax[0].set_ylabel("Δ CHM height  (crown A − B, m)")
    ax[0].set_title(f"touching-crown pairs (n={pk_.size})\n"
                    f"ordering accuracy = {acc*100:.1f}%  (chance 50%)",
                    fontsize=14)
    ax[0].legend(loc="lower right", fontsize=12)
    # accuracy vs |Δh|
    abdh = np.abs(ph_)
    edges = np.array([0.5, 1, 2, 3, 5, 8, 30])
    bc, accv, nn = [], [], []
    for i in range(len(edges) - 1):
        sel = (abdh >= edges[i]) & (abdh < edges[i + 1])
        if sel.sum() >= 10:
            bc.append((edges[i] + min(edges[i + 1], 12)) / 2)
            accv.append(pk_[sel].mean()); nn.append(int(sel.sum()))
    ax[1].plot(bc, np.array(accv) * 100, "-o", color="#9467bd", lw=2.5, ms=8)
    for x, y, c in zip(bc, np.array(accv) * 100, nn):
        ax[1].annotate(f"n={c}", (x, y), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=10)
    ax[1].axhline(50, color="k", ls="--", alpha=.6, label="chance")
    ax[1].set_xlabel("|Δ CHM height| between neighbours (m)")
    ax[1].set_ylabel("ordering accuracy (%)")
    ax[1].set_ylim(40, 102)
    ax[1].set_title("the larger the true height gap,\nthe more reliably depth orders them",
                    fontsize=14)
    ax[1].legend(loc="lower right", fontsize=12)
    fig.tight_layout()
    fig.savefig(osp.join(out, "fig3_neighbor_pairs.png"), dpi=140)
    plt.close(fig)

    # ================= fig4: spatial rho map =================
    cog = DEFAULT_COG.format(z=z, n=n)
    with rasterio.open(cog) as ds:
        cH, cW = ds.height, ds.width
        scale = 1500 / max(cH, cW)
        oh, ow = int(cH * scale), int(cW * scale)
        rgb = np.transpose(ds.read((1, 2, 3), out_shape=(3, oh, ow),
                                   resampling=Resampling.bilinear), (1, 2, 0))
    fig, ax = plt.subplots(figsize=(11, 11 * oh / ow))
    ax.imshow(rgb)
    sx, sy = ow / cW, oh / cH
    tile = man["tile"]
    rr = [tile_rho[t] for t in tile_rho]
    for t, rho in tile_rho.items():
        rec = tiles_by_id[t]
        x = rec["col_off"] * sx; y = rec["row_off"] * sy
        w = tile * sx; hgt = tile * sy
        col = plt.cm.RdYlGn((rho - 0) / 1.0)
        ax.add_patch(Rectangle((x, y), w, hgt, facecolor=col, alpha=.55,
                               edgecolor="k", linewidth=.3))
    sm = plt.cm.ScalarMappable(cmap="RdYlGn",
                               norm=plt.Normalize(vmin=0, vmax=1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("per-tile Spearman ρ (DAv2 vs CHM)")
    ax.set_title("Spatial map of depth–CHM agreement "
                 f"(median ρ={np.median(rr):.2f})")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(osp.join(out, "fig4_spatial_rho.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # ================= fig5: hero touching-crown tiles =================
    heroes = sorted(hero_score, key=lambda k: -hero_score[k])[:args.n_hero]
    nH = len(heroes)
    if nH:
        fig, ax = plt.subplots(nH, 4, figsize=(20, 5 * nH))
        if nH == 1:
            ax = ax[None, :]
        with rasterio.open(cog) as ds:
            for i, tid in enumerate(heroes):
                rec = tiles_by_id[tid]
                win = Window(rec["col_off"], rec["row_off"],
                             rec["width"], rec["height"])
                rgb = np.transpose(ds.read((1, 2, 3), window=win), (1, 2, 0))
                d = np.load(osp.join(dd, f"{tid}_depth.npy")).astype(np.float32)
                h = np.load(osp.join(cd, f"{tid}_chm.npy")).astype(np.float32)
                inst = np.load(osp.join(rd, f"{tid}_inst.npy"))
                m = np.isfinite(d) & np.isfinite(h)
                s, t, _ = ce.robust_scale_shift(d[m], h[m], args.trim)
                d_al = np.where(m, s * d + t, np.nan)

                vmax_t = float(np.nanpercentile(h, 98))   # per-tile contrast
                ax[i, 0].imshow(rgb)
                ax[i, 0].contour(inst, levels=np.unique(inst)[1:],
                                 colors="yellow", linewidths=0.6)
                ax[i, 0].set_ylabel(tid.replace(f"{z}_", ""), fontsize=10)
                ax[i, 1].imshow(np.ma.masked_invalid(d_al), cmap="viridis",
                                vmin=0, vmax=vmax_t)
                ax[i, 2].imshow(np.ma.masked_invalid(h), cmap="viridis",
                                vmin=0, vmax=vmax_t)

                # transect across the highest-|Δh| touching pair
                local = {}
                for oid in [int(x) for x in np.unique(inst) if x > 0]:
                    cm = (inst == oid) & m
                    if cm.sum() >= args.crown_min_px:
                        ys, xs = np.where(inst == oid)
                        local[oid] = (ce.agg_stat(h[cm], "p90"),
                                      float(xs.mean()), float(ys.mean()))
                best = None
                for a, b in ce.adjacency_pairs(inst, args.neighbor_dilate):
                    if a in local and b in local:
                        dh = abs(local[a][0] - local[b][0])
                        if best is None or dh > best[0]:
                            best = (dh, a, b)
                if best is not None:
                    _, a, b = best
                    (ha, xa, ya), (hb, xb, yb) = local[a], local[b]
                    nps = 220
                    xs = np.linspace(xa, xb, nps).astype(int).clip(0, d.shape[1] - 1)
                    ys = np.linspace(ya, yb, nps).astype(int).clip(0, d.shape[0] - 1)

                    def _fill(v):
                        v = v.astype(float).copy(); ix = np.arange(len(v))
                        ok = np.isfinite(v)
                        if ok.sum() >= 2:
                            v[~ok] = np.interp(ix[~ok], ix[ok], v[ok])
                        return v

                    def _norm(v):
                        lo, hi = np.nanmin(v), np.nanmax(v)
                        return (v - lo) / (hi - lo + 1e-6)

                    def _smooth(v, k=11):
                        return np.convolve(v, np.ones(k) / k, mode="same")

                    dl = _smooth(_norm(_fill(d_al[ys, xs])))
                    hl = _smooth(_norm(_fill(h[ys, xs])))
                    tt = np.linspace(0, 1, nps)
                    ax[i, 3].plot(tt, hl, "-", color="#2ca02c", lw=3,
                                  label="CHM (norm.)")
                    ax[i, 3].plot(tt, dl, "-", color="#1f77b4", lw=3,
                                  label="DAv2 depth (norm.)")
                    corr = "✓ same order" if np.sign(ha - hb) == np.sign(
                        local[a][0] - local[b][0]) else "✗"
                    ax[i, 3].set_xlabel(f"transect: crown {a} (CHM {ha:.0f}m) "
                                        f"→ crown {b} (CHM {hb:.0f}m)  {corr}")
                    ax[i, 3].set_ylabel("normalised height\n(scale-invariant)")
                    ax[i, 3].set_ylim(-0.05, 1.1)
                    ax[i, 3].legend(fontsize=11, loc="upper center", ncol=2)
                    ax[i, 3].grid(True, alpha=.3)
                    ax[i, 0].plot([xa, xb], [ya, yb], "r-", lw=2)
                    ax[i, 0].scatter([xa, xb], [ya, yb], c="red", s=40, zorder=5)
                if i == 0:
                    for j, ttl in enumerate(["RGB + GT crowns",
                                             "DAv2 → metres", "CHM (m)",
                                             "transect across touching crowns"]):
                        ax[0, j].set_title(ttl, fontsize=13)
                for j in range(3):
                    ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        fig.suptitle(f"{z}: depth resolves relative height of touching crowns",
                     y=1.0, fontsize=15)
        fig.tight_layout()
        fig.savefig(osp.join(out, "fig5_hero.png"), dpi=130,
                    bbox_inches="tight")
        plt.close(fig)

    print(f"[{z}] wrote fig1..fig5 -> {out}")


if __name__ == "__main__":
    main()
