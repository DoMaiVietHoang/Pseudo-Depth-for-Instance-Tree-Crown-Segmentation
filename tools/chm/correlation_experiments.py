"""Depth<->CHM correlation experiments (the 3-experiment validation suite).

Consumes the aligned per-tile triplets produced by the rest of tools/chm/:
    depth_tiles/<zone>/<tile_id>_depth.npy   (DAv2, generate_depth.py)
    chm_tiles/<zone>/<tile_id>_chm.npy       (crop_chm_to_tiles.py)
    crown_tiles/<zone>/<tile_id>_inst.npy    (rasterize_crowns.py)

Everything is compared *after a per-tile robust scale-shift* of the DAv2 depth to
the CHM (monocular depth is affine-invariant). Alignment is least-trimmed-squares
(LS then iterative inlier refit) so crown edges / no-data do not drag the fit.

Experiments
-----------
Exp1  Pixel-level  : per-tile robust s,t -> Spearman rho (main), Pearson r,
                     AbsRel, RMSE (post-align). One table row per zone-set.
Exp2  Crown-level  : per crown, p90(depth) vs p90(CHM) (canopy top). Global
                     scatter + within-tile rho + neighbour-pair ordering accuracy
                     (the overlap-relevant signal).
Exp3  Overlap split: pixel rho on crown-crown boundary zones vs crown interiors
                     -- tests "depth helps where RGB is ambiguous".

Usage::

    python tools/chm/correlation_experiments.py --zones zone3
    python tools/chm/correlation_experiments.py --zones zone1 zone2 zone3 \
        --min-cov 0.6 --tag quebectree
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def repo_root():
    return osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zones", nargs="+", required=True,
                   choices=["zone1", "zone2", "zone3"])
    p.add_argument("--tag", default=None,
                   help="name for the output row/files (default: zones joined)")
    p.add_argument("--min-cov", type=float, default=0.6,
                   help="only tiles with >= this CHM coverage")
    p.add_argument("--trim", type=float, default=0.2,
                   help="LTS trimming fraction for robust scale-shift")
    p.add_argument("--align-iters", type=int, default=3)
    p.add_argument("--px-per-tile", type=int, default=20000,
                   help="random valid pixels sampled per tile for pooled stats")
    p.add_argument("--crown-min-px", type=int, default=200,
                   help="min crown pixels to keep a crown")
    p.add_argument("--neighbor-dilate", type=int, default=6,
                   help="px tolerance for two crowns to count as touching")
    p.add_argument("--boundary-r", type=int, default=12,
                   help="half-width (px) of the crown-crown boundary zone")
    p.add_argument("--crown-agg", default="p90",
                   choices=["p90", "max", "mean"],
                   help="per-crown canopy-top statistic")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


# ----------------------------------------------------------------------------
def robust_scale_shift(d, h, trim=0.2, iters=3):
    """LTS-ish s,t for s*d+t ~ h. Returns (s, t, inlier_bool)."""
    inl = np.ones(d.shape, bool)
    s, t = 1.0, 0.0
    for _ in range(max(1, iters)):
        A = np.stack([d[inl], np.ones(inl.sum())], 1)
        sol, *_ = np.linalg.lstsq(A, h[inl], rcond=None)
        s, t = float(sol[0]), float(sol[1])
        res = np.abs(s * d + t - h)
        thr = np.quantile(res, 1.0 - trim)
        new = res <= thr
        if new.sum() < 50 or new.sum() == inl.sum():
            inl = new
            break
        inl = new
    return s, t, inl


def corr_block(x, y):
    if x.size < 10:
        return np.nan, np.nan
    return pearsonr(x, y)[0], spearmanr(x, y)[0]


def agg_stat(v, how):
    if how == "p90":
        return float(np.percentile(v, 90))
    if how == "max":
        return float(np.max(v))
    return float(np.mean(v))


def _crown_boxes(inst, ids, pad, H, W):
    from scipy import ndimage
    objs = ndimage.find_objects(inst)
    box = {}
    for i in ids:
        sl = objs[i - 1]
        if sl is None:
            continue
        box[i] = (max(0, sl[0].start - pad), min(H, sl[0].stop + pad),
                  max(0, sl[1].start - pad), min(W, sl[1].stop + pad))
    return box


def _pair_contest(inst, a, b, ba, bb, rad):
    """If crowns a,b are within ``rad`` px, return (r0, c0, inter_mask) on the
    *union* window (so both crowns' pixels are present for the dilation),
    else None."""
    from scipy import ndimage
    # padded bboxes must overlap (inclusive) for the pair to be close
    if (max(ba[0], bb[0]) > min(ba[1], bb[1]) or
            max(ba[2], bb[2]) > min(ba[3], bb[3])):
        return None
    r0, r1 = min(ba[0], bb[0]), max(ba[1], bb[1])
    c0, c1 = min(ba[2], bb[2]), max(ba[3], bb[3])
    sub = inst[r0:r1, c0:c1]
    inter = (ndimage.binary_dilation(sub == a, iterations=rad)
             & ndimage.binary_dilation(sub == b, iterations=rad))
    return r0, c0, inter


def crown_crown_boundary(inst, r):
    """Contested band between *different* crowns: pixels within ``r`` of two
    distinct crowns (effective width ~2r). bbox-pruned so it scales to dense
    tiles. Captures touching crowns and crowns separated by a small gap (<=2r).
    """
    H, W = inst.shape
    out = np.zeros(inst.shape, bool)
    ids = [int(i) for i in np.unique(inst) if i > 0]
    if len(ids) < 2:
        return out
    box = _crown_boxes(inst, ids, r, H, W)
    ids = [i for i in ids if i in box]
    for ai in range(len(ids)):
        a = ids[ai]
        for bi in range(ai + 1, len(ids)):
            b = ids[bi]
            res = _pair_contest(inst, a, b, box[a], box[b], r)
            if res is None:
                continue
            r0, c0, inter = res
            if inter.any():
                hh, ww = inter.shape
                out[r0:r0 + hh, c0:c0 + ww] |= inter
    return out


def adjacency_pairs(inst, dilate):
    """Set of touching crown id pairs (within `dilate` px).

    bbox-pruned: only crowns whose dilated bounding boxes overlap are tested,
    and the dilated-mask intersection is checked on the small union window.
    """
    pairs = set()
    ids = [int(i) for i in np.unique(inst) if i > 0]
    if len(ids) < 2:
        return pairs
    H, W = inst.shape
    rad = max(1, dilate)                        # need >=1 to bridge touching
    box = _crown_boxes(inst, ids, rad, H, W)
    ids = [i for i in ids if i in box]
    for ai in range(len(ids)):
        a = ids[ai]
        for bi in range(ai + 1, len(ids)):
            b = ids[bi]
            res = _pair_contest(inst, a, b, box[a], box[b], rad)
            if res is not None and res[2].any():
                pairs.add((a, b))
    return pairs


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    root = repo_root()
    tag = args.tag or "_".join(args.zones)
    out_dir = args.out_dir or osp.join(root, "dataset", "QuebecTree",
                                       "correlation_results")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    def paths(z):
        d = osp.join(root, "dataset", "QuebecTree")
        return (osp.join(d, "depth_tiles", z),
                osp.join(d, "chm_tiles", z),
                osp.join(d, "crown_tiles", z))

    # gather usable tiles across zones
    work = []   # (zone, tile_id, depth_dir, chm_dir, crown_dir)
    for z in args.zones:
        dd, cd, rd = paths(z)
        cov = json.load(open(osp.join(cd, "coverage.json")))["finite_frac"]
        for tid, f in cov.items():
            if (f >= args.min_cov
                    and osp.isfile(osp.join(dd, f"{tid}_depth.npy"))
                    and osp.isfile(osp.join(cd, f"{tid}_chm.npy"))):
                work.append((z, tid, dd, cd, rd))
    print(f"[{tag}] usable tiles: {len(work)} (cov>={args.min_cov})")
    if not work:
        raise SystemExit("no usable tiles -- run generate_depth + crop_chm + "
                         "rasterize_crowns first")

    # accumulators
    px_rho, px_r, px_absrel, px_rmse, px_s = [], [], [], [], []
    pool_d, pool_h = [], []
    crown_pts = {}          # objectid -> (dbar, hbar, npix)  (best tile)
    within_tile_rho = []
    pair_total = pair_correct = 0
    strat_pool = {"boundary": ([], []), "interior": ([], [])}

    for z, tid, dd, cd, rd in work:
        d = np.load(osp.join(dd, f"{tid}_depth.npy")).astype(np.float32)
        h = np.load(osp.join(cd, f"{tid}_chm.npy")).astype(np.float32)
        if d.shape != h.shape:
            import cv2
            d = cv2.resize(d, (h.shape[1], h.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        m = np.isfinite(d) & np.isfinite(h)
        if m.sum() < 500:
            continue
        s, t, inl = robust_scale_shift(d[m], h[m], args.trim, args.align_iters)
        d_al = s * d + t                       # depth -> metres

        # ---- Exp1 pixel-level (per tile) ----
        dv, hv = d_al[m], h[m]
        k = min(args.px_per_tile, dv.size)
        idx = rng.choice(dv.size, size=k, replace=False)
        ds_, hs_ = dv[idx], hv[idx]
        r, rho = corr_block(ds_, hs_)          # corr on a bounded subsample
        px_rho.append(rho); px_r.append(r); px_s.append(s)
        pos = hv > 0.5                          # AbsRel/RMSE on all valid px
        px_absrel.append(float(np.mean(np.abs(dv[pos] - hv[pos]) / hv[pos]))
                         if pos.any() else np.nan)
        px_rmse.append(float(np.sqrt(np.mean((dv - hv) ** 2))))
        pool_d.append(ds_); pool_h.append(hs_)

        # ---- crown-dependent experiments ----
        ipath = osp.join(rd, f"{tid}_inst.npy")
        if not osp.isfile(ipath):
            continue
        inst = np.load(ipath)

        # per-crown canopy-top stats in this tile
        local = {}
        ids = [int(i) for i in np.unique(inst) if i > 0]
        for oid in ids:
            cm = (inst == oid) & m
            npix = int(cm.sum())
            if npix < args.crown_min_px:
                continue
            dbar = agg_stat(d_al[cm], args.crown_agg)
            hbar = agg_stat(h[cm], args.crown_agg)
            local[oid] = (dbar, hbar, npix)
            # keep best (largest) measurement per crown for the global scatter
            if oid not in crown_pts or npix > crown_pts[oid][2]:
                crown_pts[oid] = (dbar, hbar, npix)

        # ---- Exp2 within-tile rho ----
        if len(local) >= 5:
            dd_ = np.array([v[0] for v in local.values()])
            hh_ = np.array([v[1] for v in local.values()])
            within_tile_rho.append(spearmanr(dd_, hh_)[0])

        # ---- Exp2 neighbour-pair ordering ----
        for a, b in adjacency_pairs(inst, args.neighbor_dilate):
            if a in local and b in local:
                da, ha = local[a][0], local[a][1]
                db, hb = local[b][0], local[b][1]
                if abs(ha - hb) < 0.5:        # ambiguous true order -> skip
                    continue
                pair_total += 1
                if np.sign(da - db) == np.sign(ha - hb):
                    pair_correct += 1

        # ---- Exp3 overlap stratification ----
        bnd = crown_crown_boundary(inst, args.boundary_r) & m
        interior = (inst > 0) & m & ~bnd
        for name, sm in (("boundary", bnd), ("interior", interior)):
            nn = int(sm.sum())
            if nn < 50:
                continue
            kk = min(args.px_per_tile, nn)
            ii = rng.choice(nn, size=kk, replace=False)
            strat_pool[name][0].append(d_al[sm][ii])
            strat_pool[name][1].append(h[sm][ii])

    # ===================== aggregate + report ============================
    px_rho = np.array(px_rho); px_r = np.array(px_r)
    px_absrel = np.array(px_absrel); px_rmse = np.array(px_rmse)
    px_s = np.array(px_s)
    pool_d = np.concatenate(pool_d); pool_h = np.concatenate(pool_h)
    pooled_r, pooled_rho = corr_block(pool_d, pool_h)

    exp1 = {
        "n_tiles": int(px_rho.size),
        "spearman_rho_mean": float(np.nanmean(px_rho)),
        "spearman_rho_std": float(np.nanstd(px_rho)),
        "pearson_r_mean": float(np.nanmean(px_r)),
        "absrel_mean": float(np.nanmean(px_absrel)),
        "rmse_m_mean": float(np.nanmean(px_rmse)),
        "pooled_spearman": float(pooled_rho),
        "pooled_pearson": float(pooled_r),
        "frac_positive_s": float(np.mean(px_s > 0)),
    }

    # Exp2 global
    oids = sorted(crown_pts)
    cd_ = np.array([crown_pts[o][0] for o in oids])
    ch_ = np.array([crown_pts[o][1] for o in oids])
    g_r, g_rho = corr_block(cd_, ch_)
    exp2 = {
        "n_crowns": int(len(oids)),
        "global_spearman": float(g_rho),
        "global_pearson": float(g_r),
        "within_tile_spearman_mean": float(np.nanmean(within_tile_rho))
        if within_tile_rho else float("nan"),
        "within_tile_spearman_std": float(np.nanstd(within_tile_rho))
        if within_tile_rho else float("nan"),
        "neighbor_pairs": int(pair_total),
        "neighbor_order_accuracy": float(pair_correct / pair_total)
        if pair_total else float("nan"),
        "crown_agg": args.crown_agg,
    }

    # Exp3
    exp3 = {}
    for name in ("boundary", "interior"):
        if strat_pool[name][0]:
            xd = np.concatenate(strat_pool[name][0])
            xh = np.concatenate(strat_pool[name][1])
            rr, rho = corr_block(xd, xh)
            exp3[name] = {"n_px": int(xd.size), "spearman": float(rho),
                          "pearson": float(rr)}
        else:
            exp3[name] = {"n_px": 0, "spearman": float("nan"),
                          "pearson": float("nan")}

    results = {"tag": tag, "zones": args.zones, "min_cov": args.min_cov,
               "exp1_pixel": exp1, "exp2_crown": exp2, "exp3_overlap": exp3}
    with open(osp.join(out_dir, f"results_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- console table ----
    print("\n================  RESULTS  ================")
    print(f"Exp1 PIXEL   rho={exp1['spearman_rho_mean']:.3f}"
          f"±{exp1['spearman_rho_std']:.3f}  r={exp1['pearson_r_mean']:.3f}  "
          f"AbsRel={exp1['absrel_mean']:.3f}  RMSE={exp1['rmse_m_mean']:.2f}m  "
          f"(pooled rho={exp1['pooled_spearman']:.3f}, "
          f"s>0 in {exp1['frac_positive_s']*100:.0f}% tiles)")
    print(f"Exp2 CROWN   global rho={exp2['global_spearman']:.3f} "
          f"r={exp2['global_pearson']:.3f}  n={exp2['n_crowns']}  | "
          f"within-tile rho={exp2['within_tile_spearman_mean']:.3f}  | "
          f"neighbour order acc={exp2['neighbor_order_accuracy']*100:.1f}% "
          f"(n={exp2['neighbor_pairs']})")
    print(f"Exp3 OVERLAP boundary rho={exp3['boundary']['spearman']:.3f} "
          f"(n={exp3['boundary']['n_px']})  vs  "
          f"interior rho={exp3['interior']['spearman']:.3f} "
          f"(n={exp3['interior']['n_px']})")
    print("==========================================\n")

    wrote = [f"results_{tag}.json"]
    if cd_.size == 0:
        print("WARNING: no crowns found -> skipping Exp2/Exp3 figures. "
              "Did you run tools/chm/rasterize_crowns.py for these zones?")

    # ---- Exp2 scatter ----
    if cd_.size >= 2:
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        ax.scatter(cd_, ch_, s=8, alpha=0.35, c="steelblue", edgecolors="none")
        lim = float(np.nanpercentile(np.concatenate([cd_, ch_]), 99)) + 2
        lim = lim if np.isfinite(lim) and lim > 0 else 30.0
        b1, b0 = np.polyfit(cd_, ch_, 1)
        xs = np.array([0, lim])
        ax.plot(xs, b1 * xs + b0, "r-", lw=1.5, label="fit")
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="y=x")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f"DAv2 {args.crown_agg} -> metres (per crown)")
        ax.set_ylabel(f"CHM {args.crown_agg} (m, per crown)")
        ax.set_title(f"Exp2 crown-level: rho={g_rho:.2f}  r={g_r:.2f}  "
                     f"n={len(oids)} crowns")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(osp.join(out_dir, f"exp2_crown_scatter_{tag}.png"), dpi=130)
        plt.close(fig)
        wrote.append(f"exp2_crown_scatter_{tag}.png")

    # ---- Exp3 bar ----
    names = ["interior", "boundary"]
    vals = [exp3[n]["spearman"] for n in names]
    if any(np.isfinite(v) for v in vals):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        pv = [v if np.isfinite(v) else 0.0 for v in vals]
        ax.bar(names, pv, color=["#4c72b0", "#dd8452"], edgecolor="k")
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, pv[i] + 0.01, f"{v:.2f}", ha="center", fontsize=11)
        ax.set_ylabel("Spearman rho (DAv2 vs CHM)")
        ax.set_ylim(0, max(0.1, max(pv) + 0.1))
        ax.set_title("Exp3: correlation by overlap stratum\n"
                     "(boundary holding up => depth helps where RGB is ambiguous)")
        fig.tight_layout()
        fig.savefig(osp.join(out_dir, f"exp3_overlap_bar_{tag}.png"), dpi=130)
        plt.close(fig)
        wrote.append(f"exp3_overlap_bar_{tag}.png")

    print(f"wrote {', '.join(wrote)}  -> {out_dir}")


if __name__ == "__main__":
    main()
