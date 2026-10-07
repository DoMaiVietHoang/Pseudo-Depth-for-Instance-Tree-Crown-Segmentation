"""Compare Depth-Anything-V2 pseudo-depth against the reference CHM, per tile.

This is the qualitative+quantitative core of the "is DAv2 depth a real height
signal?" check. For each tile it:

  1. runs DAv2 on the (georeferenced) RGB tile -> relative inverse-depth,
  2. solves the per-tile scale-and-shift  min_{s,t} ||s*d + t - CHM||^2  on the
     pixels where CHM is finite (DAv2 is affine-invariant, so raw values are NOT
     comparable to metres -- alignment is mandatory),
  3. reports Pearson r, Spearman rho, AbsRel and RMSE (post-align),
  4. draws RGB | DAv2 raw | DAv2->metres | CHM | scatter.

Also writes an aggregate figure over many tiles (pooled 2-D histogram + the
distribution of per-tile Spearman rho).

Run offline with the locally cached model:
    HF_HOME=dav_base_pretrained HF_HUB_OFFLINE=1 \
    python tools/chm/viz_depth_chm.py --zone zone3 --model dav2_base \
        --n-tiles 6 --agg-n 60
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


def repo_root():
    return osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


DEFAULT_COG = ("/mnt/hoangdmv/Research/Instance_segmentation/dataset/"
               "quebec_trees_dataset_2021-09-02/2021-09-02/{z}/"
               "2021-09-02-sbl-z{n}-rgb-cog.tif")
HUB_ID = {
    "dav2_small": "depth-anything/Depth-Anything-V2-Small-hf",
    "dav2_base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "dav2_large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zone", required=True, choices=["zone1", "zone2", "zone3"])
    p.add_argument("--model", default="dav2_base",
                   choices=list(HUB_ID))
    p.add_argument("--n-tiles", type=int, default=6, help="rows in the montage")
    p.add_argument("--agg-n", type=int, default=60,
                   help="tiles pooled for the aggregate correlation figure")
    p.add_argument("--min-cov", type=float, default=0.6,
                   help="only use tiles with >= this CHM coverage")
    p.add_argument("--vmax", type=float, default=28.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def scale_shift(d, h, m):
    """Least-squares s,t for s*d+t ~ h over mask m. Returns (s,t)."""
    dv, hv = d[m], h[m]
    A = np.stack([dv, np.ones_like(dv)], 1)
    st, *_ = np.linalg.lstsq(A, hv, rcond=None)
    return float(st[0]), float(st[1])


def metrics(d_aligned, h, m):
    from scipy.stats import pearsonr, spearmanr
    dv, hv = d_aligned[m], h[m]
    r = pearsonr(dv, hv)[0]
    rho = spearmanr(dv, hv)[0]
    pos = hv > 0.5
    absrel = float(np.mean(np.abs(dv[pos] - hv[pos]) / hv[pos])) if pos.any() else np.nan
    rmse = float(np.sqrt(np.mean((dv - hv) ** 2)))
    return r, rho, absrel, rmse


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    import torch
    import rasterio
    from rasterio.windows import Window
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    root = repo_root()
    z, n = args.zone, args.zone[-1]
    cog = DEFAULT_COG.format(z=z, n=n)
    man = json.load(open(osp.join(root, "dataset", "QuebecTree",
                                  f"tiles_manifest_{z}.json")))
    tiles_by_id = {r["tile_id"]: r for r in man["tiles"]}
    chm_dir = osp.join(root, "dataset", "QuebecTree", "chm_tiles", z)
    cov = json.load(open(osp.join(chm_dir, "coverage.json")))["finite_frac"]
    out = args.out_dir or osp.join(root, "dataset", "QuebecTree", "chm", "viz")
    os.makedirs(out, exist_ok=True)

    usable = [t for t, f in cov.items()
              if f >= args.min_cov and osp.isfile(osp.join(chm_dir, f"{t}_chm.npy"))]
    usable.sort()
    print(f"[{z}] usable tiles (cov>={args.min_cov}): {len(usable)}")

    # --- DAv2 ---------------------------------------------------------------
    hub = HUB_ID[args.model]
    proc = AutoImageProcessor.from_pretrained(hub)
    model = AutoModelForDepthEstimation.from_pretrained(hub).to(args.device).eval()

    def dav2(rgb):
        inp = proc(images=rgb, return_tensors="pt").to(args.device)
        with torch.no_grad():
            d = model(**inp).predicted_depth
        d = torch.nn.functional.interpolate(
            d[:, None], size=rgb.shape[:2], mode="bicubic",
            align_corners=False)[0, 0].float().cpu().numpy()
        return d

    def load_pair(tid, ds):
        rec = tiles_by_id[tid]
        win = Window(rec["col_off"], rec["row_off"], rec["width"], rec["height"])
        rgb = np.transpose(ds.read((1, 2, 3), window=win), (1, 2, 0))
        chm = np.load(osp.join(chm_dir, f"{tid}_chm.npy"))
        d = dav2(rgb)
        m = np.isfinite(chm) & np.isfinite(d)
        return rec, rgb, d, chm, m

    # --- montage of N tiles --------------------------------------------------
    show = usable[:: max(1, len(usable) // args.n_tiles)][:args.n_tiles]
    nN = len(show)
    fig, ax = plt.subplots(nN, 5, figsize=(20, 4 * nN))
    if nN == 1:
        ax = ax[None, :]
    with rasterio.open(cog) as ds:
        for i, tid in enumerate(show):
            rec, rgb, d, chm, m = load_pair(tid, ds)
            s, t = scale_shift(d, chm, m)
            d_al = s * d + t
            r, rho, absrel, rmse = metrics(d_al, chm, m)

            ax[i, 0].imshow(rgb); ax[i, 0].set_ylabel(tid, fontsize=8)
            ax[i, 1].imshow(np.ma.masked_invalid(d), cmap="magma")
            ax[i, 2].imshow(np.ma.masked_invalid(np.where(m, d_al, np.nan)),
                            cmap="viridis", vmin=0, vmax=args.vmax)
            ax[i, 3].imshow(np.ma.masked_invalid(chm), cmap="viridis",
                            vmin=0, vmax=args.vmax)
            sub = rng.choice(int(m.sum()), size=min(4000, int(m.sum())),
                             replace=False)
            ax[i, 4].scatter(d_al[m][sub], chm[m][sub], s=2, alpha=0.25,
                             c="steelblue")
            lim = max(args.vmax, 5)
            ax[i, 4].plot([0, lim], [0, lim], "r--", lw=1)
            ax[i, 4].set_xlim(0, lim); ax[i, 4].set_ylim(0, lim)
            ax[i, 4].set_title(f"r={r:.2f}  rho={rho:.2f}\n"
                               f"AbsRel={absrel:.2f} RMSE={rmse:.1f}m",
                               fontsize=9)
            ax[i, 4].set_xlabel("DAv2->m"); ax[i, 4].set_ylabel("CHM m")
            if i == 0:
                for j, ttl in enumerate(["RGB", "DAv2 raw (relative)",
                                         "DAv2 -> metres (scale-shift)",
                                         "CHM (m)"]):
                    ax[0, j].set_title(ttl)
            for j in range(4):
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        fig.suptitle(f"{z}: DAv2 ({args.model}) vs CHM — per-tile scale-shift",
                     y=1.0)
        fig.tight_layout()
        p1 = osp.join(out, f"{z}_depth_chm_tiles.png")
        fig.savefig(p1, dpi=110, bbox_inches="tight"); plt.close(fig)
        print("wrote", p1)

        # --- aggregate over many tiles --------------------------------------
        agg = usable if len(usable) <= args.agg_n else \
            list(rng.choice(usable, size=args.agg_n, replace=False))
        rhos, rs, absrels = [], [], []
        pool_d, pool_h = [], []
        for tid in agg:
            _, _, d, chm, m = load_pair(tid, ds)
            if m.sum() < 1000:
                continue
            s, t = scale_shift(d, chm, m)
            d_al = s * d + t
            r, rho, absrel, rmse = metrics(d_al, chm, m)
            rs.append(r); rhos.append(rho); absrels.append(absrel)
            idx = rng.choice(int(m.sum()), size=min(2000, int(m.sum())),
                             replace=False)
            pool_d.append(d_al[m][idx]); pool_h.append(chm[m][idx])

    rhos = np.array(rhos); rs = np.array(rs); absrels = np.array(absrels)
    pool_d = np.concatenate(pool_d); pool_h = np.concatenate(pool_h)
    print(f"[{z}] aggregate over {len(rhos)} tiles: "
          f"Spearman rho={np.nanmean(rhos):.3f}+-{np.nanstd(rhos):.3f}  "
          f"Pearson r={np.nanmean(rs):.3f}  AbsRel={np.nanmean(absrels):.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.3))
    hb = ax[0].hist2d(pool_d, pool_h, bins=120,
                      range=[[0, args.vmax], [0, args.vmax]], cmap="inferno",
                      cmin=1)
    ax[0].plot([0, args.vmax], [0, args.vmax], "w--", lw=1)
    ax[0].set_xlabel("DAv2 -> metres (per-tile aligned)")
    ax[0].set_ylabel("CHM (m)")
    ax[0].set_title(f"pooled pixels ({len(rhos)} tiles)\n"
                    f"mean Spearman rho={np.nanmean(rhos):.2f}")
    fig.colorbar(hb[3], ax=ax[0], label="pixels")
    ax[1].hist(rhos, bins=20, range=(-0.2, 1.0), color="seagreen",
               edgecolor="k", alpha=0.8)
    ax[1].axvline(np.nanmean(rhos), color="r", ls="--",
                  label=f"mean={np.nanmean(rhos):.2f}")
    ax[1].set_xlabel("per-tile Spearman rho (DAv2 vs CHM)")
    ax[1].set_ylabel("# tiles"); ax[1].legend()
    ax[1].set_title("distribution of per-tile rank correlation")
    fig.tight_layout()
    p2 = osp.join(out, f"{z}_depth_chm_agg.png")
    fig.savefig(p2, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("wrote", p2)
    print("done.")


if __name__ == "__main__":
    main()
