"""Visualise the CHM pipeline: zone overview + per-tile RGB/CHM alignment.

Produces two PNGs (matplotlib):

  <zone>_overview.png : RGB | CHM | CHM with the tile grid + survey footprint
  <zone>_tiles.png    : for N sample tiles, RGB | CHM | CHM contours over RGB
                        (visually confirms tall canopy in RGB == high CHM)

RGB is read on the fly from the COG via each tile's col_off/row_off in the
manifest, so it does not require tile_grid.py --dump-rgb.

Usage::

    python tools/chm/viz_chm.py --zone zone3                 # both figures
    python tools/chm/viz_chm.py --zone zone3 --mode tiles --n-tiles 6
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
from matplotlib.patches import Rectangle


def repo_root():
    return osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


DEFAULT_COG = ("/mnt/hoangdmv/Research/Instance_segmentation/dataset/"
               "quebec_trees_dataset_2021-09-02/2021-09-02/{z}/"
               "2021-09-02-sbl-z{n}-rgb-cog.tif")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zone", required=True, choices=["zone1", "zone2", "zone3"])
    p.add_argument("--mode", default="both",
                   choices=["overview", "tiles", "both"])
    p.add_argument("--n-tiles", type=int, default=6)
    p.add_argument("--vmax", type=float, default=28.0,
                   help="CHM colour scale max (m)")
    p.add_argument("--max-px", type=int, default=1400,
                   help="overview long-edge downsample target")
    p.add_argument("--chm", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--cog", default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def load_paths(args):
    root = repo_root()
    z, n = args.zone, args.zone[-1]
    chm = args.chm or osp.join(root, "dataset", "QuebecTree", "chm",
                               f"{z}_chm.tif")
    man = args.manifest or osp.join(root, "dataset", "QuebecTree",
                                    f"tiles_manifest_{z}.json")
    cog = args.cog or DEFAULT_COG.format(z=z, n=n)
    out = args.out_dir or osp.join(root, "dataset", "QuebecTree", "chm", "viz")
    for pth in (chm, man, cog):
        if not osp.isfile(pth):
            raise FileNotFoundError(pth)
    os.makedirs(out, exist_ok=True)
    return chm, man, cog, out


def make_overview(args, chm_path, man, cog, out):
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(chm_path) as ds:
        H, W = ds.height, ds.width
        scale = args.max_px / max(H, W)
        oh, ow = int(H * scale), int(W * scale)
        chm = ds.read(1, out_shape=(oh, ow), resampling=Resampling.bilinear)
    with rasterio.open(cog) as ds:
        cH, cW = ds.height, ds.width
        rgb = ds.read((1, 2, 3), out_shape=(3, oh, ow),
                      resampling=Resampling.bilinear)
        rgb = np.transpose(rgb, (1, 2, 0))

    chm_m = np.ma.masked_invalid(chm)

    fig, ax = plt.subplots(1, 3, figsize=(21, 7.2))
    ax[0].imshow(rgb)
    ax[0].set_title(f"{args.zone}  RGB ortho")
    im = ax[1].imshow(chm_m, cmap="viridis", vmin=0, vmax=args.vmax)
    ax[1].set_title("CHM (m)  — gray = no SfM data")
    ax[1].set_facecolor("0.8")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="height (m)")

    ax[2].imshow(chm_m, cmap="viridis", vmin=0, vmax=args.vmax)
    ax[2].set_facecolor("0.8")
    ax[2].set_title(f"tile grid  (kept {man['n_kept']}/{man['n_total']}, "
                    f"green=kept)")
    # all-grid rectangles; manifest holds only kept tiles
    kept = {r["tile_id"] for r in man["tiles"]}
    tile, stride = man["tile"], man["stride"]
    sx, sy = ow / cW, oh / cH
    nr, nc = man["grid"]["nrows"], man["grid"]["ncols"]
    for ri in range(nr):
        for ci in range(nc):
            tid = f"{args.zone}_r{ri:02d}_c{ci:02d}"
            x = ci * stride * sx
            y = ri * stride * sy
            kp = tid in kept
            ax[2].add_patch(Rectangle(
                (x, y), tile * sx, tile * sy, fill=False,
                edgecolor=("lime" if kp else "red"),
                linewidth=0.6, alpha=0.8 if kp else 0.4))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    path = osp.join(out, f"{args.zone}_overview.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def make_tiles(args, chm_path, man, cog, out):
    import rasterio
    from rasterio.windows import Window

    root = repo_root()
    cov_path = osp.join(root, "dataset", "QuebecTree", "chm_tiles",
                        args.zone, "coverage.json")
    tiles_by_id = {r["tile_id"]: r for r in man["tiles"]}
    chm_dir = osp.join(root, "dataset", "QuebecTree", "chm_tiles", args.zone)

    if osp.isfile(cov_path):
        cov = json.load(open(cov_path))["finite_frac"]
        # pick a spread of well-covered tiles
        ranked = sorted(cov.items(), key=lambda kv: -kv[1])
        ranked = [t for t, _ in ranked if osp.isfile(
            osp.join(chm_dir, f"{t}_chm.npy"))]
        step = max(1, len(ranked) // args.n_tiles)
        chosen = ranked[::step][:args.n_tiles]
    else:
        chosen = [osp.basename(p)[:-8] for p in
                  sorted(os.listdir(chm_dir)) if p.endswith("_chm.npy")
                  ][:args.n_tiles]

    n = len(chosen)
    fig, ax = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        ax = ax[None, :]
    with rasterio.open(cog) as ds:
        for i, tid in enumerate(chosen):
            rec = tiles_by_id[tid]
            win = Window(rec["col_off"], rec["row_off"],
                         rec["width"], rec["height"])
            rgb = np.transpose(ds.read((1, 2, 3), window=win), (1, 2, 0))
            chm = np.load(osp.join(chm_dir, f"{tid}_chm.npy"))
            chm_m = np.ma.masked_invalid(chm)

            ax[i, 0].imshow(rgb)
            ax[i, 0].set_ylabel(tid, fontsize=9)
            ax[i, 0].set_title("RGB" if i == 0 else "")
            im = ax[i, 1].imshow(chm_m, cmap="viridis", vmin=0, vmax=args.vmax)
            ax[i, 1].set_facecolor("0.8")
            ax[i, 1].set_title("CHM (m)" if i == 0 else "")
            fig.colorbar(im, ax=ax[i, 1], fraction=0.046, pad=0.04)

            ax[i, 2].imshow(rgb)
            if np.isfinite(chm).any():
                levels = np.linspace(2, args.vmax, 7)
                ax[i, 2].contour(chm_m, levels=levels, cmap="autumn",
                                 linewidths=0.7, alpha=0.9)
            ax[i, 2].set_title("CHM contours over RGB" if i == 0 else "")
            for j in range(3):
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle(f"{args.zone}: RGB ↔ CHM tile alignment", y=1.0)
    fig.tight_layout()
    path = osp.join(out, f"{args.zone}_tiles.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    args = parse_args()
    chm_path, man_path, cog, out = load_paths(args)
    man = json.load(open(man_path))
    if args.mode in ("overview", "both"):
        make_overview(args, chm_path, man, cog, out)
    if args.mode in ("tiles", "both"):
        make_tiles(args, chm_path, man, cog, out)
    print("done.")


if __name__ == "__main__":
    main()
