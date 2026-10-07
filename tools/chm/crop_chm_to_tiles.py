"""Crop a zone CHM raster into per-tile .npy maps aligned to the RGB/depth tiles.

Reads the tile manifest from tile_grid.py and, for every tile, samples the CHM
GeoTIFF (built by build_chm.py) over the tile's exact geographic bounds, warping
to the tile's pixel grid (default 1024x1024). The output ``<tile_id>_chm.npy``
is then pixel-aligned with the RGB tile and the DAv2 depth tile, so depth<->CHM
correlation can be computed per pixel and per crown.

Pixels with no CHM data (outside the SfM surface / nodata) are written as NaN so
downstream code can mask them in the scale-shift alignment.

Usage::

    python tools/chm/crop_chm_to_tiles.py --zone zone1
    python tools/chm/crop_chm_to_tiles.py --zone zone1 \
        --chm dataset/QuebecTree/chm/zone1_chm.tif \
        --manifest dataset/QuebecTree/tiles_manifest_zone1.json \
        --out-dir dataset/QuebecTree/chm_tiles/zone1 --resampling bilinear
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

import numpy as np


def repo_root():
    return osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zone", required=True, choices=["zone1", "zone2", "zone3"])
    p.add_argument("--chm", default=None,
                   help="CHM GeoTIFF (default: dataset/QuebecTree/chm/"
                        "<zone>_chm.tif)")
    p.add_argument("--manifest", default=None,
                   help="tile manifest json (default: dataset/QuebecTree/"
                        "tiles_manifest_<zone>.json)")
    p.add_argument("--out-dir", default=None,
                   help="default: dataset/QuebecTree/chm_tiles/<zone>")
    p.add_argument("--resampling", default="bilinear",
                   choices=["nearest", "bilinear", "cubic", "average"])
    p.add_argument("--min-chm-frac", type=float, default=0.0,
                   help="skip tiles whose finite-CHM coverage is below this "
                        "(edge tiles outside the SfM footprint); 0=keep all")
    p.add_argument("--save-png", action="store_true",
                   help="also dump a normalized 8-bit PNG preview per tile")
    return p.parse_args()


def main():
    args = parse_args()
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds

    root = repo_root()
    chm_path = args.chm or osp.join(root, "dataset", "QuebecTree", "chm",
                                    f"{args.zone}_chm.tif")
    man_path = args.manifest or osp.join(root, "dataset", "QuebecTree",
                                         f"tiles_manifest_{args.zone}.json")
    out_dir = args.out_dir or osp.join(root, "dataset", "QuebecTree",
                                       "chm_tiles", args.zone)
    for pth in (chm_path, man_path):
        if not osp.isfile(pth):
            raise FileNotFoundError(pth)
    os.makedirs(out_dir, exist_ok=True)

    manifest = json.load(open(man_path))
    tiles = manifest["tiles"]
    resampling = getattr(Resampling, args.resampling)

    n_ok = n_empty = n_lowcov = 0
    coverage = {}
    with rasterio.open(chm_path) as src:
        for rec in tiles:
            minx, miny, maxx, maxy = rec["bounds"]
            H, W = rec["height"], rec["width"]
            win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            arr = src.read(
                1, window=win, out_shape=(H, W),
                resampling=resampling, boundless=True,
                fill_value=np.nan).astype(np.float32)
            finite = np.isfinite(arr)
            frac = float(finite.mean())
            if frac <= 0:
                n_empty += 1
                continue
            if frac < args.min_chm_frac:
                n_lowcov += 1
                continue
            n_ok += 1
            coverage[rec["tile_id"]] = round(frac, 4)
            np.save(osp.join(out_dir, f"{rec['tile_id']}_chm.npy"), arr)

            if args.save_png:
                from PIL import Image
                v = arr.copy()
                lo, hi = np.nanpercentile(v, [2, 98])
                v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
                v[~finite] = 0
                Image.fromarray((v * 255).astype(np.uint8)).save(
                    osp.join(out_dir, f"{rec['tile_id']}_chm.png"))

    with open(osp.join(out_dir, "coverage.json"), "w") as f:
        json.dump({"zone": args.zone, "chm": osp.abspath(chm_path),
                   "min_chm_frac": args.min_chm_frac,
                   "finite_frac": coverage}, f, indent=1)
    print(f"[{args.zone}] CHM tiles written: {n_ok}  "
          f"(empty: {n_empty}, below min_chm_frac: {n_lowcov})")
    print(f"[{args.zone}] coverage.json + tiles -> {out_dir}")


if __name__ == "__main__":
    main()
