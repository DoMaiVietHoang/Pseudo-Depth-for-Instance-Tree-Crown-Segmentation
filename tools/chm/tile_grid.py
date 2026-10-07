"""Geo-aware tiler for the QuebecTree RGB COGs.

This fixes the central blocker: the existing ``{N}_tile_{XXXX}.jpg`` tiles carry
NO georeferencing (the COCO converter only stored file_name/width/height), so a
depth tile cannot be matched to a CHM patch. Here we re-tile the RGB COG on a
deterministic grid and record, for every tile, the exact affine transform and
geographic bounds. The resulting manifest lets any raster (CHM/DSM) be cropped
to the identical footprint as a tile (see crop_chm_to_tiles.py).

Tiles whose valid (alpha>0) coverage is below ``--min-valid`` are skipped, so we
do not emit empty border tiles.

Outputs (under <root>/dataset/QuebecTree/):
    tiles/<zone>/<zone>_r{row}_c{col}.jpg        # RGB tiles (if --dump-rgb)
    tiles_manifest_<zone>.json                   # list of tile records

Each manifest record::

    {
      "tile_id": "zone1_r03_c07",
      "zone": "zone1",
      "file_name": "zone1_r03_c07.jpg",
      "col_off": 7168, "row_off": 3072, "width": 1024, "height": 1024,
      "transform": [a, b, c, d, e, f],          # rasterio affine, row-major
      "bounds": [minx, miny, maxx, maxy],        # UTM (EPSG:32618)
      "crs": "EPSG:32618",
      "res": 0.0189,
      "valid_frac": 1.0
    }

Usage::

    python tools/chm/tile_grid.py --zone zone1 --tile 1024 --stride 1024 \
        --dump-rgb
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp

import numpy as np

DEFAULT_ROOT = ("/mnt/hoangdmv/Research/Instance_segmentation/dataset/"
                "quebec_trees_dataset_2021-09-02/2021-09-02")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zone", required=True, choices=["zone1", "zone2", "zone3"])
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--out-dir", default=None,
                   help="default: <repo>/dataset/QuebecTree")
    p.add_argument("--tile", type=int, default=1024, help="tile size (px)")
    p.add_argument("--stride", type=int, default=1024,
                   help="step between tiles (px); <tile for overlap")
    p.add_argument("--min-valid", type=float, default=0.5,
                   help="skip tiles with valid(alpha>0) fraction below this")
    p.add_argument("--dump-rgb", action="store_true",
                   help="also write the RGB jpg tiles (drops alpha)")
    p.add_argument("--jpg-quality", type=int, default=95)
    return p.parse_args()


def main():
    args = parse_args()
    import rasterio
    from rasterio.windows import Window
    from PIL import Image

    zn = args.zone[-1]
    cog = osp.join(args.root, args.zone, f"2021-09-02-sbl-z{zn}-rgb-cog.tif")
    if not osp.isfile(cog):
        raise FileNotFoundError(cog)

    out_root = args.out_dir or osp.join(
        osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
        "dataset", "QuebecTree")
    tile_dir = osp.join(out_root, "tiles", args.zone)
    if args.dump_rgb:
        os.makedirs(tile_dir, exist_ok=True)

    records = []
    n_total = n_kept = 0
    with rasterio.open(cog) as ds:
        crs_str = str(ds.crs)
        res = float(ds.res[0])
        has_alpha = ds.count >= 4
        ncols = (ds.width - args.tile) // args.stride + 1
        nrows = (ds.height - args.tile) // args.stride + 1
        # extend by one partial step so the far edge is covered
        col_offs = list(range(0, ds.width - args.tile + 1, args.stride))
        row_offs = list(range(0, ds.height - args.tile + 1, args.stride))
        if col_offs and col_offs[-1] != ds.width - args.tile:
            col_offs.append(ds.width - args.tile)
        if row_offs and row_offs[-1] != ds.height - args.tile:
            row_offs.append(ds.height - args.tile)

        for ri, row_off in enumerate(row_offs):
            for ci, col_off in enumerate(col_offs):
                n_total += 1
                win = Window(col_off, row_off, args.tile, args.tile)
                # cheap validity check via the alpha band
                if has_alpha:
                    alpha = ds.read(4, window=win)
                    valid_frac = float((alpha > 0).mean())
                else:
                    valid_frac = 1.0
                if valid_frac < args.min_valid:
                    continue
                n_kept += 1

                t = ds.window_transform(win)
                wb = rasterio.windows.bounds(win, ds.transform)
                tile_id = f"{args.zone}_r{ri:02d}_c{ci:02d}"
                fname = f"{tile_id}.jpg"
                records.append({
                    "tile_id": tile_id,
                    "zone": args.zone,
                    "file_name": fname,
                    "col_off": int(col_off), "row_off": int(row_off),
                    "width": int(args.tile), "height": int(args.tile),
                    "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
                    "bounds": [wb[0], wb[1], wb[2], wb[3]],
                    "crs": crs_str, "res": res,
                    "valid_frac": round(valid_frac, 4),
                })

                if args.dump_rgb:
                    rgb = ds.read((1, 2, 3), window=win)  # (3,H,W)
                    rgb = np.transpose(rgb, (1, 2, 0))
                    Image.fromarray(rgb).save(
                        osp.join(tile_dir, fname), quality=args.jpg_quality)

    manifest = {
        "zone": args.zone,
        "source_cog": osp.abspath(cog),
        "crs": crs_str,
        "res": res,
        "tile": args.tile,
        "stride": args.stride,
        "grid": {"nrows": len(row_offs), "ncols": len(col_offs)},
        "n_total": n_total,
        "n_kept": n_kept,
        "tiles": records,
    }
    os.makedirs(out_root, exist_ok=True)
    out_json = osp.join(out_root, f"tiles_manifest_{args.zone}.json")
    with open(out_json, "w") as f:
        json.dump(manifest, f, indent=1)

    print(f"[{args.zone}] grid {len(row_offs)}x{len(col_offs)}  "
          f"kept {n_kept}/{n_total} tiles (min_valid={args.min_valid})")
    if args.dump_rgb:
        print(f"[{args.zone}] RGB tiles -> {tile_dir}")
    print(f"[{args.zone}] manifest  -> {out_json}")


if __name__ == "__main__":
    main()
