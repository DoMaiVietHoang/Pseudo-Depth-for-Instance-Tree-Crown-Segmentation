"""Build DSM / DTM / CHM rasters from a QuebecTree photogrammetric point cloud.

The QuebecTree point clouds (``*.copc.laz``) are *single-return, unclassified*
SfM surfaces (point format 7, classification all 0). They therefore behave like
a Digital Surface Model (canopy/structure top), NOT a multi-return LiDAR cloud
with ground points under the canopy. We derive:

    DSM  = per-cell maximum Z        (surface top)
    DTM  = morphological-opening of the per-cell minimum Z   (estimated ground)
    CHM  = DSM - DTM                 (canopy height above ground)

Caveat (write this in the paper): under closed canopy the SfM surface rarely
sees the true ground, so the morphological DTM is only an *estimate* and CHM is
biased low where there are no canopy gaps. For the depth<->CHM correlation study
this is acceptable because (a) the per-image scale-shift alignment removes a
constant offset, and (b) crown-level *relative* height ordering is preserved.

All three rasters are written on a grid that shares the RGB COG's CRS and
top-left origin, so they can be windowed by geographic bounds and resampled to
any tile's pixel grid (see tile_grid.py / crop_chm_to_tiles.py).

Usage::

    python tools/chm/build_chm.py --zone zone1 --res 0.2
    python tools/chm/build_chm.py --zone zone1 --res 0.2 \
        --bbox 577100 5093300 577300 5093500     # small bbox for testing
"""

from __future__ import annotations

import argparse
import math
import os
import os.path as osp
import time

import numpy as np

DEFAULT_ROOT = ("/mnt/hoangdmv/Research/Instance_segmentation/dataset/"
                "quebec_trees_dataset_2021-09-02/2021-09-02")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zone", required=True, choices=["zone1", "zone2", "zone3"])
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help="dir containing zone{1,2,3}/ with COG + copc.laz")
    p.add_argument("--out-dir", default=None,
                   help="output dir for the GeoTIFFs "
                        "(default: <repo>/dataset/QuebecTree/chm)")
    p.add_argument("--res", type=float, default=0.2,
                   help="CHM/DSM/DTM ground sampling distance in metres")
    p.add_argument("--chunk", type=int, default=8_000_000,
                   help="points per streaming chunk")
    p.add_argument("--ground-open-m", type=float, default=12.0,
                   help="morphological opening window (m) for DTM; ~= the "
                        "largest crown diameter so canopy is removed but "
                        "terrain relief is kept")
    p.add_argument("--ground-pctl", type=float, default=5.0,
                   help="percentile of per-cell Z used as the low surface "
                        "before opening (0=true min; 5 rejects low noise)")
    p.add_argument("--smooth-m", type=float, default=3.0,
                   help="median-filter window (m) applied to the DTM")
    p.add_argument("--max-height", type=float, default=60.0,
                   help="clamp CHM to [0, max] m (drops SfM blunders)")
    p.add_argument("--bbox", type=float, nargs=4, default=None,
                   metavar=("MINX", "MINY", "MAXX", "MAXY"),
                   help="optional UTM bbox to restrict processing (testing)")
    return p.parse_args()


def zone_paths(root: str, zone: str):
    zn = zone[-1]
    cog = osp.join(root, zone, f"2021-09-02-sbl-z{zn}-rgb-cog.tif")
    laz = osp.join(root, zone, f"2021-09-02-sbl-z{zn}.copc.laz")
    for pth in (cog, laz):
        if not osp.isfile(pth):
            raise FileNotFoundError(pth)
    return cog, laz


def main():
    args = parse_args()
    import rasterio
    from rasterio.transform import from_origin
    import laspy
    from laspy import CopcReader
    from laspy.copc import Bounds
    from scipy import ndimage

    cog, laz = zone_paths(args.root, args.zone)

    # --- grid definition: share the COG's CRS + top-left origin ---------------
    with rasterio.open(cog) as ds:
        crs = ds.crs
        b = ds.bounds
    if args.bbox:
        minx, miny, maxx, maxy = args.bbox
    else:
        minx, miny, maxx, maxy = b.left, b.bottom, b.right, b.top
    res = args.res
    # snap origin to the COG top-left so tiles line up cleanly
    left = b.left + math.floor((minx - b.left) / res) * res
    top = b.top - math.floor((b.top - maxy) / res) * res
    W = int(math.ceil((maxx - left) / res))
    H = int(math.ceil((top - miny) / res))
    transform = from_origin(left, top, res, res)
    print(f"[{args.zone}] grid {H} x {W} @ {res} m  origin=({left:.1f},{top:.1f})")

    NEG = np.float32(-1e9)
    POS = np.float32(1e9)
    dsm = np.full(H * W, NEG, np.float32)   # per-cell max Z
    zmin = np.full(H * W, POS, np.float32)  # per-cell min Z

    def accumulate(x, y, z):
        col = ((x - left) / res).astype(np.int64)
        row = ((top - y) / res).astype(np.int64)
        m = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        if not m.any():
            return 0
        idx = row[m] * W + col[m]
        zz = z[m].astype(np.float32)
        np.maximum.at(dsm, idx, zz)
        np.minimum.at(zmin, idx, zz)
        return int(m.sum())

    # --- stream the cloud -----------------------------------------------------
    t0 = time.time()
    n_used = 0
    if args.bbox:
        with CopcReader.open(laz) as cr:
            bounds = Bounds(mins=np.array([minx, miny]),
                            maxs=np.array([maxx, maxy]))
            pts = cr.query(bounds)
            n_used += accumulate(np.asarray(pts.x), np.asarray(pts.y),
                                 np.asarray(pts.z))
    else:
        with laspy.open(laz) as f:
            total = f.header.point_count
            done = 0
            for chunk in f.chunk_iterator(args.chunk):
                n_used += accumulate(np.asarray(chunk.x), np.asarray(chunk.y),
                                     np.asarray(chunk.z))
                done += len(chunk)
                print(f"\r  streamed {done/1e6:6.1f}M / {total/1e6:.1f}M pts",
                      end="", flush=True)
            print()
    print(f"[{args.zone}] rasterised {n_used/1e6:.1f}M pts in "
          f"{time.time()-t0:.1f}s")

    dsm = dsm.reshape(H, W)
    zmin = zmin.reshape(H, W)
    valid = dsm > NEG / 2          # cells that received >=1 point

    # --- DTM: fill -> percentile-low -> morphological opening -> smooth -------
    # fill empty cells by nearest valid (so the morphology has no holes)
    zfill = zmin.copy()
    zfill[~valid] = np.nan
    nan_mask = np.isnan(zfill)
    if nan_mask.any():
        idx = ndimage.distance_transform_edt(
            nan_mask, return_distances=False, return_indices=True)
        zfill = zfill[tuple(idx)]

    open_px = max(1, int(round(args.ground_open_m / res)))
    if args.ground_pctl > 0:
        # percentile-rank filter approximates a low surface robust to noise
        lo = ndimage.percentile_filter(zfill, args.ground_pctl,
                                        size=max(3, open_px // 4))
    else:
        lo = zfill
    # grey opening = erosion (rolling min) then dilation -> removes canopy,
    # keeps terrain relief at scales larger than open_px
    dtm = ndimage.grey_opening(lo, size=open_px)
    smooth_px = max(1, int(round(args.smooth_m / res)))
    if smooth_px > 1:
        dtm = ndimage.median_filter(dtm, size=smooth_px)
    dtm = dtm.astype(np.float32)

    # --- CHM ------------------------------------------------------------------
    chm = np.where(valid, dsm - dtm, np.nan).astype(np.float32)
    chm = np.clip(chm, 0.0, args.max_height)
    chm[~valid] = np.nan

    dsm_out = np.where(valid, dsm, np.nan).astype(np.float32)

    vchm = chm[np.isfinite(chm)]
    print(f"[{args.zone}] CHM stats: valid={valid.mean()*100:.1f}%  "
          f"p50={np.percentile(vchm,50):.1f}  p95={np.percentile(vchm,95):.1f}  "
          f"max={vchm.max():.1f} m")

    # --- write ----------------------------------------------------------------
    out_dir = args.out_dir or osp.join(
        osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
        "dataset", "QuebecTree", "chm")
    os.makedirs(out_dir, exist_ok=True)
    profile = dict(driver="GTiff", height=H, width=W, count=1,
                   dtype="float32", crs=crs, transform=transform,
                   nodata=np.nan, compress="deflate", predictor=3,
                   tiled=True, blockxsize=512, blockysize=512)
    suffix = "_bbox" if args.bbox else ""
    for name, arr in (("dsm", dsm_out), ("dtm", dtm), ("chm", chm)):
        path = osp.join(out_dir, f"{args.zone}_{name}{suffix}.tif")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)
        print(f"[{args.zone}] wrote {path}")
    print("done.")


if __name__ == "__main__":
    main()
