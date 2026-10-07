"""Rasterize the GT crown polygons (Z*_polygons.gpkg) onto the canonical tiles.

Produces, per tile, an instance label map aligned pixel-for-pixel with the RGB /
depth / CHM tiles (same transform from the manifest), plus crown metadata. This
is what the crown-level (Exp 2) and overlap-stratified (Exp 3) correlation
experiments consume.

Each output pixel holds the polygon ``OBJECTID`` (0 = background). OBJECTID is
globally unique across a zone, so a crown spanning several tiles keeps the same
id and can be de-duplicated downstream.

Outputs (under dataset/QuebecTree/crown_tiles/<zone>/):
    <tile_id>_inst.npy     # int32 (H, W), 0=bg, else OBJECTID
    crowns.json            # {tile_id: [{objectid, species, area_px}], ...}
                           # + {"species_of": {objectid: species}}

Usage::

    python tools/chm/rasterize_crowns.py --zone zone3
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
    p.add_argument("--gpkg", default=None,
                   help="default: <dataset>/quebec_trees_dataset_2021-09-02/"
                        "Z<N>_polygons.gpkg")
    p.add_argument("--manifest", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--all-touched", action="store_true",
                   help="burn every pixel the polygon touches (thin crowns)")
    return p.parse_args()


def main():
    args = parse_args()
    import geopandas as gpd
    from shapely.geometry import box
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    root = repo_root()
    z, n = args.zone, args.zone[-1]
    gpkg = args.gpkg or osp.join(
        root, "dataset", "quebec_trees_dataset_2021-09-02",
        f"Z{n}_polygons.gpkg")
    man_path = args.manifest or osp.join(
        root, "dataset", "QuebecTree", f"tiles_manifest_{z}.json")
    out_dir = args.out_dir or osp.join(
        root, "dataset", "QuebecTree", "crown_tiles", z)
    for pth in (gpkg, man_path):
        if not osp.isfile(pth):
            raise FileNotFoundError(pth)
    os.makedirs(out_dir, exist_ok=True)

    man = json.load(open(man_path))
    gdf = gpd.read_file(gpkg).reset_index(drop=True)
    if str(gdf.crs).upper() != man["crs"].upper():
        gdf = gdf.to_crs(man["crs"])
    # pyogrio reads the gpkg fid as the index (not a column), so we assign our
    # own contiguous, positive crown ids (1..N) -- unique across the zone, so a
    # crown spanning tiles keeps the same id. 0 is reserved for background.
    oids = np.arange(1, len(gdf) + 1, dtype="int64")
    lab_col = next((c for c in ("Label", "label", "species") if c in gdf.columns),
                   None)
    geoms = gdf.geometry.values
    species = (gdf[lab_col].astype(str).values if lab_col is not None
               else np.array(["?"] * len(gdf)))
    sindex = gdf.sindex

    crowns = {}
    species_of = {}
    n_tiles = n_inst = 0
    for rec in man["tiles"]:
        minx, miny, maxx, maxy = rec["bounds"]
        H, W = rec["height"], rec["width"]
        t = Affine(*rec["transform"])
        cand = list(sindex.query(box(minx, miny, maxx, maxy)))
        if not cand:
            continue
        shapes = [(geoms[i], int(oids[i])) for i in cand]
        inst = rasterize(shapes, out_shape=(H, W), transform=t, fill=0,
                         all_touched=args.all_touched, dtype="int32")
        present, counts = np.unique(inst[inst > 0], return_counts=True)
        if present.size == 0:
            continue
        n_tiles += 1
        recs = []
        for oid, c in zip(present.tolist(), counts.tolist()):
            recs.append({"objectid": oid, "area_px": int(c)})
            n_inst += 1
        crowns[rec["tile_id"]] = recs
        np.save(osp.join(out_dir, f"{rec['tile_id']}_inst.npy"), inst)
    for i in range(len(gdf)):
        species_of[int(oids[i])] = str(species[i])

    with open(osp.join(out_dir, "crowns.json"), "w") as f:
        json.dump({"zone": z, "gpkg": osp.abspath(gpkg),
                   "tiles": crowns, "species_of": species_of}, f)
    print(f"[{z}] rasterised crowns on {n_tiles} tiles, "
          f"{n_inst} (tile,crown) instances")
    print(f"[{z}] -> {out_dir}")


if __name__ == "__main__":
    main()
