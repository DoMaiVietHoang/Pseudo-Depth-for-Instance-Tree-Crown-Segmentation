# CHM reference + geo-aware tiling (depth↔CHM validation)

Pipeline that unblocks the "is DAv2 pseudo-depth a real height signal?" study by
producing a **reference CHM** from the QuebecTree point clouds and a
**georeferenced tiling** whose footprints can be shared across RGB, depth and
CHM. Run everything in the `depthgate` conda env (needs `rasterio`, `laspy[lazrs]`,
`pyproj`, `scipy` — already installed).

```
PY=~/miniconda3/envs/depthgate/bin/python
```

## Why this exists

- DAv2 depth (`dataset/QuebecTree/depth_*/*.npy`) is **relative / affine-invariant**
  (values ~1–12), CHM is **metres**. They are only comparable *after a per-tile
  scale-and-shift alignment* — never compare raw pixel values.
- The existing `{N}_tile_XXXX.jpg` tiles carry **no georeferencing**
  (`quebectree_to_coco.py` only stored file_name/width/height), so a depth tile
  could not be matched to a CHM patch. This pipeline re-tiles the RGB COG and
  records each tile's affine transform + UTM bounds, so any raster can be cropped
  to the identical footprint.

## Data caveat (state in the paper)

The point clouds are **single-return, unclassified photogrammetric (SfM)
surfaces** (point format 7, classification all 0) — effectively a DSM. There are
no ground returns under closed canopy, so the DTM is **estimated** by a
morphological opening of the per-cell minimum Z, and CHM is biased low where
there are no canopy gaps. This is acceptable here because the per-tile
scale-shift removes a constant offset and crown-level *relative* height ordering
is preserved — but absolute metre errors should be reported with this caveat.

## Steps

### 1. Build DSM / DTM / CHM rasters (per zone)
Streams the full cloud once (~20 s/zone), writes GeoTIFFs on the COG grid
(EPSG:32618, top-left aligned), NaN nodata.
```bash
$PY tools/chm/build_chm.py --zone zone3 --res 0.2
# -> dataset/QuebecTree/chm/zone3_{dsm,dtm,chm}.tif
# tuning: --ground-open-m (crown size), --ground-pctl, --smooth-m, --max-height
# fast test: --bbox MINX MINY MAXX MAXY
```

### 2. Geo-aware tiler → manifest with per-tile transform
Deterministic grid over the COG; drops border tiles via the alpha band.
```bash
$PY tools/chm/tile_grid.py --zone zone3 --tile 1024 --stride 1024 --dump-rgb
# -> dataset/QuebecTree/tiles_manifest_zone3.json   (transform+bounds per tile)
# -> dataset/QuebecTree/tiles/zone3/*.jpg           (if --dump-rgb)
```
Each record has `transform` (rasterio affine), `bounds` (UTM), `col_off/row_off`,
`valid_frac`. **Alignment guarantee:** reading the COG by `Window(col_off,row_off)`
is bit-identical to reading by `bounds` — so cropping any raster by `bounds` is
co-registered pixel-for-pixel with the RGB tile.

### 3. Crop CHM into per-tile .npy (aligned to the tile pixel grid)
```bash
$PY tools/chm/crop_chm_to_tiles.py --zone zone3 --min-chm-frac 0.3
# -> dataset/QuebecTree/chm_tiles/zone3/<tile_id>_chm.npy   (1024x1024, NaN holes)
# -> dataset/QuebecTree/chm_tiles/zone3/coverage.json       (finite frac per tile)
```
NaN = no SfM data; mask these in the scale-shift least squares.

### 4. Visualise (sanity-check alignment)
```bash
$PY tools/chm/viz_chm.py --zone zone3 --mode both --n-tiles 6
# -> dataset/QuebecTree/chm/viz/zone3_overview.png  (RGB | CHM | tile grid)
# -> dataset/QuebecTree/chm/viz/zone3_tiles.png     (RGB | CHM | contours over RGB)
```

### 5. Compare DAv2 depth vs CHM (the actual signal check)
Runs DAv2 on the new RGB tiles, per-tile scale-shift to metres, reports
Pearson r / Spearman ρ / AbsRel / RMSE, and draws the comparison.
```bash
HF_HOME=dav_base_pretrained HF_HUB_OFFLINE=1 \
$PY tools/chm/viz_depth_chm.py --zone zone3 --model dav2_base --n-tiles 6 --agg-n 60
# -> chm/viz/zone3_depth_chm_tiles.png  (RGB | DAv2 raw | DAv2->m | CHM | scatter)
# -> chm/viz/zone3_depth_chm_agg.png    (pooled hist2d + per-tile ρ distribution)
```
zone3 result (60 tiles): pixel-level Spearman ρ ≈ 0.47±0.24, Pearson r ≈ 0.48,
AbsRel ≈ 0.51 — a real but moderate height signal; ~half the tiles have ρ>0.5,
a tail near 0 on homogeneous-canopy / water tiles.

## Outputs

| Path | What |
|---|---|
| `dataset/QuebecTree/chm/<zone>_{dsm,dtm,chm}.tif` | zone rasters (QGIS-viewable) |
| `dataset/QuebecTree/tiles_manifest_<zone>.json` | tile → transform/bounds/validity |
| `dataset/QuebecTree/tiles/<zone>/*.jpg` | RGB tiles (optional) |
| `dataset/QuebecTree/chm_tiles/<zone>/*_chm.npy` | per-tile CHM aligned to RGB/depth |

### 6. Rasterize GT crowns onto the tiles (for Exp 2 & 3)
```bash
$PY tools/chm/rasterize_crowns.py --zone zone3
# -> dataset/QuebecTree/crown_tiles/zone3/<tile_id>_inst.npy  (int32, 0=bg, else OBJECTID)
# -> dataset/QuebecTree/crown_tiles/zone3/crowns.json
```

### 7. Run the 3 correlation experiments
```bash
$PY tools/chm/correlation_experiments.py --zones zone3            # one zone
$PY tools/chm/correlation_experiments.py --zones zone1 zone2 zone3 --tag quebectree
# -> dataset/QuebecTree/correlation_results/results_<tag>.json
# -> exp2_crown_scatter_<tag>.png , exp3_overlap_bar_<tag>.png
```
- **Exp1 pixel**: per-tile LTS scale-shift → mean Spearman ρ, Pearson r, AbsRel, RMSE.
- **Exp2 crown**: per-crown p90(depth) vs p90(CHM) → global ρ scatter + within-tile ρ
  + **neighbour-pair ordering accuracy** (the overlap-relevant number).
- **Exp3 overlap**: pixel ρ on crown–crown boundary band vs crown interior.

### 8. Publication-grade figures
```bash
$PY tools/chm/viz_outstanding.py --zone zone3
# -> correlation_results/figs/zone3/
#    fig1_crown_joint.png    joint density (hexbin+marginals) crown depth vs CHM
#    fig2_overlap_curve.png  rho/AbsRel vs distance-to-crown-boundary (flat=good)
#    fig3_neighbor_pairs.png Δdepth vs ΔCHM quadrants + accuracy vs |ΔCHM|
#    fig4_spatial_rho.png    per-tile rho painted over the RGB ortho
#    fig5_hero.png           touching-crown tiles + scale-invariant transect
```

## Full pipeline (per zone), in order

```bash
PY=~/miniconda3/envs/depthgate/bin/python
HF=dataset/../dav_base_pretrained        # repo-root/dav_base_pretrained

$PY tools/chm/build_chm.py          --zone zoneX --res 0.2
$PY tools/chm/tile_grid.py          --zone zoneX --dump-rgb
$PY tools/chm/crop_chm_to_tiles.py  --zone zoneX --min-chm-frac 0.3
HF_HOME=$HF HF_HUB_OFFLINE=1 \
$PY tools/generate_depth.py --model dav2_base \
    --img-dir dataset/QuebecTree/tiles/zoneX \
    --out-dir dataset/QuebecTree/depth_tiles/zoneX
$PY tools/chm/rasterize_crowns.py   --zone zoneX
$PY tools/chm/correlation_experiments.py --zones zoneX
# optional visuals:
$PY tools/chm/viz_chm.py --zone zoneX
HF_HOME=$HF HF_HUB_OFFLINE=1 $PY tools/chm/viz_depth_chm.py --zone zoneX
```

Note: this uses the **fresh canonical tiling** here, not the original
`{N}_tile_XXXX` tiles (their geo-windows are unrecoverable). Depth is regenerated
with `tools/generate_depth.py` and masks with `rasterize_crowns.py` on this grid
so RGB / depth / CHM / GT masks are all co-registered.
