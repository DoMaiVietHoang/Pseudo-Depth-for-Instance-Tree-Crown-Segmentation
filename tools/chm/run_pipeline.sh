#!/usr/bin/env bash
# Full depth<->CHM pipeline for one or more QuebecTree zones.
#
#   bash tools/chm/run_pipeline.sh zone1 zone2
#   bash tools/chm/run_pipeline.sh zone1 zone2 zone3   # everything
#
# Steps per zone: build CHM -> geo-aware tiling -> crop CHM -> DAv2 depth ->
# rasterize GT crowns -> 3 correlation experiments -> 5 figures.
# After all zones, also runs the pooled multi-zone correlation table.
#
# Env knobs (override on the command line, e.g. RES=0.25 bash ...):
set -euo pipefail

# --- config ---------------------------------------------------------------
REPO="/mnt/hoangdmv/Research/Instance_segmentation"
PY="${PY:-/home/hoangdmv/miniconda3/envs/depthgate/bin/python}"
HF_HOME="${HF_HOME:-$REPO/dav_base_pretrained}"
export HF_HOME HF_HUB_OFFLINE=1
RES="${RES:-0.2}"             # CHM resolution (m)
MODEL="${MODEL:-dav2_base}"   # DAv2 variant (must be cached for offline)
MIN_CHM_FRAC="${MIN_CHM_FRAC:-0.3}"
MIN_COV="${MIN_COV:-0.6}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"   # 1 = reuse already-generated depth tiles

cd "$REPO"
ZONES=("$@")
if [ ${#ZONES[@]} -eq 0 ]; then
  echo "usage: bash tools/chm/run_pipeline.sh zone1 [zone2 ...]"; exit 1
fi

DEPTH_SKIP=""
[ "$SKIP_EXISTING" = "1" ] && DEPTH_SKIP="--skip-existing"

echo "=== zones: ${ZONES[*]} | model=$MODEL res=$RES ==="

for Z in "${ZONES[@]}"; do
  echo ""
  echo "############################  $Z  ############################"

  echo "--- [1/7] build CHM ($Z) ---"
  $PY tools/chm/build_chm.py --zone "$Z" --res "$RES"

  echo "--- [2/7] geo-aware tiling + dump RGB ($Z) ---"
  $PY tools/chm/tile_grid.py --zone "$Z" --tile 1024 --stride 1024 --dump-rgb

  echo "--- [3/7] crop CHM to tiles ($Z) ---"
  $PY tools/chm/crop_chm_to_tiles.py --zone "$Z" --min-chm-frac "$MIN_CHM_FRAC"

  echo "--- [4/7] DAv2 depth tiles ($Z) ---"
  $PY tools/generate_depth.py --model "$MODEL" $DEPTH_SKIP \
      --img-dir "dataset/QuebecTree/tiles/$Z" \
      --out-dir "dataset/QuebecTree/depth_tiles/$Z"

  echo "--- [5/7] rasterize GT crowns ($Z) ---"
  $PY tools/chm/rasterize_crowns.py --zone "$Z"

  echo "--- [6/7] correlation experiments ($Z) ---"
  $PY tools/chm/correlation_experiments.py --zones "$Z" --min-cov "$MIN_COV"

  echo "--- [7/7] publication figures ($Z) ---"
  $PY tools/chm/viz_outstanding.py --zone "$Z" --min-cov "$MIN_COV"
done

if [ ${#ZONES[@]} -gt 1 ]; then
  echo ""
  echo "############  pooled multi-zone correlation  ############"
  $PY tools/chm/correlation_experiments.py --zones "${ZONES[@]}" \
      --tag quebectree --min-cov "$MIN_COV"
fi

echo ""
echo "DONE. Results -> dataset/QuebecTree/correlation_results/"
