#!/usr/bin/env bash
# Profile every config in configs/ and configs/ablations/ to one table.
# Writes profile.md.

set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

OUT="profile.md"
{
    echo "| Config | Params (M) | Trainable (M) | FLOPs (G) | Latency (ms) |"
    echo "|---|---|---|---|---|"
} > "$OUT"

shopt -s nullglob
for cfg in configs/mask2former_r50_bamforest.py \
           configs/mask2former_r50_depthgate_bamforest.py \
           configs/ablations/*.py; do
    name=$(basename "${cfg%.py}")
    echo "[profile] $name"
    raw=$(python tools/eval/profile_model.py "$cfg" 2>&1 || true)
    pt=$(grep -oE "params_total\s*=\s*[0-9.]+" <<<"$raw" | awk '{print $NF}')
    pr=$(grep -oE "params_trainable\s*=\s*[0-9.]+" <<<"$raw" | awk '{print $NF}')
    fl=$(grep -oE "flops\s*=\s*[0-9.]+" <<<"$raw" | awk '{print $NF}')
    lt=$(grep -oE "latency.*=\s*[0-9.]+" <<<"$raw" | awk '{print $NF}')
    echo "| $name | ${pt:--} | ${pr:--} | ${fl:--} | ${lt:--} |" >> "$OUT"
done
echo "[done] $OUT"
