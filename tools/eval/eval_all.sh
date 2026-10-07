#!/usr/bin/env bash
# After all training runs are done, evaluate everything and aggregate results.
#
# For each work_dirs/* with a best_*.pth, runs:
#   1. tools/test.py (dumps results.pkl + scalars to vis_data)
#   2. tools/boundary_ap.py (writes boundary_ap.txt)
#   3. (optional) tools/eval/touching_crown_eval.py

set -euo pipefail

GT="${GT:-data/bamforest/annotations/val.json}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

shopt -s nullglob
for d in work_dirs/*; do
    [[ -d "$d" ]] || continue
    cfg_name="${d##*/}"
    cfg_name="${cfg_name%__seed*}"

    # Find config file
    cfg=""
    for cand in "configs/${cfg_name}.py" "configs/ablations/${cfg_name}.py"; do
        if [[ -f "$cand" ]]; then cfg="$cand"; break; fi
    done
    [[ -n "$cfg" ]] || { echo "[skip] no config for $d"; continue; }

    # Find best checkpoint
    ckpt=$(ls -1 "$d"/best_*.pth 2>/dev/null | head -1 || true)
    [[ -n "$ckpt" ]] || ckpt=$(ls -1 "$d"/epoch_*.pth 2>/dev/null | sort -V | tail -1 || true)
    [[ -n "$ckpt" ]] || { echo "[skip] no ckpt for $d"; continue; }

    pkl="$d/results.pkl"
    if [[ ! -f "$pkl" ]]; then
        echo "[test] $d"
        python tools/test.py "$cfg" "$ckpt" --out "$pkl" || { echo "[fail] test $d"; continue; }
    fi

    if [[ ! -f "$d/boundary_ap.txt" ]]; then
        echo "[boundary] $d"
        python tools/boundary_ap.py "$GT" "$pkl" > "$d/boundary_ap.txt" || true
    fi

    if [[ ! -f "$d/touching.txt" ]]; then
        echo "[touching] $d"
        python tools/eval/touching_crown_eval.py \
            --gt "$GT" --results "$pkl" \
            --out-gt "$d/val_touching.json" \
            --out-dets "$d/dets_coco.json" \
            > "$d/touching.txt" 2>&1 || true
    fi
done

echo "[aggregate]"
python tools/eval/aggregate_results.py "work_dirs/*" --out results.md --csv results.csv
echo "[done] see results.md / results.csv"
