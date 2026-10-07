#!/usr/bin/env bash
# Orchestrate the full ablation study.
#
# Strategy (for ~1 GPU × 2-4 weeks budget):
#   - 3 seeds for the 3 headline configs (baseline, concat, depthgate-ours).
#   - 1 seed for every other ablation (still gives directional signal).
#
# Per-run output goes to work_dirs/<config>__seed<N>/. Aggregate with:
#   python tools/eval/aggregate_results.py "work_dirs/*" --out results.md
#
# Edit SEEDS / GPU / EXTRA_FLAGS to your machine.
#
# Usage:
#   bash tools/run_ablations.sh           # run everything
#   bash tools/run_ablations.sh fusion    # only fusion-mechanism group
#   bash tools/run_ablations.sh headline  # only the 3 main configs × 3 seeds

set -euo pipefail

GPU="${GPU:-0}"
SEEDS_HEADLINE=(0 1 2)
SEEDS_OTHER=(0)
EXTRA_FLAGS="--amp"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

run() {
    local cfg="$1"
    local seed="$2"
    local name
    name="$(basename "${cfg%.py}")__seed${seed}"
    local out="work_dirs/${name}"
    if [[ -f "${out}/last_checkpoint" ]]; then
        echo "[skip-existing] ${name}"
        return
    fi
    echo "[run] ${name}"
    python tools/train.py "$cfg" \
        --work-dir "$out" \
        ${EXTRA_FLAGS} \
        --cfg-options "randomness.seed=${seed}"
}

# -- Headline group: baseline / concat / depthgate, 3 seeds each --
HEADLINE=(
    configs/mask2former_r50_bamforest.py
    configs/ablations/fusion_concat.py
    configs/mask2former_r50_depthgate_bamforest.py
)

# -- Fusion mechanism group: other fusion choices, 1 seed --
FUSION=(
    configs/ablations/fusion_sum.py
    configs/ablations/fusion_film.py
    configs/ablations/fusion_se.py
)

# -- Architecture group --
ARCH=(
    configs/ablations/levels_p5_only.py
    configs/ablations/levels_p4_p5.py
    configs/ablations/levels_all.py
    configs/ablations/encoder_unfrozen.py
    configs/ablations/encoder_small.py
    configs/ablations/encoder_large.py
    configs/ablations/gate_init_rgb.py
)

# -- Robustness training-time injection --
ROBUST=(
    configs/ablations/robustness_noise.py
    configs/ablations/robustness_dropout.py
)

# -- Depth source --
DSRC=(
    configs/ablations/depth_zero.py
    # Uncomment after you generate the corresponding depth folders:
    # configs/ablations/depth_dav2_small.py
    # configs/ablations/depth_midas.py
    # configs/ablations/depth_zoe.py
)

group="${1:-all}"

case "$group" in
    headline) RUN=("${HEADLINE[@]}"); SEED_LIST=("${SEEDS_HEADLINE[@]}") ;;
    fusion)   RUN=("${FUSION[@]}");   SEED_LIST=("${SEEDS_OTHER[@]}") ;;
    arch)     RUN=("${ARCH[@]}");     SEED_LIST=("${SEEDS_OTHER[@]}") ;;
    robust)   RUN=("${ROBUST[@]}");   SEED_LIST=("${SEEDS_OTHER[@]}") ;;
    dsrc)     RUN=("${DSRC[@]}");     SEED_LIST=("${SEEDS_OTHER[@]}") ;;
    all)
        # Headlines first (3 seeds), then the rest (1 seed each)
        for cfg in "${HEADLINE[@]}"; do
            for s in "${SEEDS_HEADLINE[@]}"; do run "$cfg" "$s"; done
        done
        for cfg in "${FUSION[@]}" "${ARCH[@]}" "${ROBUST[@]}" "${DSRC[@]}"; do
            for s in "${SEEDS_OTHER[@]}"; do run "$cfg" "$s"; done
        done
        echo "[done] all"
        exit 0
        ;;
    *) echo "unknown group: $group" >&2; exit 2 ;;
esac

for cfg in "${RUN[@]}"; do
    for s in "${SEED_LIST[@]}"; do
        run "$cfg" "$s"
    done
done
echo "[done] $group"
