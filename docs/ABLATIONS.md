# Ablation & Evaluation Guide

This guide walks through every experiment needed for a mid-tier paper
(BMVC / WACV / ACCV). One GPU, ~2-4 weeks budget. 1 seed per ablation, 3 seeds
for the 3 headline configs.

## 0. Prerequisites

```bash
# baseline already trained: work_dirs/mask2former_r50_bamforest/best_*.pth
# DepthGate already trained: work_dirs/mask2former_r50_depthgate_bamforest/best_*.pth
```

## 1. Configs available

All ablations are in `configs/ablations/` and inherit from the main DepthGate
config so they share data, schedule, and base model:

| Group | Config | What it tests |
|---|---|---|
| **Fusion** | `fusion_concat.py` | RGB-D concat (simple baseline) |
| | `fusion_sum.py` | Element-wise sum |
| | `fusion_film.py` | FiLM (Perez 2018) |
| | `fusion_se.py` | SE block (depth-driven channel attn) |
| **Architecture** | `levels_p5_only.py` | Fuse at stride 32 only |
| | `levels_p4_p5.py` | Stride 16 + 32 |
| | `levels_all.py` | All 4 FPN levels |
| | `encoder_unfrozen.py` | Train depth encoder |
| | `encoder_small.py` | Depth encoder base=16 |
| | `encoder_large.py` | Depth encoder base=64 |
| | `gate_init_rgb.py` | Init g≈1 instead of 0.5 |
| **Robustness (train-time)** | `robustness_noise.py` | Inject Gaussian noise to depth |
| | `robustness_dropout.py` | Random depth dropout (sim missing) |
| **Depth source** | `depth_zero.py` | Zero depth (sanity control) |
| | `depth_dav2_small.py` | Smaller DAv2 |
| | `depth_midas.py` | MiDaS v3.1 |
| | `depth_zoe.py` | ZoeDepth (metric) |

## 2. Train everything

```bash
# 3 seeds × {baseline, concat, DepthGate} + 1 seed × every other ablation
bash tools/run_ablations.sh           # everything
# Or one group at a time:
bash tools/run_ablations.sh headline  # 9 runs (3 × 3 seeds) — start here
bash tools/run_ablations.sh fusion
bash tools/run_ablations.sh arch
bash tools/run_ablations.sh robust
bash tools/run_ablations.sh dsrc
```

For the depth-source ablations (MiDaS / Zoe / DAv2-small), generate the
alternative depth maps first:

```bash
python tools/generate_depth.py --model midas \
    --img-dir data/bamforest/images/train --out-dir data/bamforest/depth_midas/train
python tools/generate_depth.py --model midas \
    --img-dir data/bamforest/images/val   --out-dir data/bamforest/depth_midas/val
# repeat for zoe_n / dav2_small ...
```

## 3. Evaluate everything

```bash
bash tools/eval/eval_all.sh
# Generates per-run: results.pkl + boundary_ap.txt + touching.txt
# Generates global:  results.md + results.csv
```

For mean ± std across the 3 headline seeds:
```bash
python tools/eval/aggregate_seeds.py work_dirs --out results_meanstd.md
```

## 4. Efficiency table (params, FLOPs, latency)

```bash
bash tools/eval/profile_all.sh   # writes profile.md
```

## 5. Per-size and touching-crown analysis

```bash
# Per-size: AP_S / AP_M / AP_L for any single model
python tools/eval/per_size_eval.py \
    --gt data/bamforest/annotations/val.json \
    --results work_dirs/mask2former_r50_depthgate_bamforest__seed0/results.pkl

# Touching-crown subset: re-eval on instances that touch >=1 other instance
python tools/eval/touching_crown_eval.py \
    --gt data/bamforest/annotations/val.json \
    --results work_dirs/mask2former_r50_depthgate_bamforest__seed0/results.pkl
```

## 6. Robustness at test time

Sweep noise + dropout on a TRAINED model (don't retrain):

```bash
python tools/eval/test_robustness.py \
    configs/mask2former_r50_depthgate_bamforest.py \
    work_dirs/mask2former_r50_depthgate_bamforest__seed0/best_*.pth \
    --noise 0.0 0.05 0.1 0.2 0.3 --dropout 0.0 0.2 0.5 1.0 \
    --out robustness.json

python tools/eval/plot_robustness.py robustness.json --out-dir figs/
```

## 7. Visualizations

### Gate heatmaps (interpretability)
```bash
python tools/eval/visualize_gate.py \
    configs/mask2former_r50_depthgate_bamforest.py \
    work_dirs/mask2former_r50_depthgate_bamforest__seed0/best_*.pth \
    --img data/bamforest/images/val/Stadtwald_31_1815.tif \
    --depth data/bamforest/depth/val/Stadtwald_31_1815_depth.npy \
    --out-dir gate_vis/
```

### Qualitative side-by-side
```bash
python tools/eval/compare_qualitative.py \
    --img-dir data/bamforest/images/val \
    --gt data/bamforest/annotations/val.json \
    --base-config configs/mask2former_r50_bamforest.py \
    --base-ckpt   work_dirs/mask2former_r50_bamforest__seed0/best_*.pth \
    --our-config  configs/mask2former_r50_depthgate_bamforest.py \
    --our-ckpt    work_dirs/mask2former_r50_depthgate_bamforest__seed0/best_*.pth \
    --out-dir qual_vis --max-images 30
```

## 8. Paper tables

After step 3-4, you have:

- `results_meanstd.md` — **headline table** (3 seeds × 3 main configs, mean±std)
- `results.md` — **full ablation table** (every config, 1+ seeds)
- `profile.md` — **efficiency table**
- `figs/robustness_*.png` — **robustness plots**
- `gate_vis/*.jpg` — **interpretability figure** (gate heatmaps)
- `qual_vis/*.jpg` — **qualitative figure** (GT | baseline | ours)

## 9. Recommended paper structure

| Section | Use |
|---|---|
| Table 1: main results | `results_meanstd.md` |
| Table 2: fusion ablation | filter `results.md` to `baseline / concat / sum / film / se / depthgate` |
| Table 3: arch ablation | filter to `levels_* / encoder_* / gate_init_*` |
| Table 4: depth source | filter to `depth_zero / depth_dav2_* / depth_midas / depth_zoe` |
| Table 5: efficiency | `profile.md` |
| Figure: robustness | `figs/robustness_*.png` |
| Figure: gate interpretability | `gate_vis/*.jpg` (pick 3-4) |
| Figure: qualitative | `qual_vis/*.jpg` (pick 4-6) |

## 10. Recommended run order (to de-risk early)

1. Train baseline + DepthGate + concat with 3 seeds → **9 runs, biggest claims** (~1 week)
2. Train `depth_zero` → sanity check (1 run, ~12h)
3. Train fusion ablation (`sum / film / se`) → 3 runs (~1.5 day)
4. Train arch ablation (`levels_* / encoder_*`) → 6 runs (~3 days)
5. (Optional) Depth source ablation → generate alternative depth first
6. Run all eval + viz scripts in parallel — they're cheap

If something is going wrong, step 1 + 2 reveal it before you burn 2 weeks.
