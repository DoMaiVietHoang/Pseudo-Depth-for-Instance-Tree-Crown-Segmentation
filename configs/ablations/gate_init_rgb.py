"""Ablation: init gate close to g=1 (RGB-only at start). Tests whether
the depth-balanced init g≈0.5 matters."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(gate_init="rgb")
