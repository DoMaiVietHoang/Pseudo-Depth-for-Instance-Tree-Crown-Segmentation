"""Ablation: fuse at all 4 FPN levels (include stride 4 — DAv2 noise risk)."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(fuse_levels=(0, 1, 2, 3))
