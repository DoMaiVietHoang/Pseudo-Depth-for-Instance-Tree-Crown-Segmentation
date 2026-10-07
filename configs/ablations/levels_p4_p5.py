"""Ablation: fuse at the two coarsest levels (stride 16, 32)."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(fuse_levels=(2, 3))
