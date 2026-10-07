"""Ablation: fuse only at the coarsest level (stride 32)."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(fuse_levels=(3,))
