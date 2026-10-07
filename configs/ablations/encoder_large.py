"""Ablation: larger depth encoder (base_channels=64)."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(depth_base_channels=64)
