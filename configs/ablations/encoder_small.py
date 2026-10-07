"""Ablation: smaller depth encoder (base_channels=16, ~4x fewer params)."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(depth_base_channels=16)
