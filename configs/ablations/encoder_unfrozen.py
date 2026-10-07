"""Ablation: train the depth encoder jointly (no freezing)."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(freeze_depth_encoder=False)
