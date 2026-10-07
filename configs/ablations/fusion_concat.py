"""Ablation: RGB-D concat fusion (instead of DepthGate)."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(fusion_type="concat")
