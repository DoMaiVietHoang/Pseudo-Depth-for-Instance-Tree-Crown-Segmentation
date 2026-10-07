"""Ablation: RGB+depth element-wise sum fusion."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(fusion_type="sum")
