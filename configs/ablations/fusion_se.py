"""Ablation: Squeeze-and-Excitation (depth-driven) channel attention fusion."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(fusion_type="se")
