"""Depth source: ZoeDepth (metric depth) instead of DAv2 (relative).

Expects: <data_root>/depth_zoe/{train,val}/<stem>_depth.npy
"""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
train_dataloader = dict(
    dataset=dict(depth_subfolder="depth_zoe/train"),
)
val_dataloader = dict(
    dataset=dict(depth_subfolder="depth_zoe/val"),
)
test_dataloader = val_dataloader
