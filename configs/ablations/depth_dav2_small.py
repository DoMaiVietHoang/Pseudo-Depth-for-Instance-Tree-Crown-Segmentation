"""Depth source: DAv2-small (vits) instead of default (vitl/vitb)."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
train_dataloader = dict(
    dataset=dict(depth_subfolder="depth_dav2_small/train"),
)
val_dataloader = dict(
    dataset=dict(depth_subfolder="depth_dav2_small/val"),
)
test_dataloader = val_dataloader
