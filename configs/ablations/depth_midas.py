"""Depth source: MiDaS v3.1 instead of DAv2.

Expects depth files at: <data_root>/depth_midas/{train,val}/<stem>_depth.npy
Generate them with `python tools/generate_depth.py --model midas ...`
"""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
train_dataloader = dict(
    dataset=dict(depth_subfolder="depth_midas/train"),
)
val_dataloader = dict(
    dataset=dict(depth_subfolder="depth_midas/val"),
)
test_dataloader = val_dataloader
