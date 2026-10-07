"""Ablation: naive RGBD early fusion (depth as a 4th input channel).

Depth is concatenated onto the RGB tensor and fed to a ResNet stem inflated
from 3 to 4 channels, rather than encoded separately and fused at FPN levels.
The data_preprocessor mean/std stay 3-channel: normalisation happens before
the model, and the depth channel is concatenated inside extract_feat (depth
maps are already scaled to [0, 1] by the pipeline).
"""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]

model = dict(type="Mask2FormerRGBD")

# No gate to log -- the DepthGate hook has nothing to read on this model.
custom_hooks = []
