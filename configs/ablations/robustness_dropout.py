"""Robustness: randomly zero the depth input with prob=0.2 during training.

Simulates missing-depth at inference. The gate should learn to fall back
to RGB when depth is absent (g → 1)."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(depth_dropout_prob=0.2)
