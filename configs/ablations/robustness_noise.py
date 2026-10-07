"""Robustness: train with Gaussian depth noise (std=0.1) injected.

Tests whether DepthGate can degrade gracefully to noisy depth."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(depth_noise_std=0.1)
