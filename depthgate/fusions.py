"""Baseline fusion modules for ablation comparison against DepthGate.

All modules take (f_rgb, f_depth) and return a fused feature of the same
shape as f_rgb. Same channel count is assumed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


def _align(f_rgb: torch.Tensor, f_depth: torch.Tensor) -> torch.Tensor:
    if f_depth.shape[-2:] != f_rgb.shape[-2:]:
        f_depth = F.interpolate(
            f_depth, size=f_rgb.shape[-2:], mode="bilinear", align_corners=False
        )
    return f_depth


@MODELS.register_module()
class ConcatFusion(nn.Module):
    """Concatenate then 1x1 conv back to `channels`."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels * 2, channels, 1)
        self._last_gate_mean = None  # for API compatibility with logger hook

    def forward(self, f_rgb, f_depth):
        f_depth = _align(f_rgb, f_depth)
        return self.proj(torch.cat([f_rgb, f_depth], dim=1))

    @property
    def last_gate_mean(self):
        return self._last_gate_mean


@MODELS.register_module()
class SumFusion(nn.Module):
    """Element-wise sum (depth pre-projected)."""

    def __init__(self, channels: int):
        super().__init__()
        self.depth_proj = nn.Conv2d(channels, channels, 1)
        self._last_gate_mean = None

    def forward(self, f_rgb, f_depth):
        f_depth = _align(f_rgb, f_depth)
        return f_rgb + self.depth_proj(f_depth)

    @property
    def last_gate_mean(self):
        return self._last_gate_mean


@MODELS.register_module()
class FiLMFusion(nn.Module):
    """FiLM (Perez et al. 2018): modulate RGB with depth-conditioned (gamma, beta).

    F_out = gamma(F_depth) * F_rgb + beta(F_depth)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.gamma = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.beta = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        # init gamma final near 1, beta near 0 (identity at start)
        nn.init.zeros_(self.gamma[-1].weight)
        nn.init.ones_(self.gamma[-1].bias)
        nn.init.zeros_(self.beta[-1].weight)
        nn.init.zeros_(self.beta[-1].bias)
        self._last_gate_mean = None

    def forward(self, f_rgb, f_depth):
        f_depth = _align(f_rgb, f_depth)
        return self.gamma(f_depth) * f_rgb + self.beta(f_depth)

    @property
    def last_gate_mean(self):
        return self._last_gate_mean


@MODELS.register_module()
class SEFusionDepth(nn.Module):
    """Channel-wise SE block driven by depth global stats, applied to RGB.

    F_out = SE(GAP(F_depth)) * F_rgb + F_depth_proj
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.depth_proj = nn.Conv2d(channels, channels, 1)
        self._last_gate_mean = None

    def forward(self, f_rgb, f_depth):
        f_depth = _align(f_rgb, f_depth)
        w = self.se(f_depth)
        return w * f_rgb + self.depth_proj(f_depth)

    @property
    def last_gate_mean(self):
        return self._last_gate_mean
