"""DepthGate: depth-gated feature enhancement for instance segmentation.

Wires a frozen depth encoder into a standard RGB backbone via per-pixel
gating and depth-attention. Applied at multiple FPN scales.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS


class DepthEncoder(nn.Module):
    """Small CNN that maps a single-channel depth map to multi-scale features.

    Output strides match a standard FPN: {4, 8, 16, 32}. Each scale produces
    `out_channels` features so the gate module can fuse them with the RGB
    backbone features at the matching FPN level.
    """

    def __init__(self, out_channels: int = 256, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c),
            nn.GELU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c),
            nn.GELU(),
        )  # stride 4
        self.stage2 = self._block(c, c * 2)         # stride 8
        self.stage3 = self._block(c * 2, c * 4)     # stride 16
        self.stage4 = self._block(c * 4, c * 8)     # stride 32

        self.proj = nn.ModuleList([
            nn.Conv2d(c,     out_channels, 1),
            nn.Conv2d(c * 2, out_channels, 1),
            nn.Conv2d(c * 4, out_channels, 1),
            nn.Conv2d(c * 8, out_channels, 1),
        ])

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
        )

    def forward(self, depth: torch.Tensor) -> list[torch.Tensor]:
        x1 = self.stem(depth)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        feats = [x1, x2, x3, x4]
        return [proj(f) for proj, f in zip(self.proj, feats)]


@MODELS.register_module()
class DepthGate(nn.Module):
    """Gated fusion of an RGB feature map and a depth feature map.

    For each spatial location, learns:
      g  in [0,1]: how much to trust RGB vs. depth
      a  in [0,1]: spatial attention over depth (highlights ridges/edges)

    Output: g * F_rgb + (1 - g) * (a * F_depth)
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 16)

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

        self.depth_attn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

        # Initialize the final gate conv near zero so g ~ 0.5 at start
        # (balanced trust); training learns to push g toward 1 where depth
        # is unreliable and toward 0 where depth helps.
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)

        self._last_gate_mean:  float | None = None
        self._last_alpha_mean: float | None = None

    def forward(self, f_rgb: torch.Tensor, f_depth: torch.Tensor) -> torch.Tensor:
        if f_depth.shape[-2:] != f_rgb.shape[-2:]:
            f_depth = F.interpolate(
                f_depth, size=f_rgb.shape[-2:], mode="bilinear", align_corners=False
            )
        g = self.gate(torch.cat([f_rgb, f_depth], dim=1))
        a = self.depth_attn(f_depth)
        out = g * f_rgb + (1.0 - g) * (a * f_depth)

        if self.training:
            with torch.no_grad():
                self._last_gate_mean  = g.mean().item()
                self._last_alpha_mean = a.mean().item()
        return out

    @property
    def last_gate_mean(self) -> float | None:
        return self._last_gate_mean

    @property
    def last_alpha_mean(self) -> float | None:
        return self._last_alpha_mean
