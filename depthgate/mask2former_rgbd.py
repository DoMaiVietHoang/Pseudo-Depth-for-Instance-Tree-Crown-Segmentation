"""Mask2Former with early-fusion RGBD input (depth as a 4th input channel).

This is the naive early-fusion baseline that DepthGate is measured against:
instead of encoding depth separately and fusing at FPN levels, depth is simply
stacked onto the RGB image and fed to a ResNet stem widened to 4 channels.

ResNet-50's pretrained `conv1` is 3-channel, so it is inflated to 4: the RGB
weights are copied across and the depth channel is zero-initialised, which
leaves the network's initial function identical to the RGB baseline and lets
the depth channel learn its contribution from there.

Depth arrives the same way as in Mask2FormerDepthGate -- via `batch_data_samples`
metainfo -- so `_collect_depth` (and its pad-to-batch_input_shape handling) is
inherited rather than reimplemented.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mmdet.registry import MODELS

from .mask2former_depthgate import Mask2FormerDepthGate


@MODELS.register_module()
class Mask2FormerRGBD(Mask2FormerDepthGate):
    def __init__(self, *args, **kwargs):
        # The parent builds a depth encoder / projections / fusion modules that
        # early fusion does not use. fuse_levels=() skips the projections and
        # fusion modules; the encoder is built regardless, so drop all three
        # below to keep unused parameters out of the param/FLOP counts.
        kwargs["fuse_levels"] = ()
        super().__init__(*args, **kwargs)

        del self.depth_encoder
        del self.depth_proj
        del self.depth_gates

        self._inflate_stem()

    def _inflate_stem(self) -> None:
        """Replace backbone.conv1 (3ch) with a 4ch conv, preserving pretraining."""
        old = self.backbone.conv1
        if old.in_channels == 4:
            return
        new = nn.Conv2d(
            4, old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=old.bias is not None,
        )
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = 0.0
            if old.bias is not None:
                new.bias.copy_(old.bias)
        self.backbone.conv1 = new

    def extract_feat(self, batch_inputs, batch_data_samples=None):
        if batch_data_samples is not None:
            depth = self._collect_depth(batch_data_samples)
            if depth is not None:
                depth = self._maybe_perturb_depth(depth)
                depth = depth.to(batch_inputs.dtype)
                if depth.shape[-2:] != batch_inputs.shape[-2:]:
                    depth = nn.functional.interpolate(
                        depth, size=batch_inputs.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                batch_inputs = torch.cat([batch_inputs, depth], dim=1)

        if batch_inputs.shape[1] == 3:
            # No depth available: feed a zero channel, which the zero-init
            # stem treats exactly as the RGB baseline would.
            zeros = batch_inputs.new_zeros(
                batch_inputs.shape[0], 1, *batch_inputs.shape[-2:])
            batch_inputs = torch.cat([batch_inputs, zeros], dim=1)

        feats = self.backbone(batch_inputs)
        if self.with_neck:
            feats = self.neck(feats)
        return tuple(feats)
