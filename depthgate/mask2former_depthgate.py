"""Mask2Former wrapper with pluggable RGB-depth fusion.

This subclasses MMDetection's Mask2Former and overrides `extract_feat` so
that the multi-scale RGB features can be fused with depth features at
configurable FPN levels via a configurable fusion module.

Supports several ablation knobs in a single class:
    fusion_type:        which fusion module (depthgate / concat / sum / film / se)
    fuse_levels:        which FPN levels to fuse
    depth_channels:     channel count for depth features
    depth_base_channels:size of the depth encoder (controls capacity)
    freeze_depth_encoder: freeze depth CNN
    gate_init:          'zero' (g≈0.5 init, default) or 'rgb' (g≈1, depth off)
    depth_noise_std:    Gaussian noise std added to depth input at TRAIN time
    depth_dropout_prob: probability of zeroing the entire depth input at TRAIN time
"""

from __future__ import annotations

import torch
import torch.nn as nn
from mmdet.models.detectors import Mask2Former
from mmdet.registry import MODELS

from .depth_gate import DepthEncoder, DepthGate
from .fusions import ConcatFusion, SumFusion, FiLMFusion, SEFusionDepth


_FUSION_REGISTRY = {
    "depthgate": DepthGate,
    "concat": ConcatFusion,
    "sum": SumFusion,
    "film": FiLMFusion,
    "se": SEFusionDepth,
}


@MODELS.register_module()
class Mask2FormerDepthGate(Mask2Former):
    def __init__(
        self,
        *args,
        depth_channels: int = 256,
        depth_base_channels: int = 32,
        fuse_levels: tuple[int, ...] = (1, 2, 3),
        rgb_channels: tuple[int, ...] = (256, 512, 1024, 2048),
        freeze_depth_encoder: bool = True,
        fusion_type: str = "depthgate",
        gate_init: str = "zero",
        depth_noise_std: float = 0.0,
        depth_dropout_prob: float = 0.0,
        perturb_at_eval: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.perturb_at_eval = bool(perturb_at_eval)
        if fusion_type not in _FUSION_REGISTRY:
            raise ValueError(
                f"fusion_type={fusion_type} not in {list(_FUSION_REGISTRY)}"
            )
        self.fusion_type = fusion_type
        self.fuse_levels = tuple(fuse_levels)
        self.rgb_channels = tuple(rgb_channels)
        self.depth_noise_std = float(depth_noise_std)
        self.depth_dropout_prob = float(depth_dropout_prob)

        # Depth encoder still emits `depth_channels` per scale; we re-project
        # to the RGB channel count per fused level before fusion so the fusion
        # module sees matching channels on both sides.
        self.depth_encoder = DepthEncoder(
            out_channels=depth_channels, base_channels=depth_base_channels
        )
        self.depth_proj = nn.ModuleDict({
            str(i): nn.Conv2d(depth_channels, self.rgb_channels[i], 1)
            for i in self.fuse_levels if i < len(self.rgb_channels)
        })

        fusion_cls = _FUSION_REGISTRY[fusion_type]
        self.depth_gates = nn.ModuleDict({
            str(i): fusion_cls(self.rgb_channels[i])
            for i in self.fuse_levels if i < len(self.rgb_channels)
        })

        # Optional alternative gate init (only meaningful for DepthGate).
        if fusion_type == "depthgate" and gate_init == "rgb":
            for mod in self.depth_gates.values():
                # Push final gate conv toward +large so sigmoid -> ~1 (RGB only).
                nn.init.zeros_(mod.gate[-2].weight)
                nn.init.constant_(mod.gate[-2].bias, 4.0)
        elif gate_init not in ("zero", "rgb"):
            raise ValueError(f"gate_init must be 'zero' or 'rgb', got {gate_init}")

        if freeze_depth_encoder:
            for p in self.depth_encoder.parameters():
                p.requires_grad = False
            self.depth_encoder.eval()

    # ------------------------------------------------------------------ utils

    def _collect_depth(self, batch_data_samples) -> torch.Tensor | None:
        # Per-sample depth maps come out of the pipeline at the pre-pad image
        # size (ResizeDepth matches image shape before DetDataPreprocessor
        # pads to pad_size_divisor / batch_input_shape). To stack into a batch
        # we pad each depth map (with zeros) to the common batch_input_shape.
        target_hw = None
        for ds in batch_data_samples:
            shape = ds.metainfo.get("batch_input_shape", None)
            if shape is not None:
                target_hw = tuple(shape)
                break

        depths = []
        for ds in batch_data_samples:
            d = ds.metainfo.get("depth", None)
            if d is None:
                return None
            if target_hw is not None and d.shape[-2:] != target_hw:
                # d is (1, H, W); pad right/bottom to (1, target_H, target_W).
                _, h, w = d.shape
                th, tw = target_hw
                pad_h = max(th - h, 0)
                pad_w = max(tw - w, 0)
                if pad_h or pad_w:
                    d = torch.nn.functional.pad(d, (0, pad_w, 0, pad_h), value=0.0)
                if d.shape[-2:] != target_hw:
                    d = d[..., :th, :tw]
            depths.append(d)
        depth = torch.stack(depths, dim=0).to(next(self.parameters()).device)
        return depth

    def _maybe_perturb_depth(self, depth: torch.Tensor) -> torch.Tensor:
        if not (self.training or self.perturb_at_eval):
            return depth
        if self.depth_dropout_prob > 0:
            B = depth.shape[0]
            mask = (torch.rand(B, 1, 1, 1, device=depth.device)
                    > self.depth_dropout_prob).float()
            depth = depth * mask
        if self.depth_noise_std > 0:
            depth = depth + self.depth_noise_std * torch.randn_like(depth)
            depth = depth.clamp(0.0, 1.0)
        return depth

    # -------------------------------------------------------- core override

    def extract_feat(self, batch_inputs, batch_data_samples=None):
        rgb_feats = self.backbone(batch_inputs)
        if self.with_neck:
            rgb_feats = self.neck(rgb_feats)
        rgb_feats = list(rgb_feats)

        if batch_data_samples is None:
            return tuple(rgb_feats)

        depth = self._collect_depth(batch_data_samples)
        if depth is None:
            return tuple(rgb_feats)

        depth = self._maybe_perturb_depth(depth)

        with torch.set_grad_enabled(not self._depth_frozen()):
            depth_feats = self.depth_encoder(depth)

        for lvl in self.fuse_levels:
            if lvl >= len(rgb_feats) or lvl >= len(depth_feats):
                continue
            d = self.depth_proj[str(lvl)](depth_feats[lvl])
            rgb_feats[lvl] = self.depth_gates[str(lvl)](
                rgb_feats[lvl], d
            )

        return tuple(rgb_feats)

    def _depth_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.depth_encoder.parameters())

    # --------- MMDet 3.x routing: pass batch_data_samples to extract_feat ---

    def loss(self, batch_inputs, batch_data_samples):
        x = self.extract_feat(batch_inputs, batch_data_samples)
        losses = self.panoptic_head.loss(x, batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        feats = self.extract_feat(batch_inputs, batch_data_samples)
        mask_cls_results, mask_pred_results = self.panoptic_head.predict(
            feats, batch_data_samples
        )
        results_list = self.panoptic_fusion_head.predict(
            mask_cls_results, mask_pred_results, batch_data_samples,
            rescale=rescale,
        )
        results = self.add_pred_to_datasample(batch_data_samples, results_list)
        return results
