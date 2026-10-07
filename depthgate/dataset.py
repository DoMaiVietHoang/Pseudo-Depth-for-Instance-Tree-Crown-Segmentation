"""COCO dataset + transforms that load a paired DAv2 depth map (.npy/.npz).

Conventions:
    - RGB image:   <data_root>/<img_subfolder>/<file_name>
    - Depth array: <data_root>/<depth_subfolder>/<stem><depth_suffix>.npy
                   (or .npz, key 'depth')
    - Depth values: any float range. Normalized per-tile to [0, 1] at load.

Error-handling strategy:
    1. Init-time: skip samples whose image OR depth file does not exist on disk.
    2. Runtime:   wrap the pipeline in try/except so corrupted files / bad
                  arrays do not crash training — mmengine re-samples a new index.
    3. All transforms validate their inputs and raise a descriptive error if
       anything is off, so the wrapper above can log + skip cleanly.
"""

from __future__ import annotations

import os
import numpy as np

from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS, TRANSFORMS
from mmcv.transforms import BaseTransform


# Rate-limited logger so a few thousand bad samples don't flood stdout.
_LOG_LIMIT = 50
_log_count = {"skipped_runtime": 0}


def _log_skip(msg: str) -> None:
    n = _log_count["skipped_runtime"]
    if n < _LOG_LIMIT:
        print(msg)
    elif n == _LOG_LIMIT:
        print(f"[DepthCocoDataset] ... (further skip messages suppressed)")
    _log_count["skipped_runtime"] = n + 1


@DATASETS.register_module()
class DepthCocoDataset(CocoDataset):
    """COCO dataset that also records the path to a depth file per image."""

    METAINFO = {
        "classes": ("tree",),
        "palette": [(0, 200, 0)],
    }

    def __init__(self, depth_subfolder: str = "depth", depth_ext: str = ".npy",
                 depth_suffix: str = "", **kwargs):
        self.depth_subfolder = depth_subfolder
        self.depth_ext = depth_ext
        self.depth_suffix = depth_suffix
        super().__init__(**kwargs)

    def parse_data_info(self, raw_data_info):
        data_info = super().parse_data_info(raw_data_info)
        if data_info is None:
            return None

        img_path = data_info.get("img_path")
        if not img_path:
            return None

        stem = os.path.splitext(os.path.basename(img_path))[0]
        depth_path = os.path.join(
            self.data_root, self.depth_subfolder,
            stem + self.depth_suffix + self.depth_ext,
        )
        # Init-time existence check (cheap). Decode errors handled at runtime.
        if not os.path.isfile(img_path) or not os.path.isfile(depth_path):
            return None
        data_info["depth_path"] = depth_path
        return data_info

    def load_data_list(self):
        data_list = super().load_data_list()
        kept = [d for d in data_list if d is not None]
        dropped = len(data_list) - len(kept)
        if dropped:
            print(f"[DepthCocoDataset] Skipped {dropped} samples with missing "
                  f"image or depth files at init time. Kept {len(kept)}.")
        if not kept:
            raise RuntimeError(
                "[DepthCocoDataset] No valid samples remain after filtering. "
                "Check data_root / depth_subfolder / depth_suffix in config."
            )
        return kept

    def prepare_data(self, idx: int):
        # Any exception in the pipeline (corrupted image, unreadable depth,
        # shape mismatch, NaN, etc.) becomes a None return so mmengine
        # transparently retries with another sample.
        try:
            return super().prepare_data(idx)
        except Exception as e:
            try:
                info = self.get_data_info(idx)
                path = info.get("img_path", "<unknown>")
            except Exception:
                path = "<unknown>"
            _log_skip(f"[DepthCocoDataset] Skip idx={idx} ({path}) "
                      f"— {type(e).__name__}: {e}")
            return None


@TRANSFORMS.register_module()
class LoadDepthFromFile(BaseTransform):
    """Load a depth array from .npy / .npz and attach it as `results['depth']`.

    Depth is stored as float32 with shape (H, W) and normalized per-tile to
    [0, 1] using the 2nd/98th percentile to suppress outliers. NaN/Inf are
    treated as missing data (replaced with the percentile-low value before
    normalization).
    """

    def __init__(self, key: str = "depth", to_float32: bool = True,
                 percentile_clip=(2.0, 98.0)):
        self.key = key
        self.to_float32 = to_float32
        self.percentile_clip = percentile_clip

    def _load_array(self, path: str) -> np.ndarray:
        if path.endswith(".npz"):
            with np.load(path) as data:
                if self.key in data.files:
                    return data[self.key]
                if "depth" in data.files:
                    return data["depth"]
                if len(data.files) == 1:
                    return data[data.files[0]]
                raise KeyError(
                    f"No usable key in {path}; available: {data.files}"
                )
        return np.load(path)

    def transform(self, results: dict) -> dict:
        path = results.get("depth_path")
        if not path:
            raise KeyError("LoadDepthFromFile: 'depth_path' missing in results")

        arr = self._load_array(path)

        # Squeeze (1, H, W) / (H, W, 1) → (H, W).
        if arr.ndim == 3:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(
                f"Depth at {path} has shape {arr.shape}, expected 2D."
            )
        if arr.size == 0:
            raise ValueError(f"Depth at {path} is empty.")

        if self.to_float32:
            arr = arr.astype(np.float32, copy=False)

        # Replace NaN/Inf so np.percentile doesn't poison the stats.
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        lo, hi = np.percentile(arr, self.percentile_clip)
        if hi - lo > 1e-6:
            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        results["depth"] = arr
        results["depth_shape"] = arr.shape
        return results


@TRANSFORMS.register_module()
class ResizeDepth(BaseTransform):
    """Resize the depth map to match the current image size.

    Robust to: missing depth, missing image, already-correct size, and odd
    shapes. Uses bilinear interpolation.
    """

    def transform(self, results: dict) -> dict:
        import cv2
        depth = results.get("depth")
        img = results.get("img")
        if depth is None or img is None:
            return results
        h, w = img.shape[:2]
        if depth.shape[:2] != (h, w):
            # cv2.resize expects (w, h)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            depth = depth.astype(np.float32, copy=False)
        results["depth"] = depth
        results["depth_shape"] = depth.shape
        return results


@TRANSFORMS.register_module()
class LoadZeroDepth(BaseTransform):
    """Drop-in replacement for LoadDepthFromFile that returns an all-zeros
    depth array. Used by the depth_zero ablation (control: model gets the
    extra parameters but no real signal)."""

    def transform(self, results: dict) -> dict:
        img = results.get("img")
        if img is None:
            # Defer until image is available; use a tiny placeholder.
            results["depth"] = np.zeros((1, 1), dtype=np.float32)
            results["depth_shape"] = (1, 1)
            return results
        h, w = img.shape[:2]
        arr = np.zeros((h, w), dtype=np.float32)
        results["depth"] = arr
        results["depth_shape"] = arr.shape
        return results


@TRANSFORMS.register_module()
class PackDepthInputs(BaseTransform):
    """Convert the depth array to a (1, H, W) float32 tensor and attach it to
    ``data_samples.metainfo['depth']`` so the model can consume it.

    Works in either pipeline position:
        - BEFORE PackDetInputs: converts ``results['depth']`` to tensor in
          place. ``PackDetInputs`` must include ``'depth'`` in its
          ``meta_keys`` so the tensor is copied into ``data_samples.metainfo``.
        - AFTER PackDetInputs: also attaches the tensor directly onto
          ``results['data_samples']`` (back-compat path; relies on
          ``results['depth']`` still being present, which it usually isn't
          since ``PackDetInputs`` builds a fresh dict — prefer the BEFORE
          position).
    """

    def transform(self, results: dict) -> dict:
        import torch
        depth = results.get("depth")
        if depth is None:
            return results

        if isinstance(depth, np.ndarray):
            depth_t = torch.from_numpy(np.ascontiguousarray(depth)).float()
        elif torch.is_tensor(depth):
            depth_t = depth.float()
        else:
            raise TypeError(
                f"PackDepthInputs: unsupported depth type {type(depth)}"
            )

        if depth_t.ndim == 2:
            depth_t = depth_t.unsqueeze(0)
        elif depth_t.ndim == 3 and depth_t.shape[0] != 1:
            depth_t = depth_t[:1]
        elif depth_t.ndim != 3:
            raise ValueError(
                f"PackDepthInputs: depth tensor has shape {tuple(depth_t.shape)}, "
                f"expected (H, W) or (1, H, W)."
            )

        results["depth"] = depth_t
        data_samples = results.get("data_samples")
        if data_samples is not None:
            data_samples.set_metainfo({"depth": depth_t})
        return results
