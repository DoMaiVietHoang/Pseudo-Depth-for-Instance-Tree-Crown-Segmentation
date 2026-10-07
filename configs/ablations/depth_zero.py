"""Control: zero-depth (architecture identical, but depth signal removed).

Replaces LoadDepthFromFile with LoadZeroDepth in train+test pipelines.
If DepthGate-with-real-depth matches this, the gain isn't from the depth
signal — it's just extra parameters. Critical sanity check."""

_base_ = ["../mask2former_r50_depthgate_bamforest.py"]

# Rebuild the pipelines with LoadZeroDepth in place of LoadDepthFromFile.
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="LoadZeroDepth"),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomResize", scale=(768, 768), ratio_range=(0.8, 1.5),
         resize_type="Resize", keep_ratio=True),
    dict(type="RandomCrop", crop_size=(768, 768), recompute_bbox=True,
         allow_negative_crop=True),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1.0, 1.0), keep_empty=False),
    dict(type="ResizeDepth"),
    dict(type="PackDepthInputs"),
    dict(type="PackDetInputs",
         meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                    "scale_factor", "flip", "flip_direction", "depth")),
]
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(768, 768), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="LoadZeroDepth"),
    dict(type="ResizeDepth"),
    dict(type="PackDepthInputs"),
    dict(type="PackDetInputs",
         meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                    "scale_factor", "depth")),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
