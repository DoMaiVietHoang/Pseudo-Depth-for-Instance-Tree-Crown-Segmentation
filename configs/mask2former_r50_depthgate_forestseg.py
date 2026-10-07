"""DepthGate: Mask2Former (ResNet-50) + DepthGate fusion on ForestSeg-T1.

Dataset converted from LabelMe via tools/forestseg_to_coco.py. Depth maps
generated via tools/generate_depth.py into <Split>/depth/<stem>_depth.npy.

Layout (data_root below):
    Train/images/<stem>.jpg          Val/images/<stem>.jpg
    Train/depth/<stem>_depth.npy     Val/depth/<stem>_depth.npy
    annotations/instances_train.json annotations/instances_val.json

Trains at the native tile resolution 1024x1024 (the base config uses 768).
Because the base builds its pipelines with the base `image_size` at parse time,
overriding the size here requires redefining the pipelines too (below).
"""

_base_ = ["./mask2former_r50_depthgate_bamforest.py"]

# ForestSeg-T1 tiles are 1024x1024 — train at full resolution.
image_size = (1024, 1024)
backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="LoadDepthFromFile"),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomResize",
        scale=image_size,
        ratio_range=(0.8, 1.5),
        resize_type="Resize",
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=image_size, recompute_bbox=True,
         allow_negative_crop=True),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1.0, 1.0), keep_empty=False),
    dict(type="ResizeDepth"),
    # Depth -> tensor BEFORE PackDetInputs so it survives via meta_keys=('depth',).
    dict(type="PackDepthInputs"),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                   "scale_factor", "flip", "flip_direction", "depth"),
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=image_size, keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="LoadDepthFromFile"),
    dict(type="ResizeDepth"),
    dict(type="PackDepthInputs"),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                   "scale_factor", "depth"),
    ),
]

# data_root + paths + the 1024 pipeline together.
data_root = "/mnt/hoangdmv/Research/Instance_segmentation/dataset/ForestSeg-T1/"

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file="annotations/instances_train.json",
        data_prefix=dict(img="Train/images"),
        depth_subfolder="Train/depth",
        depth_ext=".npy",
        depth_suffix="_depth",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file="annotations/instances_val.json",
        data_prefix=dict(img="Val/images"),
        depth_subfolder="Val/depth",
        depth_ext=".npy",
        depth_suffix="_depth",
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + "annotations/instances_val.json")
test_evaluator = val_evaluator
