"""DepthGate: Mask2Former (ResNet-50) + DepthGate fusion on QuebecTree.

Dataset converted from YOLO-seg via tools/quebectree_to_coco.py. Depth maps
generated via tools/generate_depth.py into depth_{train,val}/<stem>_depth.npy.

Layout (data_root below):
    images/{train,val}/<stem>.jpg
    annotations/instances_{train,val}.json
    depth_{train,val}/<stem>_depth.npy
"""

_base_ = ["./mask2former_r50_depthgate_bamforest.py"]

# QuebecTree tiles are 1024x1024 — train at full resolution. The base builds its
# pipelines with the base `image_size` (768) at parse time, so overriding the
# size here requires redefining the pipelines too (below).
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

data_root = "/mnt/hoangdmv/Research/Instance_segmentation/dataset/QuebecTree/"

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file="annotations/instances_train.json",
        data_prefix=dict(img="images/train"),
        depth_subfolder="depth_train",
        depth_ext=".npy",
        depth_suffix="_depth",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file="annotations/instances_val.json",
        data_prefix=dict(img="images/val"),
        depth_subfolder="depth_val",
        depth_ext=".npy",
        depth_suffix="_depth",
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + "annotations/instances_val.json")
test_evaluator = val_evaluator
