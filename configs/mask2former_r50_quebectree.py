"""Baseline: Mask2Former (ResNet-50) on QuebecTree, RGB only (no DepthGate).

Uses the COCO json produced by tools/quebectree_to_coco.py. No depth maps are
needed: this overrides the depth dataset/pipeline/hooks that the BAMFOREST base
config pulls in, falling back to a plain CocoDataset + standard pipeline.

    python tools/train.py configs/mask2former_r50_quebectree.py
"""

_base_ = ["./mask2former_r50_bamforest.py"]

# QuebecTree tiles are 1024x1024; train at full 1024 resolution.
dataset_type = "CocoDataset"
data_root = "/mnt/hoangdmv/Research/Instance_segmentation/dataset/QuebecTree/"
classes = ("tree",)
image_size = (1024, 1024)

backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
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
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=image_size, keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                   "scale_factor"),
    ),
]

train_dataloader = dict(
    dataset=dict(
        # _delete_ replaces the inherited DepthCocoDataset dict wholesale so its
        # depth_subfolder/depth_ext/depth_suffix kwargs don't leak into CocoDataset.
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_train.json",
        data_prefix=dict(img="images/train"),
        filter_cfg=dict(filter_empty_gt=True, min_size=8),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes),
    ),
)

val_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_val.json",
        data_prefix=dict(img="images/val"),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + "annotations/instances_val.json")
test_evaluator = val_evaluator

# No DepthGate -> no gate logging hook, and no need to import the depthgate pkg.
custom_hooks = []
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
