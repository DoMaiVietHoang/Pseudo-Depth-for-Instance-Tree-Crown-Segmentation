"""Shared base settings for BAMFOREST experiments (single class: tree).

Tuned for a single 16 GB GPU (RTX 4080). Tile size 768x768, batch 2.
"""

# -------- Dataset --------
dataset_type = "DepthCocoDataset"
data_root = "/media/toanhm5/Workspace/Source_Code/hoangdmv/dataset/"
classes = ("tree",)

backend_args = None

# RGB normalization (ImageNet defaults; adjust if BAMFOREST stats differ)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)

image_size = (768, 768)

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
    dict(type="RandomCrop", crop_size=image_size, recompute_bbox=True, allow_negative_crop=True),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1.0, 1.0), keep_empty=False),
    dict(type="ResizeDepth"),
    # Convert depth to tensor BEFORE PackDetInputs so it can be picked up via
    # meta_keys=('depth', ...). Otherwise PackDetInputs builds a fresh dict and
    # drops the 'depth' key, leaving the model with no depth input.
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

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="coco1024/annotations/instances_tree_train2023.json",
        data_prefix=dict(img="coco1024/train2023"),
        depth_subfolder="depth_train",
        depth_ext=".npy",
        depth_suffix="_depth",
        filter_cfg=dict(filter_empty_gt=True, min_size=8),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes),
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="coco1024/annotations/instances_tree_eval2023.json",
        data_prefix=dict(img="coco1024/val2023"),
        depth_subfolder="depth_val",
        depth_ext=".npy",
        depth_suffix="_depth",
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "coco1024/annotations/instances_tree_eval2023.json",
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

# -------- Training schedule --------
max_epochs = 100
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=2)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
            "query_embed": dict(lr_mult=1.0, decay_mult=0.0),
            "query_feat": dict(lr_mult=1.0, decay_mult=0.0),
            "level_embed": dict(lr_mult=1.0, decay_mult=0.0),
        },
        norm_decay_mult=0.0,
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0, end=max_epochs,
        by_epoch=True,
        milestones=[int(max_epochs * 0.8), int(max_epochs * 0.95)],
        gamma=0.1,
    ),
]

default_scope = "mmdet"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=2, max_keep_ckpts=3, save_best="coco/segm_mAP"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

custom_hooks = [
    dict(type="LogDepthGateHook", interval=20),
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(
    type="LogProcessor", window_size=50, by_epoch=True,
    custom_cfg=[
        dict(data_src="gate/L1", method_name="mean", window_size=20),
        dict(data_src="gate/L2", method_name="mean", window_size=20),
        dict(data_src="gate/L3", method_name="mean", window_size=20),
        dict(data_src="gate/mean", method_name="mean", window_size=20),
        dict(data_src="alpha/L1", method_name="mean", window_size=20),
        dict(data_src="alpha/L2", method_name="mean", window_size=20),
        dict(data_src="alpha/L3", method_name="mean", window_size=20),
        dict(data_src="alpha/mean", method_name="mean", window_size=20),
    ],
)
log_level = "INFO"
load_from = None
resume = False

custom_imports = dict(imports=["depthgate"], allow_failed_imports=False)
