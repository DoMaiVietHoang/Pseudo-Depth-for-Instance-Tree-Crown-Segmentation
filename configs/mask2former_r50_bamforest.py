"""Baseline: Mask2Former (ResNet-50) on BAMFOREST, RGB only."""

_base_ = ["./_base_.py"]

num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

model = dict(
    type="Mask2Former",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,
        mask_pad_value=0,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    panoptic_head=dict(
        type="Mask2FormerHead",
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type="MSDeformAttnPixelDecoder",
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256, num_heads=8, num_levels=3, num_points=4,
                        im2col_step=64, dropout=0.0, batch_first=True),
                    ffn_cfg=dict(embed_dims=256, feedforward_channels=1024,
                                 num_fcs=2, ffn_drop=0.0, act_cfg=dict(type="ReLU", inplace=True)),
                ),
                init_cfg=None,
            ),
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None,
        ),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0, batch_first=True),
                cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0, batch_first=True),
                ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, num_fcs=2,
                             ffn_drop=0.0, act_cfg=dict(type="ReLU", inplace=True)),
            ),
            init_cfg=None,
        ),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=2.0,
                      reduction="mean", class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=5.0),
        loss_dice=dict(type="DiceLoss", use_sigmoid=True, activate=True, reduction="mean",
                       naive_dice=True, eps=1.0, loss_weight=5.0),
    ),
    panoptic_fusion_head=dict(
        type="MaskFormerFusionHead",
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None,
    ),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="ClassificationCost", weight=2.0),
                dict(type="CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
                dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
            ],
        ),
        sampler=dict(type="MaskPseudoSampler"),
    ),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        iou_thr=0.8,
        filter_low_score=True,
    ),
    init_cfg=None,
)

# -------- Dataset paths (override the base /media/... root) --------
# Layout: <data_root>/coco1024/{train2023,val2023,depth_train,depth_val,annotations}
# Depth lives *inside* coco1024, so depth_subfolder carries the coco1024/ prefix
# (path = <data_root>/<depth_subfolder>/<stem>_depth.npy).
data_root = "/mnt/hoangdmv/Research/Instance_segmentation/dataset/Bamberg_coco1024/"

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        depth_subfolder="coco1024/depth_train",
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        depth_subfolder="coco1024/depth_val",
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + "coco1024/annotations/instances_tree_eval2023.json")
test_evaluator = val_evaluator
