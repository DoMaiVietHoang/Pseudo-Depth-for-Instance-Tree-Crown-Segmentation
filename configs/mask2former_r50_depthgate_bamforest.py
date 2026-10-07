"""DepthGate: Mask2Former (ResNet-50) + DepthGate fusion on BAMFOREST."""

_base_ = ["./mask2former_r50_bamforest.py"]

model = dict(
    type="Mask2FormerDepthGate",
    depth_channels=256,
    fuse_levels=(1, 2, 3),         # P3, P4, P5 (skip P2 at stride 4)
    freeze_depth_encoder=True,
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
