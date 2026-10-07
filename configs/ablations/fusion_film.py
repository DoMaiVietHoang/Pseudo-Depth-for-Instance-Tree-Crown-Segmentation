"""Ablation: FiLM fusion (Perez et al. 2018)."""
_base_ = ["../mask2former_r50_depthgate_bamforest.py"]
model = dict(fusion_type="film")
