# -*- coding: utf-8 -*-
"""Helpers shared with the BiLoRA allocation host (bilora_d2): a loader adapter and the
frozen scoring backbone used to compute brain-alignment scores for BiLoRA's ViT-B/16.
"""


class _StripIdxLoader:
    """BiLoRA's DummyDataset yields (idx, x, y); the brain-score calculator expects (x, y)."""

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            if len(batch) == 3:
                _, x, y = batch
            else:
                x, y = batch
            yield x, y

    def __len__(self):
        return len(self._loader)


def _scoring_backbone(args, device):
    """Build the scoring backbone: the same 12-block ViT-B/16 and the same weights as the
    BiLoRA backbone, so layer indices map one-to-one and scores transfer to the base model
    without touching its training."""
    import os

    from baselines.bilora_adapter.timm_compat import DEFAULT_AUGREG_NPZ
    from model_m.common.timm_backbone import ModifiedTimmViT

    weights = getattr(args, "timm_weights", None) or None
    fmt = getattr(args, "timm_weights_format", "augreg_npz")
    # Same rule as timm_compat uses for the base backbone: without an explicit file, use the
    # local AugReg npz. The scoring and base backbones must share weights for the layer
    # mapping to hold, and an implicit hub download would fail when HF_HUB_OFFLINE=1.
    if weights is None and fmt == "augreg_npz" and os.path.exists(DEFAULT_AUGREG_NPZ):
        weights = DEFAULT_AUGREG_NPZ
        print(f"[gate] --timm_weights not given; scoring backbone uses local {DEFAULT_AUGREG_NPZ} (same weights as the base).")
    return ModifiedTimmViT(
        model_name=getattr(args, "timm_model", "vit_base_patch16_224.augreg_in21k"),
        weights_file=weights, pretrained=True, weights_format=fmt,
    ).to(device)
