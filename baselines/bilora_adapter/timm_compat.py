# -*- coding: utf-8 -*-
"""timm 1.x compatibility shim: lets baselines/BiLoRA (written for timm 0.6) build its model
under timm 1.x.

Blocking points in models/fft.py::_create_vision_transformer (old API):
  1) `resolve_pretrained_cfg(v)` returns a `PretrainedCfg` dataclass in timm 1.x, which is not
     subscriptable -> `pretrained_cfg['num_classes']` raises TypeError;
  2) the `pretrained_custom_load=...` argument of `build_model_with_cfg` no longer exists in
     timm 1.x;
  3) less obvious: `models/vit_base.py` re-registers many timm model names with @register_model
     (importing it prints "Overwriting vit_base_patch16_224 in registry"). After importing
     BiLoRA, even `timm.create_model("vit_base_patch16_224.augreg_in21k")` is routed into
     BiLoRA's old `_create_vision_transformer` and fails again, so timm.create_model cannot be
     used to fetch the source weights.

The upstream BiLoRA checkout is used unmodified, so this module replaces
`models.fft._create_vision_transformer` wholesale: it builds ViT_lora_fft (random init) and then
loads pretrained weights from
  - a local augreg .npz (default ./pretrained/vit_b16_augreg_in21k.npz) via BiLoRA's own
    `models.vit_base._load_weights` (matches its ViT structure and works offline);
  - a local SSL .pth (iBOT/DINO) via model_m.common.timm_backbone._remap_ssl_state_dict,
    loaded by name;
  - otherwise, a state_dict downloaded from the HF hub (bypassing the hijacked model registry).

Real-backbone check: the core weights of all 12 blocks are compared before and after loading,
and any unchanged one raises. Without this check a silently random-initialised backbone would
turn every BiLoRA result into noise.
"""
import os
import warnings

import torch

# Per-block sub-keys that the pretrained weights must overwrite (any unchanged one = backbone not really loaded)
_REQUIRED_BLOCK_SUFFIXES = (
    "norm1.weight", "attn.qkv.weight", "attn.qkv.bias",
    "attn.proj.weight", "norm2.weight", "mlp.fc1.weight", "mlp.fc2.weight",
)
_REQUIRED_GLOBAL_KEYS = ("cls_token", "pos_embed", "patch_embed.proj.weight", "norm.weight")

# Old BiLoRA cfg names -> current HF hub names (only used by the hub fallback)
_HUB_MAP = {
    "vit_base_patch16_224_in21k": "timm/vit_base_patch16_224.augreg_in21k",
    "vit_base_patch16_224": "timm/vit_base_patch16_224.augreg2_in21k_ft_in1k",
}

DEFAULT_AUGREG_NPZ = "./pretrained/vit_b16_augreg_in21k.npz"

_PATCHED = False

# timm registry dicts to restore after BiLoRA overwrites them (see blocking point 3 in the module docstring)
_REGISTRY_DICTS = ("_model_entrypoints", "_model_to_module", "_model_default_cfgs",
                   "_model_pretrained_cfgs", "_model_has_pretrained")


def snapshot_timm_registry():
    """Snapshot the timm model registry *before* importing BiLoRA.

    BiLoRA's models/vit_base.py re-registers vit_base_patch16_224 and other names with
    @register_model; after that import, `timm.create_model("vit_base_patch16_224.augreg_in21k")`
    in this process is routed into BiLoRA's old factory and fails on PretrainedCfg, which would
    also break our own ModifiedTimmViT (the BioScore scoring backbone). Hence: snapshot before
    the import, restore after it."""
    import timm  # noqa: F401  make sure timm has populated its registry
    from timm.models import _registry

    return {name: dict(getattr(_registry, name)) if isinstance(getattr(_registry, name), dict)
            else set(getattr(_registry, name))
            for name in _REGISTRY_DICTS if hasattr(_registry, name)}


def restore_timm_registry(snap):
    """Restore the entries overwritten by BiLoRA to the timm originals (only same-name keys are
    overwritten; names that only BiLoRA defines are kept)."""
    from timm.models import _registry

    n = 0
    for name, saved in snap.items():
        cur = getattr(_registry, name)
        if isinstance(cur, dict):
            for k, v in saved.items():
                if cur.get(k) is not v:
                    cur[k] = v
                    n += 1
        else:
            cur |= saved
    print(f"[timm_compat] restored {n} timm model-registry entries overwritten by BiLoRA"
          f" (keeps our own timm.create_model working).")


def _required_keys(model):
    keys = list(_REQUIRED_GLOBAL_KEYS)
    for i in range(len(model.blocks)):
        keys += [f"blocks.{i}.{s}" for s in _REQUIRED_BLOCK_SUFFIXES]
    return [k for k in keys if k in dict(model.named_parameters())]


def _snapshot(model, keys):
    params = dict(model.named_parameters())
    return {k: params[k].detach().clone() for k in keys}


def _assert_really_loaded(model, before, source):
    """Compare key by key: a weight still bit-identical to the random init after loading was
    not loaded."""
    params = dict(model.named_parameters())
    stale = [k for k, v0 in before.items() if torch.equal(params[k].detach(), v0)]
    if stale:
        raise RuntimeError(
            f"[timm_compat] BiLoRA backbone did not actually load pretrained weights (source={source}): "
            f"{len(stale)}/{len(before)} key weights are bit-identical to the random init, first 5={stale[:5]}."
            " Refusing to continue with a random backbone (results would be noise)."
        )
    print(f"[timm_compat] backbone verified: all {len(before)} key weights were loaded from {source}.")


def _load_from_npz(model, path):
    from models.vit_base import _load_weights  # BiLoRA's own augreg npz loader

    _load_weights(model, path)


def _load_from_ssl_pth(model, path):
    from model_m.common.timm_backbone import _remap_ssl_state_dict

    sd = _remap_ssl_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    own = model.state_dict()
    take = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(take, strict=False)
    print(f"[timm_compat] SSL weights: loaded {len(take)} tensors by name <- {os.path.basename(path)}")


def _load_from_hub(model, variant):
    from timm.models._hub import load_state_dict_from_hf  # bypasses the model registry overwritten by BiLoRA

    hub_id = _HUB_MAP.get(variant, f"timm/{variant}")
    sd = load_state_dict_from_hf(hub_id)
    own = model.state_dict()
    take = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(take, strict=False)
    print(f"[timm_compat] hub weights: loaded {len(take)} tensors by name <- {hub_id}")


def make_create_vision_transformer(weights_file=None):
    """Return the replacement _create_vision_transformer (same signature as BiLoRA's original).
    weights_file: .npz = augreg (BiLoRA's native backbone); .pth = iBOT/DINO self-supervised;
    None = try the default npz, then the hub."""
    from models.fft import ViT_lora_fft  # deferred import: the BiLoRA repo must be on sys.path first

    def _create(variant, pretrained=False, **kwargs):
        kwargs.pop("representation_size", None)   # BiLoRA does not use the in21k representation layer (cls token only)
        kwargs.setdefault("num_classes", 0)       # head=Identity, out_dim=768
        model = ViT_lora_fft(**kwargs)
        if not pretrained:
            warnings.warn("[timm_compat] pretrained=False: the BiLoRA backbone is randomly initialised (unit tests only).")
            return model

        wf = weights_file
        if not wf and os.path.exists(DEFAULT_AUGREG_NPZ):
            wf = DEFAULT_AUGREG_NPZ

        before = _snapshot(model, _required_keys(model))
        if wf and wf.endswith(".npz"):
            _load_from_npz(model, wf)
            source = f"npz:{os.path.basename(wf)}"
        elif wf and os.path.exists(wf):
            _load_from_ssl_pth(model, wf)
            source = f"ssl_pth:{os.path.basename(wf)}"
        elif wf:
            # An explicit weights file that does not exist must not fall back to hub weights.
            raise FileNotFoundError(f"[timm_compat] weights file not found: {wf}")
        else:
            _load_from_hub(model, variant)
            source = f"hub:{_HUB_MAP.get(variant, variant)}"
        _assert_really_loaded(model, before, source)
        return model

    return _create


def patch_bilora_for_timm1(weights_file=None, force=False):
    """Replace models.fft._create_vision_transformer with the timm 1.x version. Idempotent; must
    be called before SiNet(). force=True re-patches when the weights file changes."""
    global _PATCHED
    if _PATCHED and not force:
        return
    import models.fft as fft

    fft._create_vision_transformer = make_create_vision_transformer(weights_file)
    _PATCHED = True
    print(f"[timm_compat] models.fft._create_vision_transformer replaced by the timm-1.x compatible version"
          f" (weights={weights_file or 'auto'}).")
