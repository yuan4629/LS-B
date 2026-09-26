"""timm ViT-B/16 backbone + classifier heads, mirroring clip_backbone.py.

Lets model_m_timm and the bound baselines run on the SAME backbone as BiLoRA
(vit_base_patch16_224.augreg_in21k):

  - ModifiedTimmViT          : timm ViT wrapper with the ModifiedCLIP interface of the
                               third-party brainnet package
  - TimmClassifier           : selector model (with .fc), mirrors CLIPClassifier
  - SharedHeadTimmClassifier : shared-head full fine-tuning model, mirrors SharedHeadCLIPClassifier
  - LoRATaskTimmClassifier   : model_m_timm, mirrors LoRATaskCLIPClassifier + two routes (ncm / merge_all)

Offline weights: when weights_file (npz) exists, build_timm_vit loads it via
pretrained_cfg_overlay=dict(file=...), avoiding HF cache paths that break across OSes.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm

from model_m.common.lora import LoRALinear


# LoRA injection targets shared by the upper/lower bound baselines: attention qkv + proj.
# Same attention surface as BiLoRA's "attention LoRA" (the fused qkv Linear adds to q, k, v alike).
# Both bounds use the same targets so they stay comparable.
BOUND_LORA_TARGETS = ("attn.qkv", "attn.proj")


def _remap_ssl_state_dict(raw):
    """Normalize a self-supervised (DINO/iBOT) .pth state_dict to timm ViT key names:
    1) take the sub-dict (DINO/iBOT often store {teacher/student/state_dict/model: sd});
    2) strip common prefixes (module./backbone./encoder.).
    The ViT trunk keys (cls_token/pos_embed/patch_embed/blocks.*/norm.*) already match timm, so the
    result loads with strict=False; extra head/projection keys end up in unexpected_keys."""
    sd = raw
    if isinstance(raw, dict):
        for k in ("teacher", "student", "state_dict", "model"):
            if k in raw and isinstance(raw[k], dict):
                sd = raw[k]
                break
    out = {}
    for k, v in sd.items():
        nk = k
        for pre in ("module.", "backbone.", "encoder."):
            if nk.startswith(pre):
                nk = nk[len(pre):]
        out[nk] = v
    return out


def build_timm_vit(model_name="vit_base_patch16_224.augreg_in21k", weights_file=None, pretrained=True,
                   weights_format="augreg_npz"):
    """Build a timm ViT (num_classes=0, feature extractor only).
    weights_format: augreg_npz = JAX npz overlay; dino_pth/ibot_pth = self-supervised .pth + key remap (strict=False).
    For BiLoRA's self-supervised backbones use dino_pth/ibot_pth with --timm_model vit_base_patch16_224."""
    if not (pretrained and weights_file and os.path.exists(weights_file)):
        return timm.create_model(model_name, pretrained=bool(pretrained), num_classes=0)
    if weights_format == "augreg_npz":
        return timm.create_model(
            model_name, pretrained=True, num_classes=0,
            pretrained_cfg_overlay=dict(file=weights_file),
        )
    # dino_pth / ibot_pth: random-init architecture + remapped SSL weights (strict=False tolerates head/projection mismatch).
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    # weights_only=False: DINO/iBOT checkpoints contain non-tensor metadata (args Namespace, epoch).
    sd = _remap_ssl_state_dict(torch.load(weights_file, map_location="cpu", weights_only=False))
    incompat = model.load_state_dict(sd, strict=False)
    missing = [k for k in incompat.missing_keys if not k.startswith("head")]
    if missing:
        print(f"[build_timm_vit:{weights_format}] {len(missing)} missing(non-head) first5={missing[:5]}")
    if incompat.unexpected_keys:
        print(f"[build_timm_vit:{weights_format}] {len(incompat.unexpected_keys)} unexpected first5={incompat.unexpected_keys[:5]}")
    return model


class ModifiedTimmViT(nn.Module):
    """timm ViT wrapper: vision_model is the timm model (selectors address it via .blocks);
    encode() returns the final-normed cls token (as BiLoRA's extract_vector); get_tokens() feeds BioScore."""

    def __init__(self, model_name="vit_base_patch16_224.augreg_in21k", weights_file=None, pretrained=True,
                 weights_format="augreg_npz"):
        super().__init__()
        self.vision_model = build_timm_vit(model_name, weights_file, pretrained, weights_format)
        self.vision_model.requires_grad_(False)
        self.vision_model.eval()
        self.feature_dim = self.vision_model.num_features  # 768

    def encode(self, x):
        # Global feature = cls token after forward_features (includes final norm). LoRA hooks apply during forward.
        return self.vision_model.forward_features(x)[:, 0]

    def get_tokens(self, x):
        # Per-layer (local, global) tokens, mirroring ModifiedMAE.get_tokens (global = per-layer cls, before final norm).
        vm = self.vision_model
        x = vm.patch_embed(x)
        x = vm._pos_embed(x)      # prepend cls token + position embedding
        x = vm.patch_drop(x)
        x = vm.norm_pre(x)
        local_tokens, global_tokens = {}, {}
        for i, blk in enumerate(vm.blocks):
            x = blk(x)
            saved = x.clone()
            global_tokens[str(i)] = saved[:, 0, :]            # [B, 768]
            patches = saved[:, 1:, :]                          # [B, N, 768]
            p = int(np.sqrt(patches.shape[1]))
            local_tokens[str(i)] = rearrange(patches, "b (p1 p2) c -> b c p1 p2", p1=p, p2=p)
        return local_tokens, global_tokens


class TimmClassifier(nn.Module):
    """Selector model: timm backbone + linear head. The grad selector uses .fc and the blocks (.blocks)."""

    def __init__(self, num_classes=100, model_name="vit_base_patch16_224.augreg_in21k",
                 weights_file=None, pretrained=True, weights_format="augreg_npz"):
        super().__init__()
        self.backbone = ModifiedTimmViT(model_name, weights_file, pretrained, weights_format)
        self.fc = nn.Linear(self.backbone.feature_dim, num_classes)

    def forward(self, x):
        return self.fc(self.backbone.encode(x))


class SharedHeadTimmClassifier(nn.Module):
    """Class-IL shared-head full fine-tuning model (timm version), mirrors SharedHeadCLIPClassifier."""

    task_aware = False

    def __init__(self, num_classes=100, model_name="vit_base_patch16_224.augreg_in21k",
                 weights_file=None, pretrained=True, feature_dim=None, weights_format="augreg_npz"):
        super().__init__()
        self.backbone = ModifiedTimmViT(model_name, weights_file, pretrained, weights_format)
        self.num_classes = num_classes
        self.feature_dim = feature_dim or self.backbone.feature_dim
        self.shared_head = nn.Linear(self.feature_dim, num_classes)
        self.register_buffer("class_seen_mask", torch.zeros(num_classes, dtype=torch.bool))

    def encode(self, x):
        return self.backbone.encode(x)

    def freeze_for_task(self, train_backbone=True):
        for p in self.backbone.parameters():
            p.requires_grad = train_backbone
        for p in self.shared_head.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def update_seen_classes(self, loader, device=None, max_steps=0):
        del device
        for step, (_, y) in enumerate(loader, start=1):
            if max_steps and step > max_steps:
                break
            for cls in y.unique(sorted=True).tolist():
                self.class_seen_mask[int(cls)] = True

    def mask_unseen_logits(self, logits):
        if not self.class_seen_mask.any().item():
            return logits
        unseen_mask = ~self.class_seen_mask.to(logits.device)
        if unseen_mask.any().item():
            logits = logits.masked_fill(unseen_mask.unsqueeze(0), torch.finfo(logits.dtype).min)
        return logits

    def forward(self, x, restrict_to_seen=None):
        logits = self.shared_head(self.encode(x))
        if restrict_to_seen:
            logits = self.mask_unseen_logits(logits)
        return logits


class LoRATaskTimmClassifier(nn.Module):
    """model_m_timm: naive LoRA (mlp.fc1/fc2) on selected layers + shared head + two routes.

    route="merge_all" (default, as BiLoRA): train and test both sum all existing task LoRAs (each on its
    own selected layers; overlapping hooks add up) in one forward pass; only the current task's bank trains.
    route="ncm" (ablation): at test time NCM routes each sample to a single task's LoRA."""

    accepts_task_id = True
    task_aware = False
    LORA_TARGETS = ("mlp.fc1", "mlp.fc2")

    def __init__(self, num_classes=100, lora_rank=16, lora_alpha=16.0, route="merge_all",
                 model_name="vit_base_patch16_224.augreg_in21k", weights_file=None,
                 pretrained=True, feature_dim=None, weights_format="augreg_npz"):
        super().__init__()
        self.backbone = ModifiedTimmViT(model_name, weights_file, pretrained, weights_format)
        self.num_classes = num_classes
        self.feature_dim = feature_dim or self.backbone.feature_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.route = route

        self.shared_head = nn.Linear(self.feature_dim, num_classes)
        self.lora_weights = nn.ModuleDict()  # task_id -> ModuleDict(layer_idx -> ModuleDict(sane_target -> LoRALinear))

        self.register_buffer("class_prototypes", torch.zeros(num_classes, self.feature_dim))
        self.register_buffer("class_seen_mask", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("class_to_task", torch.full((num_classes,), -1, dtype=torch.long))

    # ---------------- LoRA injection ----------------
    @property
    def _sane_to_path(self):
        return {t.replace(".", "_"): t for t in self.LORA_TARGETS}

    def add_task(self, task_id, selected_layers=None):
        key = str(task_id)
        if key in self.lora_weights:
            return
        selected_layers = selected_layers or []
        blocks = self.backbone.vision_model.blocks
        bank = nn.ModuleDict()
        for i in selected_layers:
            if not (0 <= i < len(blocks)):
                continue
            layer_mod = nn.ModuleDict()
            for tgt in self.LORA_TARGETS:
                linear = blocks[i].get_submodule(tgt)
                layer_mod[tgt.replace(".", "_")] = LoRALinear(
                    linear.in_features, linear.out_features, rank=self.lora_rank, alpha=self.lora_alpha
                )
            bank[str(i)] = layer_mod
        self.lora_weights[key] = bank

    @staticmethod
    def _make_hook(lora):
        def hook(module, inputs, output):
            return output + lora(inputs[0]).to(output.dtype)
        return hook

    def _encode_raw(self, x):
        return self.backbone.encode(x)

    def encode_base(self, x):
        return self._encode_raw(x)

    def _encode_with_banks(self, x, task_keys):
        """Temporarily hook fc1/fc2 on all selected layers of task_keys, then run one forward pass.
        Multiple hooks on the same linear accumulate -> sum_t lora_t(in), the merge_all semantics."""
        keys = [k for k in task_keys if k in self.lora_weights and len(self.lora_weights[k]) > 0]
        if not keys:
            return self.encode_base(x)
        blocks = self.backbone.vision_model.blocks
        sane_to_path = self._sane_to_path
        handles = []
        try:
            for k in keys:
                for layer_key, layer_mod in self.lora_weights[k].items():
                    block = blocks[int(layer_key)]
                    for sane, lora in layer_mod.items():
                        linear = block.get_submodule(sane_to_path[sane])
                        handles.append(linear.register_forward_hook(self._make_hook(lora)))
            return self._encode_raw(x)
        finally:
            for h in handles:
                h.remove()

    def encode_with_task(self, x, task_id):
        return self._encode_with_banks(x, [str(task_id)])

    def encode_with_all_tasks(self, x):
        return self._encode_with_banks(x, list(self.lora_weights.keys()))

    def encode(self, x, task_id=None):
        if task_id is None:
            return self.encode_base(x)
        return self.encode_with_task(x, task_id)

    def freeze_for_task(self, task_id, train_backbone=False):
        for p in self.backbone.parameters():
            p.requires_grad = train_backbone
        for p in self.shared_head.parameters():
            p.requires_grad = True
        for bank in self.lora_weights.values():
            for p in bank.parameters():
                p.requires_grad = False
        if str(task_id) in self.lora_weights:
            for p in self.lora_weights[str(task_id)].parameters():
                p.requires_grad = True

    def task_lora_params(self, task_id):
        key = str(task_id)
        if key not in self.lora_weights:
            return 0
        return sum(p.numel() for p in self.lora_weights[key].parameters())

    # ---------------- NCM routing (route="ncm") ----------------
    def mask_unseen_logits(self, logits):
        if not self.class_seen_mask.any().item():
            return logits
        unseen_mask = ~self.class_seen_mask.to(logits.device)
        if unseen_mask.any().item():
            logits = logits.masked_fill(unseen_mask.unsqueeze(0), torch.finfo(logits.dtype).min)
        return logits

    def predict_task_ids_from_base(self, base_feat):
        seen_classes = torch.where(self.class_seen_mask)[0]
        if seen_classes.numel() == 0:
            raise RuntimeError("Class-IL routing requires prototypes, but no class prototype has been registered yet.")
        feat = F.normalize(base_feat, dim=-1)
        prototypes = F.normalize(self.class_prototypes[seen_classes].to(base_feat.device), dim=-1)
        nearest = (feat @ prototypes.t()).argmax(dim=1)
        pred_classes = seen_classes.to(base_feat.device)[nearest]
        pred_tasks = self.class_to_task[pred_classes]
        if (pred_tasks < 0).any():
            raise RuntimeError("Found class prototype(s) without a valid task assignment.")
        return pred_tasks

    @torch.no_grad()
    def update_task_prototypes(self, loader, device, task_id, max_steps=0):
        """Build NCM prototypes from un-augmented base features and set class_seen_mask / class_to_task.
        merge_all does not use prototypes but needs seen_mask, so both routes call this."""
        was_training = self.training
        self.to(device).eval()
        proto_sum, proto_count = {}, {}
        for step, (x, y) in enumerate(loader, start=1):
            if max_steps and step > max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feat = F.normalize(self.encode_base(x), dim=-1)
            for cls in y.unique(sorted=True).tolist():
                cls = int(cls)
                cls_feat = feat[y == cls]
                if cls not in proto_sum:
                    proto_sum[cls] = cls_feat.sum(dim=0)
                    proto_count[cls] = cls_feat.size(0)
                else:
                    proto_sum[cls] += cls_feat.sum(dim=0)
                    proto_count[cls] += cls_feat.size(0)
        if not proto_sum:
            raise RuntimeError(f"Failed to build prototypes for task {task_id}: loader is empty.")
        for cls, feat_sum in proto_sum.items():
            prototype = F.normalize(feat_sum / max(proto_count[cls], 1), dim=0)
            self.class_prototypes[cls].copy_(prototype.to(self.class_prototypes.device))
            self.class_seen_mask[cls] = True
            self.class_to_task[cls] = int(task_id)
        if was_training:
            self.train()

    # ---------------- forward ----------------
    def forward(self, x, task_id=None, restrict_to_seen=None, return_task_ids=False):
        if self.route == "merge_all":
            # Train (task_id given) and test (None) both sum all existing banks; banks are added
            # incrementally, so training task tid uses banks 0..tid.
            feat = self.encode_with_all_tasks(x)
            routed_task_ids = torch.full((x.size(0),), -1, device=x.device, dtype=torch.long)
        elif task_id is not None:
            routed_task_ids = torch.full((x.size(0),), int(task_id), device=x.device, dtype=torch.long)
            feat = self.encode_with_task(x, task_id)
        else:
            base_feat = self.encode_base(x)
            routed_task_ids = self.predict_task_ids_from_base(base_feat)
            feat = torch.empty_like(base_feat)
            for routed_task in routed_task_ids.unique(sorted=True).tolist():
                routed_task = int(routed_task)
                task_mask = routed_task_ids == routed_task
                if str(routed_task) in self.lora_weights:
                    feat[task_mask] = self.encode_with_task(x[task_mask], routed_task)
                else:
                    feat[task_mask] = base_feat[task_mask]

        logits = self.shared_head(feat)
        if restrict_to_seen is None:
            restrict_to_seen = task_id is None
        if restrict_to_seen:
            logits = self.mask_unseen_logits(logits)
        if return_task_ids:
            return logits, routed_task_ids
        return logits
