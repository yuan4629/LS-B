"""Brain-alignment scoring (BioScore) on top of the third-party brainnet encoder, plus small helpers."""
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _backbone_fingerprint(args, base_backbone):
    """Atlas cache fingerprint = backbone type + timm weight identity (model, weight file name, format)
    + PLModel training-quality params (brainnet_epochs / limit_train_batches / limit_val_batches / atlas_seed).
    Keying on the type name alone would silently reuse an old atlas after a weight swap (augreg -> iBOT);
    any change in the quality params also changes the key and forces a rebuild.
    Backbones without timm weights (CLIP) use type name + quality suffix only.

    Note: the fingerprint uses atlas_seed, not the CL seed. A brain prior that changed with the continual-
    learning seed would not be a prior, and would make brainnet selections vary across seeds through atlas
    noise (plus a ~25 min PLModel retrain per seed). atlas_seed defaults to 0; callers without the attribute
    fall back to seed so existing cache keys stay valid."""
    name = type(base_backbone).__name__
    # PLModel training-quality suffix: any change must trigger a rebuild (reuse only on an identical key).
    ep = getattr(args, "brainnet_epochs", "")
    ltb = getattr(args, "brainnet_limit_train_batches", "")
    lvb = getattr(args, "brainnet_limit_val_batches", "")
    seed = getattr(args, "atlas_seed", None)
    if seed is None:
        seed = getattr(args, "seed", "")
    qual = f"ep={ep}-ltb={ltb}-lvb={lvb}-seed={seed}"
    if "Timm" not in name:
        return re.sub(r"[^A-Za-z0-9._=-]+", "-", f"{name}__{qual}")
    timm_model = getattr(args, "timm_model", "") or ""
    timm_weights = getattr(args, "timm_weights", "") or ""
    timm_fmt = getattr(args, "timm_weights_format", "") or ""
    wb = os.path.basename(str(timm_weights)) if timm_weights else "none"
    return re.sub(r"[^A-Za-z0-9._=-]+", "-", f"{name}__{timm_model}__{wb}__{timm_fmt}__{qual}")


def select_layers_from_scores(scores, threshold, min_layers=1):
    num_layers = len(scores)
    
    # threshold is a fraction of layers, e.g. 0.2 * 12 -> k = int(2.4) = 2; clamp to [min_layers, num_layers].
    k = int(num_layers * threshold)
    k = max(k, min_layers)
    k = min(k, num_layers)

    # Top-k scoring layers, returned in ascending layer order.
    top_k_indices = np.argsort(scores)[-k:].tolist()

    return sorted(top_k_indices)


class BioScoreCalculator:
    """Offline atlas (W_roi_ch, A_roi_layer) + online task profile.
    The atlas is built once and cached on disk; each CIL task then costs one backbone
    forward pass + two matrix products."""

    def __init__(self, args, device, base_backbone):
        self.args = args
        self.device = device
        self.base_backbone = base_backbone  # ModifiedCLIP or ModifiedTimmViT
        # Atlas cache is keyed by backbone type + weight identity, so CLIP/timm and augreg/iBOT never share one.
        self.fingerprint = _backbone_fingerprint(args, base_backbone)
        self.cache_dir = Path(args.bioscore_cache_dir) / self.fingerprint
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.atlas_path = self.cache_dir / "atlas.pt"
        self.ckpt_path = self.cache_dir / "plmodel.ckpt"
        self.meta_path = self.cache_dir / "meta.json"

        self.W_roi_ch = None       # [R, D]
        self.A_roi_layer = None    # [R, L]
        self.roi_names = None
        self.bottlenecks = None    # nn.ModuleDict: layer_key -> Linear(width, D)
        self.layers = None         # ["0", ..., "11"]
        self.is_mock = False       # True = random fallback atlas (no real fMRI/PLModel); checked before brainnet selection
        self._build_or_load_atlas()

    # ------------------------------------------------------------------
    # Build / load entry point
    # ------------------------------------------------------------------
    def _build_or_load_atlas(self):
        # force_rebuild: ignore the on-disk atlas/ckpt, retrain once and overwrite the cache.
        # The _BIO_CALCS memo still builds each key once per process, so thresholds/tasks reuse this rebuild.
        force = bool(getattr(self.args, "brainnet_force_rebuild", False))
        if force:
            print("[BioScore] brainnet_force_rebuild=True -> ignoring disk cache, retraining PLModel once.")
        # 1) Prefer the cached atlas.
        if not force and self.atlas_path.exists():
            try:
                self._load_atlas()
                print(f"[BioScore] loaded cached atlas from {self.atlas_path}")
                return
            except Exception as e:
                # Must be loud: a silent rebuild makes it unknowable which atlas a run used, and the weak
                # task signal is very sensitive to atlas differences (selections change substantially).
                print(f"[BioScore] WARNING: cached atlas failed to load -> **rebuilding**; results are not comparable with earlier runs!"
                      f" path={self.atlas_path} reason={e}")
                warnings.warn(f"[BioScore] cached atlas load failed: {e}")

        # 2) ckpt available -> rebuild PLModel and extract the atlas.
        plm = None
        if not force and self.ckpt_path.exists():
            try:
                plm = self._load_plmodel_from_ckpt()
                print(f"[BioScore] loaded PLModel ckpt from {self.ckpt_path}")
            except Exception as e:
                warnings.warn(f"[BioScore] ckpt load failed: {e}")

        # 3) fMRI data available -> train PLModel once.
        if plm is None and self._fmri_data_available():
            try:
                print(f"[BioScore] training PLModel once ({self.args.brainnet_epochs} epochs) -> caching to {self.ckpt_path}")
                plm = self._train_plmodel()
            except Exception as e:
                warnings.warn(f"[BioScore] PLModel training failed: {e}")

        # 4) Fallback: mock atlas.
        if plm is None:
            warnings.warn("[BioScore] using Mock Atlas (no PLModel weights, no fMRI data).")
            self._mock_atlas()
            self._save_atlas()
            return

        self._extract_atlas_from_plmodel(plm)
        self._save_atlas()

    def _fmri_data_available(self):
        root = Path(self.args.fmri_data_dir)
        return (root / "training_split" / "training_fmri").exists() and \
               (root / "training_split" / "training_images").exists()

    def _train_plmodel(self):
        from pytorch_lightning import Trainer
        from brainnet.config import get_cfg_defaults
        from brainnet.plmodel import PLModel

        cfg = get_cfg_defaults()
        cfg.DATASET.DATA_DIR = self.args.fmri_data_dir
        cfg.DATASET.RESOLUTION = (224, 224)
        plm = PLModel(cfg, self.base_backbone, draw=False, cached=self.args.brainnet_cached, skip_data=False)
        trainer = Trainer(
            max_epochs=self.args.brainnet_epochs,
            accelerator="gpu" if self.device.type == "cuda" else "cpu",
            devices=1,
            precision=16 if self.device.type == "cuda" else 32,
            limit_train_batches=self.args.brainnet_limit_train_batches,
            limit_val_batches=self.args.brainnet_limit_val_batches,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=True,  # training takes ~30 min; show progress so it does not look hung
        )
        trainer.fit(plm)
        trainer.save_checkpoint(str(self.ckpt_path))
        return plm

    def _load_plmodel_from_ckpt(self):
        from brainnet.config import get_cfg_defaults
        from brainnet.plmodel import PLModel

        cfg = get_cfg_defaults()
        cfg.DATASET.DATA_DIR = self.args.fmri_data_dir
        cfg.DATASET.RESOLUTION = (224, 224)
        plm = PLModel(cfg, self.base_backbone, draw=False, cached=False, skip_data=True)
        state = torch.load(str(self.ckpt_path), map_location="cpu")
        sd = state.get("state_dict", state)
        plm.load_state_dict(sd, strict=False)
        return plm

    # ------------------------------------------------------------------
    # ROI index resolution (real, with mock fallback)
    # ------------------------------------------------------------------
    def _resolve_roi_indices(self):
        try:
            from brainnet.roi import roi_dict, nsdgeneral_indices  # real anatomical ROIs (third-party brainnet)

            n_local = int(nsdgeneral_indices.shape[0])
            # fsaverage (327684) -> local (37984) lookup; -1 = outside nsdgeneral
            fsav_to_local = np.full(327684, -1, dtype=np.int64)
            fsav_to_local[nsdgeneral_indices] = np.arange(n_local)

            real_map = {}
            for name, fsav_idx in roi_dict.items():
                local = fsav_to_local[np.asarray(fsav_idx, dtype=np.int64)]
                local = local[local >= 0]
                if local.size > 0:
                    real_map[name] = local
            if len(real_map) >= 3:
                return real_map, n_local
            warnings.warn("[BioScore] real ROI map too sparse, fallback to mock segments.")
        except Exception as e:
            warnings.warn(f"[BioScore] real ROI load failed ({e}), fallback to mock segments.")

        # Mock: 5 equal segments.
        n_local = 37984
        chunk = n_local // 5
        names = ["V1", "V2", "V4", "EBA", "FFA"]
        mock_map = {}
        for i, name in enumerate(names):
            start = i * chunk
            end = (i + 1) * chunk if i < 4 else n_local
            mock_map[name] = np.arange(start, end, dtype=np.int64)
        return mock_map, n_local

    # ------------------------------------------------------------------
    # Atlas extraction from PLModel / random mock
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extract_atlas_from_plmodel(self, plm):
        plm.eval().to(self.device)
        weight = plm.model.weight.detach()                     # [N, D]
        _, sel_layer, _ = plm.get_selectors()
        sel_layer = sel_layer.detach()                         # [N, L]

        roi_map, _ = self._resolve_roi_indices()
        self.roi_names = list(roi_map.keys())
        R = len(self.roi_names)
        D = weight.shape[1]
        L = sel_layer.shape[1]

        W = torch.zeros(R, D, device=weight.device)
        A = torch.zeros(R, L, device=weight.device)
        for i, name in enumerate(self.roi_names):
            idx = torch.from_numpy(roi_map[name]).long().to(weight.device)
            W[i] = weight[idx].mean(dim=0)
            A[i] = sel_layer[idx].mean(dim=0)
        self.W_roi_ch = W.to(self.device)
        self.A_roi_layer = A.to(self.device)
        self.is_mock = False  # real atlas (from PLModel)

        # Copy the global-token bottleneck weights (one Linear(width, D) per layer).
        self.bottlenecks = nn.ModuleDict()
        for layer_key, lin in plm.model.global_token_bottleneck.items():
            new_lin = nn.Linear(lin.in_features, lin.out_features, bias=False)
            new_lin.load_state_dict(lin.state_dict())
            self.bottlenecks[layer_key] = new_lin.to(self.device)
        self.layers = list(plm.model.layers)
        print(f"[BioScore] atlas extracted: R={R} D={D} L={L} ROIs={self.roi_names}")

    def _mock_atlas(self):
        # Always the 5-segment layout (real ROIs are ignored because there is no PLModel in mock mode).
        self.is_mock = True  # random fallback atlas; brainnet selection refuses it by default (see get_selector_scores)
        R, D, L = 5, 128, 12
        self.roi_names = ["V1", "V2", "V4", "EBA", "FFA"]
        self.layers = [str(i) for i in range(L)]

        gen = torch.Generator(device="cpu").manual_seed(int(self.args.seed))
        W = torch.randn(R, D, generator=gen) * 0.1
        A_logits = torch.randn(R, L, generator=gen)
        A = torch.softmax(A_logits, dim=-1)
        self.W_roi_ch = W.to(self.device)
        self.A_roi_layer = A.to(self.device)

        self.bottlenecks = nn.ModuleDict()
        for i in range(L):
            w = torch.randn(D, 768, generator=gen) * 0.02
            lin = nn.Linear(768, D, bias=False)
            with torch.no_grad():
                lin.weight.copy_(w)
            self.bottlenecks[str(i)] = lin.to(self.device)
        print(f"[BioScore] mock atlas built (seed={self.args.seed}): R={R} D={D} L={L}")

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------
    def _save_atlas(self):
        try:
            state = {
                "W_roi_ch": self.W_roi_ch.detach().cpu(),
                "A_roi_layer": self.A_roi_layer.detach().cpu(),
                "roi_names": self.roi_names,
                "layers": self.layers,
                "bottleneck_state_dict": {k: v.state_dict() for k, v in self.bottlenecks.items()},
                "is_mock": bool(self.is_mock),  # persisted so a reloaded mock atlas is never mistaken for a real one
                "fingerprint": self.fingerprint,  # weight identity; _load_atlas checks it to prevent cross-backbone reuse
            }
            torch.save(state, str(self.atlas_path))
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "backbone": type(self.base_backbone).__name__,
                    "fingerprint": self.fingerprint,
                    "timm_model": getattr(self.args, "timm_model", None),
                    "timm_weights": getattr(self.args, "timm_weights", None),
                    "timm_weights_format": getattr(self.args, "timm_weights_format", None),
                    "is_mock": bool(self.is_mock),
                    "fmri_data_dir": str(self.args.fmri_data_dir),
                    "brainnet_epochs": int(self.args.brainnet_epochs),
                    "roi_names": list(self.roi_names),
                    "R": len(self.roi_names),
                    "D": int(self.W_roi_ch.shape[1]),
                    "L": int(self.A_roi_layer.shape[1]),
                }, f, ensure_ascii=False, indent=2)
            print(f"[BioScore] atlas cached -> {self.atlas_path}")
        except Exception as e:
            warnings.warn(f"[BioScore] atlas save failed: {e}")

    def _load_atlas(self):
        # weights_only=False is explicit: PyTorch >= 2.6 defaults to True, which makes the whole load fail on
        # non-tensor objects; _build_or_load_atlas would then rebuild, so different torch versions would use
        # different atlases.
        state = torch.load(str(self.atlas_path), map_location=self.device, weights_only=False)
        self.W_roi_ch = state["W_roi_ch"].to(self.device)
        self.A_roi_layer = state["A_roi_layer"].to(self.device)
        self.roi_names = state["roi_names"]
        self.layers = state["layers"]
        self.is_mock = bool(state.get("is_mock", False))  # older caches lack the key -> treated as real
        fp = state.get("fingerprint")  # mismatch -> refuse and rebuild; older caches without the key are accepted
        if fp is not None and fp != self.fingerprint:
            raise ValueError(f"[BioScore] atlas fingerprint mismatch: cached={fp} current={self.fingerprint} (refused -> rebuild)")
        self.bottlenecks = nn.ModuleDict()
        for layer_key, sd in state["bottleneck_state_dict"].items():
            out_f, in_f = sd["weight"].shape
            lin = nn.Linear(in_f, out_f, bias=False)
            lin.load_state_dict(sd)
            self.bottlenecks[layer_key] = lin.to(self.device)

    # ------------------------------------------------------------------
    # Online scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_score(self, loader, max_batches=10):
        # bioscore_score_mode: last = one v_task from the last layer, shared by all layers;
        # perlayer = per-layer profile (each layer's own bottleneck + global token -> v_task_l).
        # Under "last" the score's shape across layers is dominated by the offline atlas.
        mode = getattr(self.args, "bioscore_score_mode", "last") or "last"
        if mode == "perlayer":
            return self._compute_score_perlayer(loader, max_batches)
        return self._compute_score_last(loader, max_batches)

    @torch.no_grad()
    def _compute_score_last(self, loader, max_batches=10):
        backbone = self.base_backbone.to(self.device).eval()
        last_layer = self.layers[-1]
        bottleneck = self.bottlenecks[last_layer].to(self.device).eval()

        feats = []
        for i, (x, _) in enumerate(loader, start=1):
            if max_batches and i > max_batches:
                break
            x = x.to(self.device, non_blocking=True)
            _, global_tokens = backbone.get_tokens(x)            # dict[layer] -> [B, 768]
            gt = global_tokens[last_layer]                       # [B, 768]
            feats.append(bottleneck(gt))                         # [B, D]

        L = self.A_roi_layer.shape[1]
        if not feats:
            return np.ones(L, dtype=np.float32)

        feats = torch.cat(feats, dim=0)                          # [N_total, D]
        c_act = feats.var(dim=0)                                 # [D]  channel activation strength (batch variance)
        v_task = torch.relu(self.W_roi_ch @ c_act)               # [R]  ROI activation vector
        score = v_task @ self.A_roi_layer                        # [L]  per-layer selection score
        score = score / (score.mean() + 1e-8)                    # mean-normalize for select_layers_from_scores
        return score.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _compute_score_perlayer(self, loader, max_batches=10):
        """Per-layer online profile: each layer's own bottleneck projects its own global token
        -> c_act_l -> v_task_l, then score[l] = <v_task_l, A[:, l]> (only that layer's atlas column).
        In "last" mode v_task comes from the last layer and is reused for every layer, so the score shape
        is fixed by the offline A_roi_layer. Per-layer profiles let backbone- and task-specific feature
        dynamics enter the score, using the per-layer bottlenecks the atlas already has."""
        backbone = self.base_backbone.to(self.device).eval()
        L = self.A_roi_layer.shape[1]
        per_layer = {lk: [] for lk in self.layers}               # accumulated bottleneck features per layer
        n_batches = 0
        for i, (x, _) in enumerate(loader, start=1):
            if max_batches and i > max_batches:
                break
            x = x.to(self.device, non_blocking=True)
            _, global_tokens = backbone.get_tokens(x)            # dict[layer] -> [B, 768]
            for lk in self.layers:
                bl = self.bottlenecks[lk].to(self.device).eval()
                per_layer[lk].append(bl(global_tokens[lk]))      # [B, D]
            n_batches += 1

        if n_batches == 0:
            return np.ones(L, dtype=np.float32)

        score = torch.zeros(L, device=self.device)
        for li, lk in enumerate(self.layers):                    # self.layers matches A_roi_layer column order
            f = torch.cat(per_layer[lk], dim=0)                  # [N, D]
            c_act = f.var(dim=0)                                 # [D]  this layer's channel activation strength
            v_task = torch.relu(self.W_roi_ch @ c_act)           # [R]  this layer's ROI activation vector
            score[li] = v_task @ self.A_roi_layer[:, li]         # scalar: own column only
        score = score / (score.mean() + 1e-8)                    # same normalization as "last"
        return score.detach().cpu().numpy().astype(np.float32)


_BIO_CALCS = {}


def random_scores(num_layers, seed, task_id=0):
    """Random-layer control: random per-layer scores, so select_layers_from_scores picks random layers
    with the same k. Separates "which layers" from "how many layers / merge_all interference".
    Seeded by seed and task_id jointly: reproducible and varies per task (like brainnet/grad)."""
    rng = np.random.RandomState((int(seed) * 100003 + int(task_id)) % (2 ** 31 - 1))
    return rng.rand(int(num_layers)).astype(np.float32)
