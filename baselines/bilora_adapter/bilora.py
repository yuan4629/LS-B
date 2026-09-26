# -*- coding: utf-8 -*-
"""BiLoRA baseline adapter (unified interface).

NOTE ON LOCATION: on a case-insensitive filesystem (Windows) a directory named
baselines/bilora/ would collide with the upstream "BiLoRA" repo directory, so
this adapter lives in baselines/bilora_adapter/. The registered method name is
still "bilora".

Thin wrapper around the upstream BiLoRA repository (a third-party dependency,
cloned separately into ./baselines/BiLoRA). BiLoRA runs on its OWN native
DataManager + iCIFAR100 (timm vit_base_patch16_224_in21k backbone,
Normalize(mean=0,std=1)), forced to the chrono split (shuffle=False,
init_cls=increment=classes_per_task, data_root) so the class grouping matches
the other methods' chrono split.

The passed-in framework loaders are intentionally ignored (BiLoRA's harness needs
its own (idx, x, y) DummyDataset + label remapping where target // class_num ==
task_id). Outputs are mapped to the standard result dict via a per-task
task_matrix, using the same acc/bwt formulas as core.metrics.

Heavy imports (timm/BiLoRA) are deferred into run() so registry import is cheap.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from core.metrics import matrix_to_jsonable

_BILORA_ROOT = Path(__file__).resolve().parents[1] / "BiLoRA"

# Defaults copied from baselines/BiLoRA/configs/cifar100_bilora.json.
_BILORA_CFG = {
    "prefix": "reproduce",
    "dataset": "cifar100",
    "data_path": "data/",
    "memory_size": 0,
    "memory_per_class": 0,
    "fixed_memory": True,
    "shuffle": False,
    "init_cls": 10,
    "increment": 10,
    "model_name": "bilora",
    "net_type": "sip",
    "embd_dim": 768,
    "num_heads": 12,
    "total_sessions": 10,
    "seed": 0,
    "EPSILON": 1e-8,
    "init_epoch": 40,
    "optim": "adam",
    "init_lr": 0.0005,
    "init_lr_decay": 0.1,
    "init_weight_decay": 0.0,
    "epochs": 20,
    "lrate": 0.0005,
    "lrate_decay": 0.1,
    "batch_size": 128,
    "weight_decay": 0.0,
    "rank": 10,
    "lamb": 0.95,
    "lame": 1.0,
    "num_workers": 16,
    "prompt_param": [10, 10],
}


def _load_bilora(weights_file=None):
    root = str(_BILORA_ROOT)
    if not Path(root).exists():
        raise FileNotFoundError(f"BiLoRA repo not found: {root}")
    if root not in sys.path:
        sys.path.insert(0, root)
    # timm 1.x shim: BiLoRA was written for timm 0.6 and fails to build otherwise (see timm_compat).
    from baselines.bilora_adapter.timm_compat import (  # noqa: E402
        patch_bilora_for_timm1, restore_timm_registry, snapshot_timm_registry,
    )

    # Snapshot before importing BiLoRA: its models/vit_base.py re-registers timm models under the
    # same names; without restoring, the BioScore scoring backbone (ModifiedTimmViT) cannot be built.
    snap = snapshot_timm_registry()
    from utils.data_manager import DataManager  # noqa: E402
    from methods.bilora import BiLoRA  # noqa: E402
    from utils.toolkit import count_parameters  # noqa: E402

    restore_timm_registry(snap)
    # Patch before BiLoRA(cfg) builds SiNet; the patch refuses a silently random-initialised backbone.
    patch_bilora_for_timm1(weights_file, force=bool(weights_file))
    _patch_label_dtype()
    return DataManager, BiLoRA, count_parameters


def _patch_label_dtype():
    """DummyDataset returns numpy labels; on Windows numpy's default int is int32 -> torch.int32, and
    cross_entropy fails with "nll_loss... not implemented for 'Int'" (Linux defaults to int64).
    BiLoRA itself is left unmodified, so the adapter casts labels to Python int (-> torch.int64). Idempotent."""
    from utils.data_manager import DummyDataset

    if getattr(DummyDataset, "_label_dtype_patched", False):
        return
    orig = DummyDataset.__getitem__

    def __getitem__(self, idx):
        i, image, label = orig(self, idx)
        return i, image, int(label)

    DummyDataset.__getitem__ = __getitem__
    DummyDataset._label_dtype_patched = True


def _build_cfg(args, device):
    cfg = dict(_BILORA_CFG)
    cfg["data_path"] = args.data_root
    # Dataset wiring: BiLoRA's DataManager supports imagenet_r natively (utils/data.py iIMAGENET_R),
    # but its data_path needs train/ test/ subdirectories; inr_data materialises them from the official
    # split (never triggering BiLoRA's own unseeded 80/20 split). Unknown datasets raise instead of
    # silently falling back to CIFAR.
    ds = str(getattr(args, "dataset", "cifar100") or "cifar100")
    if ds == "imagenet_r":
        from baselines.bilora_adapter.inr_data import ensure_bilora_imagenet_r
        cfg["dataset"] = "imagenet_r"
        cfg["data_path"] = ensure_bilora_imagenet_r(args)
    elif ds == "cub":
        # BiLoRA's iCUB ignores data_path and hard-codes data/cub/{train,test} relative to cwd
        # (utils/data.py:66-67). ensure_bilora_cub checks that this path is <data_root>/cub and that
        # class dirs / image counts / file-list sha match the APER split; any mismatch raises.
        from baselines.bilora_adapter.cub_data import ensure_bilora_cub
        cfg["dataset"] = "cub"
        cfg["data_path"] = ensure_bilora_cub(args)
    elif ds != "cifar100":
        raise ValueError(f"bilora host does not support dataset={ds!r} (choices: cifar100/imagenet_r/cub).")
    cfg["init_cls"] = args.classes_per_task
    cfg["increment"] = args.classes_per_task
    cfg["total_sessions"] = args.num_tasks
    cfg["num_workers"] = args.num_workers
    cfg["batch_size"] = args.batch_size
    cfg["seed"] = args.seed
    cfg["device"] = [device]  # BaseLearner uses device[0] + treats list as gpu list
    # Small-step smoke: shrink epochs so --max_train_steps runs stay fast.
    # Must be >= 2: BiLoRA's CosineSchedule(K=init_epoch) divides by (K-1) (schedulers.py:54), so
    # K=1 raises ZeroDivisionError. Data is already truncated by _CappedDataManager, so 2 epochs is fast.
    if getattr(args, "max_train_steps", 0) and args.max_train_steps > 0:
        cfg["init_epoch"] = 2
        cfg["epochs"] = 2
    # Cheap proxy protocol: scale epochs proportionally. Unlike the max_train_steps smoke (2 epochs,
    # truncated data), this keeps the full data and schedule shape, so rankings can match the full
    # protocol (calibrate against known results). Lower bound 2 for the same CosineSchedule reason.
    sc = float(getattr(args, "epoch_scale", 1.0) or 1.0)
    if sc != 1.0:
        cfg["init_epoch"] = max(2, int(round(cfg["init_epoch"] * sc)))
        cfg["epochs"] = max(2, int(round(cfg["epochs"] * sc)))
        print(f"[bilora] epoch_scale={sc} -> init_epoch={cfg['init_epoch']} epochs={cfg['epochs']}")
    return cfg


class _CappedDataManager:
    """Proxy that caps train datasets to keep smoke runs fast (max_train_steps>0).

    Forwards everything to the real DataManager; only truncates the train split
    returned by get_dataset (preserving the (idx, x, y) DummyDataset contract)."""

    def __init__(self, dm, cap):
        self._dm = dm
        self._cap = cap

    def __getattr__(self, name):
        return getattr(self._dm, name)

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        ds = self._dm.get_dataset(indices, source, mode, appendent=appendent, ret_data=ret_data)
        if self._cap and source == "train" and not ret_data:
            n = min(len(ds), self._cap)
            g = torch.Generator().manual_seed(0)
            sel = torch.randperm(len(ds), generator=g)[:n].tolist()
            return Subset(ds, sel)
        return ds


@torch.no_grad()
def _eval_task_subset(network, dm, j, class_num, device, max_eval_steps, num_workers=0):
    """Class-IL (task-agnostic) accuracy on task j's test subset.

    `num_workers` parallelises JPEG decoding in the eval loader (~6x faster; T=10 runs evaluate
    55 times). Results should be bitwise identical (shuffle=False -> SequentialSampler; deterministic
    test transforms; integer counts), but with num_workers>0 DataLoader derives a worker base_seed,
    which consumes the global RNG in some torch versions. Verify bitwise equality against a
    num_workers=0 run after changing this.
    """
    test_ds = dm.get_dataset(np.arange(j * class_num, (j + 1) * class_num), source="test", mode="test")
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=num_workers)
    network.eval()
    correct = total = 0
    for step, (_, inputs, targets) in enumerate(loader, start=1):
        if max_eval_steps and step > max_eval_steps:
            break
        inputs = inputs.to(device)
        logits = network.interface(inputs)  # [B, numtask*class_num]
        pred = logits.argmax(dim=1).cpu()
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    return correct / max(total, 1)


def _make_gated_learner_cls(BiLoRA, gate):
    """Layer-selection plugin: run one selection + gating step before BiLoRA's training entry.

    Hooks train_function rather than _train: _train first sets requires_grad, then builds the
    optimizer, then calls train_function, so gating here overrides the coef it just unfroze on all
    12 layers. The optimizer holds all parameters, but Adam skips params whose grad is None and
    weight_decay=0, so frozen coef stay exactly at their initial 0."""

    class _GatedBiLoRA(BiLoRA):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.gate = gate
            self.train_sec = []

        def train_function(self, train_loader, test_loader, optimizer, scheduler):
            self.gate.select_and_apply(self, train_loader)
            t0 = time.time()
            super().train_function(train_loader, test_loader, optimizer, scheduler)
            self.train_sec.append(time.time() - t0)

    return _GatedBiLoRA


def run_bilora(train_loaders, val_loaders, test_loaders, ncls, args, device, gate=None):
    tag = "BILORA" if gate is None else f"BILORA+GATE({gate.mode}@{gate.threshold})"
    print(f"\n[{tag}] start (native DataManager, chrono split)")
    DataManager, BiLoRA, count_parameters = _load_bilora(getattr(args, "bilora_weights", None) or None)
    if gate is not None:
        gate.install()   # the forward-skip patch depends on models.fft, so it must follow _load_bilora
    cfg = _build_cfg(args, device)

    # chrono = shuffle=False; iCIFAR100.class_order is arange(100) (contiguous tasks).
    dm = DataManager(cfg["dataset"], cfg["shuffle"], cfg["seed"], cfg["init_cls"], cfg["increment"], cfg)
    cap = (args.batch_size * args.max_train_steps) if getattr(args, "max_train_steps", 0) and args.max_train_steps > 0 else 0
    dm_drv = _CappedDataManager(dm, cap) if cap else dm

    # Learner wrapper: if the gate provides make_learner_cls (the D2 host rebuilds optimizer/scheduler
    # and tracks the shared-slot delta norm, which needs a wider hook than select_and_apply), use its
    # factory; otherwise use the plain selection-plugin wrapper (BioScoreLayerGate, unchanged).
    if gate is None:
        model = BiLoRA(cfg)
    else:
        factory = getattr(gate, "make_learner_cls", None)
        learner_cls = factory(BiLoRA) if factory is not None else _make_gated_learner_cls(BiLoRA, gate)
        model = learner_cls(cfg)
    class_num = args.classes_per_task
    t = args.num_tasks
    matrix = np.full((t, t), np.nan, dtype=np.float32)
    accs, bwts, caps = [], [], []
    trainable_curve = []
    start = time.time()

    for tid in range(t):
        print(f"[{tag}] task {tid + 1}/{t}")
        model.incremental_train(dm_drv)

        network = model._network.to(device)
        sample_counts = []
        for j in range(tid + 1):
            test_ds = dm.get_dataset(np.arange(j * class_num, (j + 1) * class_num), source="test", mode="test")
            # cap at 8 so two concurrent runs do not spawn 2x16 worker processes and exhaust RAM
            matrix[tid, j] = _eval_task_subset(network, dm, j, class_num, device, args.max_eval_steps,
                                               num_workers=min(int(getattr(args, "num_workers", 0) or 0), 8))
            n = len(test_ds) if not (args.max_eval_steps) else min(len(test_ds), 64 * args.max_eval_steps)
            sample_counts.append(n)

        # acc/bwt formulas mirror core.metrics.update_matrix_and_metrics.
        weights = np.asarray(sample_counts, dtype=np.float32)
        acc = float(np.nansum(matrix[tid, : tid + 1] * weights) / max(weights.sum(), 1.0))
        if tid == 0:
            bwt = 0.0
        else:
            diag = np.diag(matrix)[:tid]
            bwt = float(np.nanmean(matrix[tid, :tid] - diag))

        trainable = count_parameters(model._network, True)
        total = count_parameters(model._network)
        cap_ratio = float(trainable) / max(int(total), 1)

        model.after_task()
        accs.append(acc)
        bwts.append(bwt)
        caps.append(cap_ratio)
        trainable_curve.append(int(trainable))
        print(f"[{tag}] ACC={acc:.4f} BWT={bwt:.4f} CAP={cap_ratio:.6f} trainable={trainable}")

    res = {
        # Output method label: a gate may provide method_label (bilora_d2); otherwise the legacy default.
        "method": "bilora" if gate is None else getattr(gate, "method_label", "bilora_gated"),
        "threshold": None if gate is None else gate.threshold,
        "task_matrix": matrix_to_jsonable(matrix),
        "acc_curve": accs,
        "bwt_curve": bwts,
        "cap_curve": caps,
        "selected_layers": [[] for _ in range(t)] if gate is None else gate.selected_layers,
        "layer_scores": [[] for _ in range(t)] if gate is None else gate.layer_scores,
        "runtime_sec": time.time() - start,
    }
    # Efficiency fields: trainable parameter count / selection wall-clock (BioScore's training-free cost) / training wall-clock.
    res["trainable_curve"] = trainable_curve
    if gate is not None:
        res["selector_mode"] = gate.mode
        res["select_sec_curve"] = gate.select_sec
        res["train_sec_curve"] = getattr(model, "train_sec", [])
        # v2-only diagnostics: per-task ROI profile (the per-task brain readout) and non-additive
        # interaction energy (~0 means the atlas carries no task-separable information). Empty otherwise.
        if getattr(gate, "roi_profiles", None):
            res["roi_profiles"] = gate.roi_profiles
            res["interaction_curve"] = gate.interaction
            # ||D^_t||/||D~_t||: what remains after task centering. Expected ~sqrt(tau) ~0.1; near 1 means it had no effect.
            res["residual_frac"] = getattr(gate, "residual_frac", [])
        # D2-host fields (shared/specific curves, shared-slot delta norm, allocation fingerprint), supplied by the gate.
        extra = getattr(gate, "extra_result_fields", None)
        if extra is not None:
            res.update(extra())
    # Checkpoint: final trainable parameters go into res; exp.py pops them before writing the json
    # and saves them with torch.save, so new evaluation endpoints need no retraining.
    res["_final_trainable"] = _collect_final_trainable(model._network)
    return res


def _collect_final_trainable(network):
    """Checkpoint: collect the run's final trainable parameters and eval routing heads (~3 MB) for exp.py.

    Raises if any of the three groups is missing (a silently empty checkpoint is worse than none):
      coef* (task-slot + shared-slot coefficients) / classifier* (incl. classifier_pool_backup used
      for eval routing, fft.py:225) / d2_shared_indices (plain tensor attribute, not in state_dict).
    Plain BiLoRA (gate=None) has no shared slot, so indices may be empty; whether this is a D2 run is
    inferred from the presence of d2_shared_coef rather than an extra flag.
    """
    sd = network.state_dict()
    keep = {k: v.detach().cpu().clone() for k, v in sd.items()
            if ("coef" in k) or ("classifier" in k)}
    n_idx = 0
    for name, mod in network.named_modules():
        t = getattr(mod, "d2_shared_indices", None)
        if t is not None and hasattr(t, "detach"):
            keep[f"{name}.d2_shared_indices"] = t.detach().cpu().clone()
            n_idx += 1
    n_coef = sum(1 for k in keep if "coef" in k)
    n_cls = sum(1 for k in keep if "classifier" in k)
    has_d2 = any("d2_shared_coef" in k for k in keep)
    if n_coef == 0 or n_cls == 0:
        raise RuntimeError(
            f"[ckpt] collected an empty shell (coef={n_coef} classifier={n_cls}): "
            "collector out of sync with the host structure; refusing to save a fake checkpoint.")
    if has_d2 and n_idx == 0:
        raise RuntimeError(
            "[ckpt] this run has d2_shared_coef but no d2_shared_indices were found "
            "(attribute renamed?). Shared coefficients without frequency indices cannot rebuild Delta W; refusing to save.")
    return keep


def run(train_loaders, val_loaders, test_loaders, ncls, args, device, threshold=None):
    return run_bilora(train_loaders, val_loaders, test_loaders, ncls, args, device)


METHOD_SPEC = {"name": "bilora", "needs_threshold": False, "needs_selector": False}
