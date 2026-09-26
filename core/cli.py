"""Command-line options, presets, seeding and device selection.
build_argparser() returns the parser; apply_preset() fills in preset values for options the
user did not set explicitly. Several options are inherited from earlier code and are unused by
the D2 pipeline; they are kept so that the identity keys of existing runs stay valid."""
import argparse
import random
import numpy as np
import torch


LEGACY_DEFAULTS = {
    "batch_size": 128,
    "num_tasks": 10,
    "classes_per_task": 10,
    "epochs": 5,
    "probe_epochs": 2,
    "lr": 1e-4,
    "task_order": "random",
    "val_split": 0.0,
    "selector_mode": "brainnet",
    "optimizer": "adamw",
    "lr_patience": 0,
    "lr_factor": 2.0,
    "lr_min": 1e-6,
}

WSN_ALIGNED_DEFAULTS = {
    "batch_size": 64,
    "num_tasks": 10,
    "classes_per_task": 10,
    "epochs": 200,
    "probe_epochs": 5,
    "lr": 1e-3,
    "task_order": "chrono",
    "val_split": 0.05,
    "selector_mode": "grad",
    "optimizer": "adamw",
    "lr_patience": 6,
    "lr_factor": 2.0,
    "lr_min": 1e-6,
}

# Strict class-incremental setting, aligned with BiLoRA's cifar100_bilora.json.
TIMM_CIL_DEFAULTS = {
    "dataset": "cifar100",   # explicit dataset, symmetric with timm_cil_imagenet_r
    "batch_size": 128,
    "num_tasks": 10,
    "classes_per_task": 10,
    "task_order": "chrono",
    "val_split": 0.0,
    "epochs": 20,        # later tasks
    "init_epoch": 40,    # first task
    "lr": 5e-4,
    # Per-group learning rates from the BiLoRA paper (following SLCA): head 1e-3 / adapter 1e-5.
    # Inherited options, unused by the D2 pipeline.
    "head_lr": 1e-3,
    "adapter_lr": 1e-5,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "precision": "fp32",
    "lr_schedule": "cosine",
    "selector_mode": "grad",  # inherited option; the D2 pipeline pins it to grad (see run_d2.bash)
}

# ImageNet-R strict class-incremental: 200 classes / 10 tasks = 20 classes per task. Apart from
# dataset/split, everything is inherited from the CIFAR version (fp32/cosine/head_lr/adapter_lr/
# epochs), so both datasets use the same hyperparameters.
TIMM_CIL_IMAGENET_R_DEFAULTS = {
    **TIMM_CIL_DEFAULTS,
    "dataset": "imagenet_r",
    "num_tasks": 10,
    "classes_per_task": 20,  # 200 classes / 10 tasks
}

# CUB-200 strict class-incremental (APER split: 9,430 train / 2,358 test): 200 classes / 10 tasks
# = 20 classes per task. Same timm-aligned hyperparameters as ImageNet-R; the class order follows
# the same procedure as for the other datasets.
TIMM_CIL_CUB_DEFAULTS = {
    **TIMM_CIL_DEFAULTS,
    "dataset": "cub",
    "num_tasks": 10,
    "classes_per_task": 20,  # 200 classes / 10 tasks
}

# Datasets with a fixed number of classes: apply_preset requires num_tasks x classes_per_task to
# equal the total. CIFAR-100 is not listed: smoke runs with NUM_TASKS=2 only train the first tasks.
_STRICT_TOTAL_CLASSES = {"imagenet_r": 200, "cub": 200}

# argparse defaults (apply_preset uses them to detect options the user set explicitly).
_ARGPARSE_DEFAULTS = {
    **LEGACY_DEFAULTS,
    "dataset": "cifar100",  # matches the --dataset default; the preset applies only if not passed explicitly
    "weight_decay": 0.0,
    "precision": "fp32",
    "init_epoch": 0,
    "lr_schedule": "constant",
    "head_lr": None,
    "adapter_lr": None,
    "selector_mode": None,  # sentinel matching default=None in build_argparser, so the preset applies only if not passed explicitly
    # Sentinels: with an argparse default of 10, num_tasks / classes_per_task could not be told
    # apart from an explicit 10, so `--preset timm_cil_imagenet_r --classes_per_task 10` would be
    # silently reset to 20. With None, explicit values always win; unset values come from the
    # preset, or from LEGACY_DEFAULTS in apply_preset when the preset has none.
    "num_tasks": None,
    "classes_per_task": None,
}


def build_argparser():
    p = argparse.ArgumentParser("General continual-learning test script")
    p.add_argument("--preset", choices=["legacy", "wsn_cifar100_split", "timm_cil_cifar100", "timm_cil_imagenet_r", "timm_cil_cub"], default="timm_cil_cifar100")
    # Dataset selection: core.data.build_split dispatches on it.
    p.add_argument("--dataset", choices=["cifar100", "imagenet_r", "cub"], default="cifar100")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--fmri_data_dir", default="./subj01")
    p.add_argument("--wsn_dir", default="./WSN")
    p.add_argument("--output_dir", default="./outputs/lifelong_general")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=LEGACY_DEFAULTS["batch_size"])
    p.add_argument("--num_workers", type=int, default=4)
    # None sentinel (see _ARGPARSE_DEFAULTS): unset -> preset value, or LEGACY_DEFAULTS if the
    # preset has none; explicit values are never overridden by the preset.
    p.add_argument("--num_tasks", type=int, default=None)
    p.add_argument("--classes_per_task", type=int, default=None)
    p.add_argument("--task_order", choices=["random", "chrono"], default=LEGACY_DEFAULTS["task_order"])
    p.add_argument("--val_split", type=float, default=LEGACY_DEFAULTS["val_split"])
    p.add_argument("--epochs", type=int, default=LEGACY_DEFAULTS["epochs"])
    # Epochs of the first task (usually trained longer). 0 = fall back to --epochs.
    p.add_argument("--init_epoch", type=int, default=0)
    p.add_argument("--probe_epochs", type=int, default=LEGACY_DEFAULTS["probe_epochs"])
    p.add_argument("--lr", type=float, default=LEGACY_DEFAULTS["lr"])
    # Per-group learning rates (inherited options, unused by the D2 pipeline): None = use --lr.
    p.add_argument("--head_lr", type=float, default=None)
    p.add_argument("--adapter_lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default=LEGACY_DEFAULTS["optimizer"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lr_min", type=float, default=LEGACY_DEFAULTS["lr_min"])
    p.add_argument("--lr_patience", type=int, default=LEGACY_DEFAULTS["lr_patience"])
    p.add_argument("--lr_factor", type=float, default=LEGACY_DEFAULTS["lr_factor"])
    p.add_argument("--precision", choices=["amp", "fp32"], default="fp32")
    # LR schedule: cosine reproduces BiLoRA's CosineSchedule (see core/train.py); constant = no schedule.
    p.add_argument("--lr_schedule", choices=["cosine", "constant"], default="constant")
    # Backbone family: None sentinel, resolved by apply_preset (timm_cil presets -> timm, otherwise clip).
    # Inherited option, unused by the D2 pipeline.
    p.add_argument("--backbone", choices=["clip", "timm"], default=None)
    # timm backbone name + offline weights. If the timm_weights file exists it is loaded on top;
    # otherwise the regular timm pretrained weights are used.
    p.add_argument("--timm_model", default="vit_base_patch16_224.augreg_in21k")
    p.add_argument("--timm_weights", default="./pretrained/vit_b16_augreg_in21k.npz")
    # Weight format: augreg_npz = JAX npz overlay; dino_pth/ibot_pth = self-supervised .pth with key
    # remapping (see timm_backbone.build_timm_vit).
    p.add_argument("--timm_weights_format", choices=["augreg_npz", "dino_pth", "ibot_pth"], default="augreg_npz")
    # Routing of legacy methods (inherited option, unused by the D2 pipeline).
    p.add_argument("--route", choices=["ncm", "merge_all"], default="merge_all")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["bilora_d2"],
        # bilora = upstream BiLoRA; bilora_d2 = shared/specific allocation host
        # (the allocation arm is chosen with --bioscore_split_mode).
        choices=["bilora", "bilora_d2"],
    )
    # Top-k ratio of legacy layer-selection methods: k=int(12*th) (inherited option).
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    # Layer-selection mode of legacy methods (inherited option). None is a sentinel so that an
    # explicit value is never overridden by a preset; when unset, apply_preset falls back to
    # LEGACY_DEFAULTS["selector_mode"]. It is part of the identity key, so the D2 pipeline pins it
    # to grad.
    p.add_argument("--selector_mode",
                   choices=["brainnet", "brainnet_v2", "drift", "grad", "random", "random_depth",
                            "first", "last", "explicit"], default=None)
    # Comma-separated layer indices (e.g. "3" or "0,5,9"), read only with --selector_mode explicit.
    p.add_argument("--selector_layers", default="")
    p.add_argument("--selector_batches", type=int, default=10)
    # BioScore v1 scoring map (inherited option): last = last layer only; perlayer = per-layer
    # online profile.
    p.add_argument("--bioscore_score_mode", choices=["last", "perlayer"], default="last")
    # A mock atlas is refused by default; this flag allows it for smoke tests only
    # (see selectors.get_selector_scores).
    p.add_argument("--allow_mock_atlas", action="store_true")
    # --- BioScore v2 options (brain readout, read through get_bioscore_v2) ----
    # Replaces the CL seed in the atlas fingerprint: a brain prior that changed with the CL seed
    # would not be a prior, and every seed would retrain the PLModel (~25 min). Default 0: all
    # seeds share one atlas.
    p.add_argument("--atlas_seed", type=int, default=0)
    # Interference penalty: layers used by semantically similar earlier tasks are not reused
    # (makes selection depend on the task order). 0 = off.
    p.add_argument("--bioscore_gamma", type=float, default=0.0)
    # Weight of the layer main effect: >0 explicitly prefers some depth. Default 0: any depth
    # preference must be a reported choice, not a hidden confound.
    p.add_argument("--bioscore_lambda_depth", type=float, default=0.0)
    # ROI weights: uniform = all brain regions weighted equally (robust default); engage = weighted
    # by this task's ROI engagement (ablation).
    p.add_argument("--bioscore_roi_weight", choices=["uniform", "engage"], default="uniform")
    # Third centring axis (tasks): running = subtract the running mean of the previous tasks' D~
    # (causal, valid in CL); none = ablation.
    # Why running is the default: the reference pool for z is NSD while the tasks come from a
    # different image domain, so the domain shift leaves an ROI x layer component shared by all
    # tasks in D~ that is much larger than the task-specific one; without removing it, almost every
    # task selects the same layers. none reproduces that failure.
    p.add_argument("--bioscore_task_center", choices=["running", "none"], default="running")
    # Number of most reliable voxels (highest validation R^2) kept per ROI; needed for memory and
    # also reduces noise.
    p.add_argument("--bioscore_voxel_topq", type=int, default=500)
    p.add_argument("--bioscore_voxel_chunk", type=int, default=2048)
    # Number of reference sub-pools: the NSD training images are split into pools of the task-side
    # size to estimate the sampling distribution of s[n,l].
    p.add_argument("--bioscore_ref_pools", type=int, default=10)
    # Cheap proxy protocol: scales BiLoRA's init_epoch/epochs proportionally; 1.0 = full.
    # Used instead of --max_train_steps, which cuts epochs to 2 and truncates the data (too
    # degraded to preserve rankings). Part of the exp.py identity key, so proxy runs are never
    # mixed up with full runs.
    p.add_argument("--epoch_scale", type=float, default=1.0)
    p.add_argument("--brainnet_epochs", type=int, default=5)
    p.add_argument("--brainnet_limit_train_batches", type=float, default=1.0)
    p.add_argument("--brainnet_limit_val_batches", type=float, default=1.0)
    p.add_argument("--brainnet_cached", action="store_true")
    # Force an atlas rebuild: ignore atlas.pt/plmodel.ckpt on disk, retrain the PLModel once in this
    # process and overwrite the cache (reused across thresholds/tasks), so a stale atlas is never reused.
    p.add_argument("--brainnet_force_rebuild", action="store_true")
    p.add_argument("--brainnet_skip_fail", action="store_true")
    p.add_argument("--bioscore_cache_dir", default="./outputs/bioscore_atlas")
    p.add_argument("--ewc_lambda", type=float, default=1000.0)
    p.add_argument("--ewc_gamma", type=float, default=1.0)
    p.add_argument("--ewc_fisher_batches", type=int, default=100)
    p.add_argument("--adapter_rank", type=int, default=32)
    # LoRA rank / scaling of legacy methods (inherited options, unused by the D2 pipeline).
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    # LoRA target modules of legacy methods (inherited option, unused by the D2 pipeline).
    p.add_argument("--lora_targets", choices=["mlp", "mlp_attn"], default="mlp_attn")
    # Subspace-orthogonality penalty of legacy methods (inherited option, unused by the D2
    # pipeline); 0 = off.
    p.add_argument("--ortho_lambda", type=float, default=0.0)
    # Penalty form for --ortho_lambda (inherited option): raw = ||dWt dWs^T||^2; cosine = normalised
    # by ||dWt||^2 ||dWs||^2 (scale-free).
    p.add_argument("--ortho_mode", choices=["raw", "cosine"], default="raw")
    # Inherited option, unused by the D2 pipeline.
    p.add_argument("--joint_epochs", type=int, default=100)
    p.add_argument("--min_selected_layers", type=int, default=2)
    # --- BiLoRA options (bilora / bilora_d2) ------------------------------------
    # BiLoRA backbone weights: empty = look for ./pretrained/vit_b16_augreg_in21k.npz (the original
    # backbone), then fall back to the HF hub; a .pth selects iBOT/DINO self-supervised weights.
    # See baselines/bilora_adapter/timm_compat.py.
    p.add_argument("--bilora_weights", default="")
    # Blocks whose dW is exactly 0 skip the adapter forward (numerically exact).
    # --no_bilora_skip_inactive turns this off, to verify that ACC is bit-identical with and without
    # skipping (skipping should only affect wall-clock time).
    p.add_argument("--bilora_skip_inactive", dest="bilora_skip_inactive", action="store_true", default=True)
    p.add_argument("--no_bilora_skip_inactive", dest="bilora_skip_inactive", action="store_false")
    # --- D2 shared adapter (methods=bilora_d2) ----------------------------------
    # The arm vocabulary must match baselines/bilora_adapter/d2_split.D2_SPLIT_MODES exactly
    # (locked by tests/test_d2.py [2]). none = not a D2 run (bilora_d2 raises on it);
    # ratio_causal = main arm (t<split_freeze_task is an all-specific warm-up, then the allocation
    # is frozen from the first K tasks' D_t); ratio_oracle = analysis upper bound (allocation
    # computed from --d2_alloc_file, active from t=0); fixed_depth = fixed-shallow heuristic;
    # fixed_depth_l = grid sweep (shallow l layers specific, l from --n_specific_layers, the job
    # name must contain l); fixed_depth_deep = deep n layers specific (mirror of fixed_depth).
    p.add_argument("--bioscore_split_mode",
                   choices=["none", "ratio_causal", "ratio_oracle", "all_shared",
                            "all_specific", "random_split", "fixed_depth",
                            "fixed_depth_l", "fixed_depth_deep",
                            # sliding windows with fixed k=4: w1..w7 (w0/w8 are aliases of existing arms)
                            "fixed_window_w1", "fixed_window_w2", "fixed_window_w3",
                            "fixed_window_w4", "fixed_window_w5", "fixed_window_w6",
                            "fixed_window_w7",
                            # exploratory: same procedure as ratio_causal, k swept over 1..8;
                            # the job name must contain k (like l for fixed_depth_l).
                            "best_approximation",
                            # pinned allocation + observation period (same procedure as
                            # best_approximation; at t=K it commits the layers given by
                            # --d2_pin_layers instead of the rho top-k).
                            "pinned"], default="none")
    # Number of specific layers k_specific (main setting 4); also the grid l for fixed_depth_l.
    p.add_argument("--n_specific_layers", type=int, default=4)
    # Shared-slot learning rate = task-specific lr x this factor (separate param group; never frozen).
    p.add_argument("--shared_adapter_lr_scale", type=float, default=0.1)
    # Number of warm-up tasks K of the causal arms: t in [0,K) all specific, the allocation applies
    # from t=K (with K=3 the causal allocation equals the oracle one on AugReg/iBOT, J=1.00).
    p.add_argument("--split_freeze_task", type=int, default=3)
    # Allocation table of ratio_oracle: a diag npz (e.g. outputs/atlas_v2_diag_k4_b10.npz), computed on
    # the fly by analyze_d2_split.allocate (hand-copied tables are refused). Deliberately *not* part
    # of the identity key (paths differ between machines); the allocation itself is logged in the
    # [gate] line together with the npz signature.
    p.add_argument("--d2_alloc_file", default="")
    # Allocation-draw seed of random_split, decoupled from the training seed.
    #   None (default) = draw with the training seed (backward compatible; identities of earlier
    #   runs are unchanged).
    #   A = the allocation depends on A only, the training seed is separate. Two uses:
    #     - calibration: fix A and vary the training seed -> measures the run noise of a fixed
    #       allocation (otherwise fully confounded with the allocation effect);
    #     - m>1 draws per training seed: average E_a[mu(a)] over m values of A (pure variance
    #       reduction).
    # Note: must be in exp.py's _IDENTITY_KEYS. It decides which layers are chosen; otherwise two
    #    runs with the same seed but different draws would be treated as the same unit and skipped.
    p.add_argument("--d2_alloc_seed", type=int, default=None)
    # Pinned layers of the pinned arm: canonical comma-separated string, e.g. "8,9,10,11" (deep) /
    # "0,1,2,3" (shallow).
    #   None (default) = no pinning. Only --bioscore_split_mode pinned accepts it: another arm
    #   receiving it, or pinned without it, raises in the bilora_d2 gate at construction (no
    #   default is guessed and nothing is silently ignored).
    # Note: must be in exp.py's _IDENTITY_KEYS: the deep and shallow runs are otherwise identical.
    #    Older metrics.json files lack the key (None) = argparse default (None), so resume
    #    decisions for existing runs are unaffected.
    # Non-canonical forms ("11,10,9,8" / "8, 9,10,11") are rejected at parse time: the string goes
    # verbatim into the identity key, so one allocation must have a single spelling.
    def _pin_layers_arg(s):
        from baselines.bilora_adapter.d2_split import parse_pin_layers  # the single parsing rule
        try:
            parse_pin_layers(s)
        except ValueError as e:                  # exception type whose message argparse prints as is
            raise argparse.ArgumentTypeError(str(e)) from None
        return s
    p.add_argument("--d2_pin_layers", type=_pin_layers_arg, default=None)
    p.add_argument("--wsn_capacity", type=float, default=0.5)
    p.add_argument("--wsn_optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--wsn_epochs", type=int, default=0)
    p.add_argument("--max_train_steps", type=int, default=0)
    p.add_argument("--max_eval_steps", type=int, default=0)
    return p


def _apply_preset_dict(args, preset_dict, skip_keys=()):
    """Apply a preset value only where the option still equals its argparse default (i.e. the
    user did not set it explicitly)."""
    for key, preset_val in preset_dict.items():
        if key in skip_keys:
            continue
        if getattr(args, key) == _ARGPARSE_DEFAULTS.get(key, preset_val):
            setattr(args, key, preset_val)


def apply_preset(args):
    if args.preset == "wsn_cifar100_split":
        # selector_mode is skipped so that the brainnet default is kept (inherited behaviour).
        _apply_preset_dict(args, WSN_ALIGNED_DEFAULTS, skip_keys=("selector_mode",))
    elif args.preset == "timm_cil_cifar100":
        _apply_preset_dict(args, TIMM_CIL_DEFAULTS)
    elif args.preset == "timm_cil_imagenet_r":
        _apply_preset_dict(args, TIMM_CIL_IMAGENET_R_DEFAULTS)
    elif args.preset == "timm_cil_cub":
        _apply_preset_dict(args, TIMM_CIL_CUB_DEFAULTS)
    # selector_mode sentinel: if neither the preset nor the user set it, fall back to the global
    # default (brainnet).
    if args.selector_mode is None:
        args.selector_mode = LEGACY_DEFAULTS["selector_mode"]
    # num_tasks / classes_per_task sentinels: the legacy preset applies no preset dict, so fall
    # back to LEGACY_DEFAULTS (10/10).
    for key in ("num_tasks", "classes_per_task"):
        if getattr(args, key) is None:
            setattr(args, key, LEGACY_DEFAULTS[key])
    # Datasets with a fixed number of classes: the split must cover all classes exactly. Since
    # explicit values are not overridden by the preset, T=20 without the per-task class count gives
    # 20x20=400 and a 10-class run gives 10x10=100; both raise immediately instead of silently
    # running out of range or on half the dataset. Checked per dataset, not per preset (they are
    # separate knobs).
    total = _STRICT_TOTAL_CLASSES.get(args.dataset)
    if total is not None and args.num_tasks * args.classes_per_task != total:
        raise ValueError(
            f"dataset={args.dataset} has {total} classes, but num_tasks={args.num_tasks} x classes_per_task="
            f"{args.classes_per_task} = {args.num_tasks * args.classes_per_task} != {total}: "
            "the split must cover all classes exactly (T=10 -> 10x20, T=20 -> 20x10).")
    # backbone sentinel: None = not passed -> default from the preset (so an explicit
    # --backbone clip stays distinguishable).
    if args.backbone is None:
        args.backbone = "timm" if args.preset in ("timm_cil_cifar100", "timm_cil_imagenet_r", "timm_cil_cub") else "clip"
    if args.wsn_epochs <= 0:
        args.wsn_epochs = args.epochs
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Same as BiLoRA's _set_random, for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

