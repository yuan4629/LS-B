# -*- coding: utf-8 -*-
"""Method `bilora_d2` = BiLoRA + a persistent shared adapter with shared/specific layer
allocation (the D2 host).

Registered separately from the plain `bilora` method; the identity key distinguishes the arms
by `bioscore_split_mode` and the related fields (see exp.py).

## How the shared slot fits into the task-slot mechanism of fft.py
In BiLoRA every block's Attention_FFT holds coef_k[t]/coef_v[t] (n_frq=3000 frequency
coefficients per task); for task `task` the forward pass uses dW = sum_{t<=task} ifft2(coef_t)
(old task slots are frozen after training but always take part in the forward pass).
This host adds a task-independent persistent slot to the attention of every shared layer:
  d2_shared_coef_k / d2_shared_coef_v (n_frq, zero-initialised) + d2_shared_indices
  (frequency positions from select_pos(t=n_tasks), seed=777+10*n_tasks, disjoint from the
  streams of task slots 0..9).
Role semantics:
  specific layer = BiLoRA default (train the current task slot, freeze it afterwards, the old
                   slots' dW keep contributing);
  shared layer   = the current task slot is frozen at 0 and only the shared slot is trained;
                   shared-slot lr = task lr x shared_adapter_lr_scale (separate param group);
                   it is never frozen and keeps updating across tasks.
Forward pass (install_d2_shared_forward patch):
  wk = sum_{t<=task} dW_t  +  dW_shared (added exactly once, independent of the task count)
Zero slots sum to exactly zero (ifft2(0)=0), so on shared layers the task slots from the freeze
point on stay in the sum but contribute nothing; task slots trained during warm-up (t<K) keep
their non-zero dW -- that is the knowledge of the all-specific warm-up, and it is not cleared.

## Warm-up semantics of the causal arms
ratio_causal: t in {0..K-1} (K=--split_freeze_task, default 3) is the warm-up = all specific =
vanilla BiLoRA verbatim (not even the optimizer changes). For each warm-up task
bioscore_v2.drive_matrix collects one raw D_raw (10-batch forward pass, RNG saved/seeded/restored,
atlas fingerprint checked) and caches it. At the start of train_function for t=K the K D_raw
matrices are stacked to [K,R,L] and passed to analyze_d2_split.allocate(k_freeze=K); the
resulting allocation creates the shared slots (zero-initialised) and stays frozen until the end.
ratio_oracle: the allocation is computed in __init__ from --d2_alloc_file (npz) and applies from
t=0.
pinned: same procedure as best_approximation (warm-up, rho computed at t=K and stored in
layer_scores), except that at t=K the layers given by --d2_pin_layers are committed instead of the
rho top-k; t=K additionally prints `[d2] pinned_layers=[...] rho_top<k>=[...]` for confirmation.

## Task layout and readout accounting (logging only; no computation changes)
Before the first task trains, `[d2] task_layout ...` is printed: number of tasks / classes per
task / model task slots, read from the actual SiNet and DataManager; a mismatch with args raises.
For each warm-up task of a causal arm, `[d2] warmup_readout task t: batches=<n>/<max> images=<m>`
is printed: when the loader has fewer than max batches, the readout uses the ones it gets (e.g.
9/10 batches and 1142 images for task 1 of ImageNet-R with T=20), and this is recorded.

## Interaction with _train (why train_function is hooked)
BiLoRA._train first sets requires_grad=False on all parameters, re-enables the current slot by
name, builds the optimizer and then calls train_function. The shared slot is therefore frozen
again at every task and must be re-enabled at the start of train_function; and since it has its
own lr, the optimizer and scheduler are rebuilt here (mirroring _train's construction branch;
CosineSchedule records base_lrs per param group, so both groups follow their own cosine).
The "Parameters to be updated" line printed by _train comes before the roles are applied and does
not include the shared slot; the [gate]/[d2] lines of this module are authoritative.
Design choice: the shared slot's Adam moments are reset at task boundaries (vanilla BiLoRA
rebuilds the optimizer per task anyway; what persists are the parameters, not the optimizer state).

All error paths raise instead of warning; violated allocation/role invariants raise RuntimeError.
"""
import time

import numpy as np
import torch

from baselines.bilora_adapter.bilora import run_bilora
from baselines.bilora_adapter.bioscore_gate import _scoring_backbone, _StripIdxLoader
from baselines.bilora_adapter.d2_split import (
    CAUSAL_MODES,
    D2_SPLIT_MODES,
    NUM_LAYERS,
    backbone_tag_from_args,
    causal_allocation,
    oracle_allocation,
    parse_pin_layers,
    persistent_param_groups,
    static_specific_layers,
    validate_n_specific,
)

_DS_TOKEN = {"cifar100": "c100", "imagenet_r": "inr", "cub": "cub"}


# ---------------------------------------------------------------------------
# Forward pass: the shared dW always takes part, added exactly once
# ---------------------------------------------------------------------------
def shared_delta_w(attn, coef, alpha=300):
    """dW of the shared slot. Structurally identical to get_delta_w_k in fft.py (same
    alpha=300, same ifft2 path); only the frequency positions (d2_shared_indices) and the
    coefficients (the persistent shared coef) differ."""
    dev = attn.qkv.weight.device
    Fm = torch.zeros(attn.dim, attn.dim, device=dev)
    idx = attn.d2_shared_indices
    Fm[idx[0, :], idx[1, :]] = coef
    return torch.fft.ifft2(Fm, dim=(-2, -1)).real * alpha


def effective_delta_weights(attn, task):
    """(wk, wv): sum_{t<=task} of the task-slot dW + the shared dW exactly once (if a shared
    slot exists).

    The task-slot sum keeps vanilla semantics: slots trained during warm-up carry the t<K
    knowledge; untrained slots are exactly zero (ifft2(0)=0), so there is no "skip" branch --
    one less place for silent degradation.
    """
    if task < 0:
        raise ValueError(f"effective_delta_weights: task={task} is invalid (dW must not be computed without a task)")
    wk = torch.stack([attn.get_delta_w_k(t) for t in range(task + 1)], dim=0).sum(dim=0)
    wv = torch.stack([attn.get_delta_w_v(t) for t in range(task + 1)], dim=0).sum(dim=0)
    if getattr(attn, "_d2_shared_on", False):
        wk = wk + shared_delta_w(attn, attn.d2_shared_coef_k)
        wv = wv + shared_delta_w(attn, attn.d2_shared_coef_v)
    return wk, wv


def install_d2_shared_forward():
    """Install the Attention_FFT forward in which the shared-slot dW always takes part.
    Idempotent; modules without _d2_shared_on (specific layers / non-D2 runs) use the original
    forward. Can be stacked with the skip-inactive patch of bioscore_gate: each patch is gated
    by its own instance attribute and falls back to the other's orig."""
    from models.fft import Attention_FFT

    if getattr(Attention_FFT, "_d2_shared_patched", False):
        return
    orig_forward = Attention_FFT.forward

    # Adapted from Attention_FFT.forward in BiLoRA (models/fft.py, github.com/yifeiacc/BiLoRA @ 78ff950).
    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False):
        if not getattr(self, "_d2_shared_on", False) or getattr(self, "MoE", False):
            return orig_forward(self, x, task, register_hook, get_feat, get_cur_feat)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        wk, wv = effective_delta_weights(self, task)
        k = k + torch.nn.functional.linear(x, wk).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v + torch.nn.functional.linear(x, wv).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))

    Attention_FFT.forward = forward
    Attention_FFT._d2_shared_patched = True
    print("[d2] Attention_FFT.forward patched with the shared-slot forward (shared dW always added, exactly once).")


# ---------------------------------------------------------------------------
# Roles / slots / optimizer. Only the fft.py contract image_encoder.blocks[l].attn is
# assumed about net, so a minimal mock can unit-test this on CPU (see tests/test_d2.py [4]).
# ---------------------------------------------------------------------------
def create_shared_slots(net, shared_layers, num_layers=NUM_LAYERS):
    """Create the persistent shared slots (zero-initialised) on the attention of the shared
    layers. Returns all shared parameters (k, v x layers). May only be called once -- a second
    call means the allocation state machine is broken, so it raises."""
    shared = set(int(i) for i in shared_layers)
    params = []
    blocks = net.image_encoder.blocks
    for lid in sorted(shared):
        attn = blocks[lid].attn
        if getattr(attn, "_d2_shared_on", False):
            raise RuntimeError(f"shared slot of layer {lid} created twice: the allocation must not change once frozen.")
        dev = attn.qkv.weight.device
        n_frq = int(attn.n_frq)
        # Frequency positions from select_pos(t=n_tasks): a deterministic stream with
        # seed=777+10*n_tasks, disjoint from task slots 0..n_tasks-1; constant across seeds and
        # runs (an architectural constant, like BiLoRA's task slots).
        attn.d2_shared_indices = attn.select_pos(len(attn.coef_k), attn.dim).to(dev)
        attn.d2_shared_coef_k = torch.nn.Parameter(torch.zeros(n_frq, device=dev))
        attn.d2_shared_coef_v = torch.nn.Parameter(torch.zeros(n_frq, device=dev))
        attn._d2_shared_on = True
        params += [attn.d2_shared_coef_k, attn.d2_shared_coef_v]
    return params


def apply_d2_roles(net, task, specific, num_layers=NUM_LAYERS):
    """Set requires_grad by role: specific layers keep the current task slot enabled by _train;
    shared layers freeze the current task slot back at 0 and re-enable the shared slot (_train
    freezes it at every task). Two invariants raise immediately: a specific layer must not have
    a shared slot / a shared layer must have one."""
    spec = set(int(i) for i in specific)
    blocks = net.image_encoder.blocks
    for lid in range(int(num_layers)):
        attn = blocks[lid].attn
        if lid in spec:
            if getattr(attn, "_d2_shared_on", False):
                raise RuntimeError(f"layer {lid} is specific but has a shared slot: the allocation drifted after freezing.")
            continue
        if not getattr(attn, "_d2_shared_on", False):
            raise RuntimeError(f"layer {lid} is shared but has no shared slot: create_shared_slots did not reach it.")
        attn.coef_k[task].requires_grad_(False)
        attn.coef_v[task].requires_grad_(False)
        attn.d2_shared_coef_k.requires_grad_(True)
        attn.d2_shared_coef_v.requires_grad_(True)


def build_split_optimizer(optim_name, base_params, shared_params, lr, weight_decay, lr_scale):
    """Two param groups: base (current task slot + classifier head) at the task lr; shared at
    task lr x lr_scale. Shared parameters appear only in the shared group (a parameter in both
    groups would be updated twice per Adam step)."""
    base_params, shared_params = list(base_params), list(shared_params)
    ids = {id(p) for p in shared_params}
    if any(id(p) in ids for p in base_params):
        raise RuntimeError("shared parameters leaked into the base param group; they would be updated twice per step.")
    groups = [{"params": base_params},
              {"params": shared_params, "lr": float(lr) * float(lr_scale)}]
    if optim_name == "adam":
        return torch.optim.Adam(groups, lr=float(lr), weight_decay=float(weight_decay), betas=(0.9, 0.999))
    if optim_name == "sgd":
        return torch.optim.SGD(groups, lr=float(lr), momentum=0.9, weight_decay=float(weight_decay))
    raise ValueError(f"unknown optimizer {optim_name!r} (BiLoRA configs only use adam/sgd); refusing to guess.")


def _unwrap(network):
    return getattr(network, "module", network)


# ---------------------------------------------------------------------------
# RNG isolation: all random draws of the allocation estimate are invisible to training
# (a layer selector leaking global RNG state has been observed to perturb training results).
# ---------------------------------------------------------------------------
class _rng_isolated:
    """Inside the with-block use a deterministic seed; on exit restore the global RNG
    (torch/np/python/cuda). collect_batches has the same guard; this one additionally covers
    building the scoring backbone (weight initialisation draws from the global RNG; this host
    builds it once, fully isolated)."""

    def __init__(self, seed):
        self.seed = int(seed) * 1000003 + 13

    def __enter__(self):
        import random
        self._st = (torch.get_rng_state(), np.random.get_state(), random.getstate(),
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed % (2 ** 31 - 1))
        random.seed(self.seed)

    def __exit__(self, *exc):
        import random
        torch.set_rng_state(self._st[0])
        np.random.set_state(self._st[1])
        random.setstate(self._st[2])
        if self._st[3] is not None:
            torch.cuda.set_rng_state_all(self._st[3])
        return False


# ---------------------------------------------------------------------------
# Gate (allocation policy of one run + confirmation log lines)
# ---------------------------------------------------------------------------
class D2SharedAdapterGate:
    """Gate protocol of run_bilora + make_learner_cls factory (see bilora.py).
    All allocation decisions are made here and printed as fixed-format confirmation lines
    that can be re-checked independently."""

    method_label = "bilora_d2"

    def __init__(self, args, device, num_layers=NUM_LAYERS):
        self.args = args
        self.device = device
        self.num_layers = int(num_layers)
        self.mode = str(getattr(args, "bioscore_split_mode", "none") or "none")
        if self.mode == "none":
            raise ValueError("methods=bilora_d2 needs an explicit --bioscore_split_mode (none is not an arm).")
        if self.mode not in D2_SPLIT_MODES:
            raise ValueError(f"bioscore_split_mode={self.mode!r} is not mapped; valid: {sorted(D2_SPLIT_MODES)}.")
        self.threshold = None            # run_bilora/exp.py use (name, None) as the done key
        self.k_specific = validate_n_specific(getattr(args, "n_specific_layers", 4), self.num_layers)
        self.k_freeze = int(getattr(args, "split_freeze_task", 3))
        self.lr_scale = float(getattr(args, "shared_adapter_lr_scale", 0.1))
        if self.lr_scale < 0:
            raise ValueError(f"shared_adapter_lr_scale={self.lr_scale} is invalid (<0).")

        # Pinned layers of the pinned arm. Both directions fail fast at construction:
        #   - another arm receiving it = the caller thinks layers are pinned while the
        #     allocation is unchanged (same rule as d2_alloc_seed);
        #   - pinned without it = unknown which layers to pin; no default is guessed.
        _pin = getattr(args, "d2_pin_layers", None)
        self._pin = None
        if self.mode == "pinned":
            if _pin is None:
                raise ValueError("bioscore_split_mode=pinned needs --d2_pin_layers (canonical form e.g. 8,9,10,11); no default is guessed.")
            self._pin = parse_pin_layers(_pin, self.k_specific, self.num_layers)
        elif _pin is not None:
            raise ValueError(f"--d2_pin_layers only applies to bioscore_split_mode=pinned, got mode={self.mode!r}: "
                             "passing it to another arm suggests pinned layers while the allocation is unchanged.")

        # Allocation (static/oracle: set up here and fail fast; causal: produced at t=K)
        self._spec = None                # frozen specific layers (sorted list)
        self._shared = None
        self._alloc_sig = "none"         # atlas_sig field of the [gate] line
        self._rho = None
        if self.mode in CAUSAL_MODES:
            if self.k_freeze < 1:
                raise ValueError(f"split_freeze_task={self.k_freeze} is invalid (causal arms need at least 1 warm-up task).")
            if int(getattr(args, "num_tasks", 0)) <= self.k_freeze:
                raise ValueError(
                    f"num_tasks={getattr(args, 'num_tasks', 0)} <= split_freeze_task={self.k_freeze}: "
                    "the warm-up would cover every task and the allocation would never apply (the run would just be all_specific).")
        elif self.mode == "ratio_oracle":
            f = str(getattr(args, "d2_alloc_file", "") or "")
            if not f:
                raise ValueError("ratio_oracle needs --d2_alloc_file (diag npz); no default path is guessed.")
            self._spec, self._shared, self._alloc_sig = oracle_allocation(
                f, backbone_tag_from_args(args), self.k_specific, self.num_layers)
        else:
            _as = getattr(args, "d2_alloc_seed", None)
            spec = static_specific_layers(self.mode, int(getattr(args, "seed", 0)),
                                          self.k_specific, self.num_layers,
                                          alloc_seed=(None if _as is None else int(_as)))
            # static_specific_layers returns None for dynamic arms; the two branches above
            # already caught those, so this must be a list.
            if spec is None:
                raise RuntimeError(f"mode dispatch is broken: {self.mode} reached the static branch.")
            self._spec = list(spec)
            self._shared = sorted(set(range(self.num_layers)) - set(spec))

        # Records (run_bilora contract + curves for analysis)
        self.selected_layers = []        # per task: specific layers (same as the [gate] line)
        self.shared_layers = []          # per task: shared layers
        self.layer_scores = []           # per task: causal=rho; otherwise a 0/1 indicator
        self.select_sec = []
        self.shared_delta_norms = []     # per task: float, or None (task without a shared slot)
        self._shared_params = []
        self._d_cache = []               # raw D_raw of the causal warm-up tasks
        self._calc_obj = None
        self._expected_task = 0

        # Persistent adapter parameter groups (capacity measure), computed at construction time
        # rather than in extra_result_fields: the latter only runs at the very end, so a missing
        # num_tasks would raise after hours of training.
        self.num_tasks = int(getattr(args, "num_tasks", 0) or 0)
        if self.num_tasks < 1:
            raise ValueError(
                f"num_tasks={getattr(args, 'num_tasks', None)!r}: cannot compute the number of persistent parameter groups. "
                "No default: a wrong capacity measure would shift the whole x-axis of the Pareto plot.")
        self.persistent_groups = persistent_param_groups(
            self.mode, self.k_effective(), self.k_freeze, self.num_tasks, self.num_layers)

    # ---- run_bilora protocol ------------------------------------------------
    def install(self):
        """Install the shared-slot forward patch. Requires the BiLoRA repo on sys.path
        (run_bilora calls this after _load_bilora)."""
        install_d2_shared_forward()

    def make_learner_cls(self, BiLoRA):
        gate = self

        class _D2BiLoRA(BiLoRA):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.gate = gate
                self.train_sec = []

            def incremental_train(self, data_manager):
                # Before the first task trains (_cur_task is still BaseLearner's initial -1): read
                # the number of tasks / classes per task / model task slots from the actual model
                # and DataManager, compare them with args and print one line; a mismatch raises.
                # Read-only attribute access, no RNG use, so the training trajectory is unchanged.
                if self._cur_task == -1:
                    gate.confirm_task_layout(self, data_manager)
                super().incremental_train(data_manager)

            def train_function(self, train_loader, test_loader, optimizer, scheduler):
                optimizer, scheduler = gate.on_task_start(self, train_loader, optimizer, scheduler)
                pre = gate.shared_snapshot()
                t0 = time.time()
                super().train_function(train_loader, test_loader, optimizer, scheduler)
                self.train_sec.append(time.time() - t0)
                gate.on_task_end(self, pre)

        return _D2BiLoRA

    def extra_result_fields(self):
        return {
            "d2_split_mode": self.mode,
            "d2_k_specific": self.k_effective(),
            "d2_split_freeze_task": self.k_freeze if self.mode in CAUSAL_MODES else None,
            # Capacity measure, stored in the results (not only logged) so analysis scripts
            # need not re-derive k/K from the arm name.
            "d2_persistent_param_groups": self.persistent_groups,
            "d2_shared_layers_curve": [list(s) for s in self.shared_layers],
            "d2_shared_delta_norm_curve": self.shared_delta_norms,
            "d2_alloc_sig": self._alloc_sig,
        }

    def k_effective(self):
        """Effective number of specific layers (all_shared=0 / all_specific=12 / otherwise
        n_specific_layers)."""
        if self._spec is not None:
            return len(self._spec)
        return self.k_specific           # causal arms before the freeze: the allocation will have k_specific layers

    def confirm_task_layout(self, learner, data_manager):
        """Confirm the task layout before the first task trains, reading from the actual objects:
          - number of tasks: DataManager.nb_tasks (data side, derived from the class order and
            init_cls/increment);
          - classes per task: DataManager.get_task_size(t) per task, and the classifier heads'
            out_features (model side);
          - model task slots: the lengths of coef_k / coef_v / indices of every block's
            Attention_FFT, and the number of classifier heads.
        Prints one line first, then compares with args.num_tasks / args.classes_per_task and
        raises on mismatch. The slot count is not copied from args: whether the host really
        passed total_sessions into SiNet is exactly what is being confirmed."""
        net = _unwrap(learner._network)
        T = int(self.num_tasks)
        cpt = int(getattr(self.args, "classes_per_task", 0) or 0)
        blocks = list(net.image_encoder.blocks)
        slots = sorted({len(b.attn.coef_k) for b in blocks} | {len(b.attn.coef_v) for b in blocks}
                       | {len(b.attn.indices) for b in blocks})
        heads = sorted({len(net.classifier_pool),
                        len(getattr(net, "classifier_pool_backup", net.classifier_pool))})
        head_cls = sorted({int(h.out_features) for h in net.classifier_pool})
        dm_T = int(data_manager.nb_tasks)
        sizes = sorted({int(data_manager.get_task_size(t)) for t in range(dm_T)})

        def one(s):
            return s[0] if len(s) == 1 else s

        print(task_layout_line(T, cpt, dm_T, one(sizes), one(slots), one(heads), one(head_cls)))
        bad = []
        if len(blocks) != self.num_layers:
            bad.append(f"number of blocks {len(blocks)} != {self.num_layers}")
        if slots != [T]:
            bad.append(f"model task slots {slots} != num_tasks={T}")
        if heads != [T]:
            bad.append(f"classifier heads {heads} != num_tasks={T}")
        if head_cls != [cpt]:
            bad.append(f"classes per head {head_cls} != classes_per_task={cpt}")
        if dm_T != T:
            bad.append(f"DataManager tasks {dm_T} != num_tasks={T}")
        if sizes != [cpt]:
            bad.append(f"DataManager classes per task {sizes} != classes_per_task={cpt}")
        if bad:
            raise RuntimeError("[d2] task layout does not match args (refusing to train): " + "; ".join(bad))

    # ---- per-task hooks -----------------------------------------------------
    def on_task_start(self, learner, train_loader, optimizer, scheduler):
        t0 = time.time()
        task = int(learner._cur_task)
        if task != self._expected_task:
            raise RuntimeError(f"task order broken: expected task {self._expected_task}, got {task}.")
        self._expected_task += 1
        net = _unwrap(learner._network)
        warmup = self.mode in CAUSAL_MODES and task < self.k_freeze

        if warmup:
            self._collect_warmup_d(train_loader)             # cache D_raw + atlas_sig
            spec = list(range(self.num_layers))
            shared = []
            scores = [0.0] * self.num_layers
            # warm-up = vanilla BiLoRA verbatim: requires_grad and the optimizer are left untouched.
        else:
            if self._spec is None:
                # only possible for a causal arm (ratio_causal / best_approximation / pinned) reaching the freeze point
                if task != self.k_freeze:
                    raise RuntimeError(f"the causal allocation should be produced at task {self.k_freeze}, got task {task}.")
                spec_l, shared_l, rho = causal_allocation(
                    np.stack(self._d_cache), self.k_specific, self.k_freeze, self.num_layers)
                if self._pin is not None:
                    # pinned: rho is still computed and stored in layer_scores; only the committed
                    # layers are replaced by the pinned ones. One line prints both the pinned layers
                    # and the rho top-k (pinning is only visible when the two differ).
                    print(f"[d2] pinned_layers={list(self._pin)} rho_top{self.k_specific}={list(spec_l)}")
                    spec_l = list(self._pin)
                    shared_l = sorted(set(range(self.num_layers)) - set(self._pin))
                self._spec, self._shared, self._rho = list(spec_l), list(shared_l), rho
            spec, shared = list(self._spec), list(self._shared)
            if not self._shared_params and shared:
                self._shared_params = create_shared_slots(net, shared, self.num_layers)
            apply_d2_roles(net, task, spec, self.num_layers)
            if self.mode in CAUSAL_MODES and self._rho is not None:
                scores = [float(x) for x in self._rho]
            else:
                scores = [1.0 if l in set(spec) else 0.0 for l in range(self.num_layers)]

        dt = time.time() - t0
        self.selected_layers.append(sorted(spec))
        self.shared_layers.append(sorted(shared))
        self.layer_scores.append(scores)
        self.select_sec.append(dt)
        k_str = str(self.k_freeze) if self.mode in CAUSAL_MODES else "-"
        # Fixed log format (parsed field by field by log checks and tests): do not change.
        print(f"[gate] task {task}: shared={sorted(shared)} specific={sorted(spec)} "
              f"src={self.mode}(K={k_str}, atlas_sig={self._alloc_sig})")

        if self._shared_params:
            opt2, sch2 = self._rebuild_optimizer(learner)
            return opt2, sch2
        return optimizer, scheduler

    def shared_snapshot(self):
        if not self._shared_params:
            return None
        return torch.cat([p.detach().float().reshape(-1).cpu() for p in self._shared_params]).clone()

    def on_task_end(self, learner, pre):
        task = int(learner._cur_task)
        cur = self.shared_snapshot()
        if cur is None:
            label = ("warmup(all-specific)"
                     if self.mode in CAUSAL_MODES and task < self.k_freeze
                     else "none(all-specific)")
            print(f"[d2] shared_coef_delta_norm={label}")
            self.shared_delta_norms.append(None)
            return
        if pre is None:                   # slot created at the start of this task -> baseline = zero init
            pre = torch.zeros_like(cur)
        d = float((cur - pre).norm())
        # Fixed log format: non-zero = the shared slot is really training. 0 with lr_scale>0 means
        # the shared dW is not in the graph or its param group was lost, i.e. a silent
        # degradation to all-specific -- raise immediately.
        print(f"[d2] shared_coef_delta_norm={d:.6e}")
        if d == 0.0 and self.lr_scale > 0:
            raise RuntimeError(
                "the shared slot was not updated during the whole task (delta_norm=0 with lr_scale>0): silent degradation to all-specific.")
        self.shared_delta_norms.append(d)

    # ---- internals ----------------------------------------------------------
    def _rebuild_optimizer(self, learner):
        """Mirror of _train's construction branch, split into two param groups. The scheduler
        must be rebuilt together with the new optimizer (the old scheduler holds the old
        optimizer, so stepping it would do nothing -- another silent degradation)."""
        net = learner._network
        ids = {id(p) for p in self._shared_params}
        base = [p for p in net.parameters() if p.requires_grad and id(p) not in ids]
        first = int(learner._cur_task) == 0
        lr = learner.init_lr if first else learner.lrate
        wd = learner.init_weight_decay if first else learner.weight_decay
        opt = build_split_optimizer(learner.optim, base, self._shared_params, lr, wd, self.lr_scale)
        if learner.optim == "adam":
            from utils.schedulers import CosineSchedule
            sch = CosineSchedule(optimizer=opt, K=learner.run_epoch)
        else:
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=opt, T_max=learner.run_epoch)
        print(f"[d2] optimizer rebuilt: base={len(base)} params @lr={lr:g} | "
              f"shared={len(self._shared_params)} params @lr={lr * self.lr_scale:g} "
              f"(scale={self.lr_scale})")
        return opt, sch

    def _calc(self):
        """Scoring calculator (memoized). The backbone is built inside RNG isolation: weight
        initialisation draws global random numbers, and without isolation the causal arms and
        the other arms would diverge in training because of different RNG consumption."""
        if self._calc_obj is None:
            from model_m.common.bioscore_v2 import get_bioscore_v2
            with _rng_isolated(getattr(self.args, "seed", 0)):
                bb = _scoring_backbone(self.args, self.device)
                self._calc_obj = get_bioscore_v2(self.args, self.device, bb)
            if len(self._calc_obj.layers) != self.num_layers:
                raise RuntimeError(
                    f"scoring backbone has {len(self._calc_obj.layers)} layers != BiLoRA's {self.num_layers}; "
                    "the allocation cannot be transferred.")
            from baselines.bilora_adapter.atlas_v2_export import atlas_signature
            self._alloc_sig = atlas_signature(self._calc_obj)["sha1"]
        return self._calc_obj

    def _collect_warmup_d(self, train_loader):
        """D_raw of a warm-up task (uncentred, same convention as *_D_full in the diag npz).
        drive_matrix -> collect_batches saves/seeds/restores the RNG and checks the atlas
        fingerprint."""
        calc = self._calc()
        max_b = int(getattr(self.args, "selector_batches", 10) or 10)
        _dt, info = calc.drive_matrix(_StripIdxLoader(train_loader), max_b)
        d_raw = np.asarray(info["D_raw"], dtype=np.float64)
        if d_raw.ndim != 2 or d_raw.shape[1] != self.num_layers:
            raise RuntimeError(f"unexpected D_raw shape {d_raw.shape} (expected [R,{self.num_layers}]).")
        # Record how many batches / images the readout actually used: when the loader has fewer
        # than max_b batches, collect_batches does not raise and uses the ones it gets (e.g. task 1
        # of ImageNet-R with T=20: 1142 images = 9 batches). This is only recorded; the readout
        # semantics are unchanged. Missing counts mean calculator and host are out of sync.
        if info.get("n_batches") is None or info.get("n_images") is None:
            raise RuntimeError("drive_matrix did not report how many batches / images the readout used: calculator and host out of sync?")
        print(warmup_readout_line(len(self._d_cache), int(info["n_batches"]), max_b, int(info["n_images"])))
        self._d_cache.append(d_raw)


# ---------------------------------------------------------------------------
# Formats of the confirmation lines (factored out so that the host output and the
# synthetic logs in tests/test_d2.py come from one source)
# ---------------------------------------------------------------------------
def task_layout_line(num_tasks, classes_per_task, dm_tasks, dm_task_size, model_task_slots,
                     classifier_heads, head_classes):
    """Task-layout line (number of tasks / classes per task / model task slots). Values come
    from the actual model and DataManager (see D2SharedAdapterGate.confirm_task_layout);
    parsed field by field, so do not change the format."""
    return (f"[d2] task_layout num_tasks={num_tasks} classes_per_task={classes_per_task} "
            f"dm_tasks={dm_tasks} dm_task_size={dm_task_size} model_task_slots={model_task_slots} "
            f"classifier_heads={classifier_heads} head_classes={head_classes}")


def warmup_readout_line(task, n_batches, max_batches, n_images):
    """Readout line of a warm-up task: how many batches / images the readout actually used.
    Parsed by log checks; do not change the format."""
    return f"[d2] warmup_readout task {task}: batches={n_batches}/{max_batches} images={n_images}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def startup_lines(gate, dataset):
    """Startup confirmation lines in a fixed format (the first line is parsed by log checks).
    Factored out so that tests/test_d2.py [6] locks the format against the same source."""
    ds_token = _DS_TOKEN.get(str(dataset or "cifar100"))
    if ds_token is None:
        raise ValueError(f"bilora_d2 does not know dataset={dataset!r} (valid: {sorted(_DS_TOKEN)}).")
    return [
        f"[d2] mode={gate.mode} k_specific={gate.k_effective()} dataset={ds_token}",
        f"[d2] shared_adapter_lr_scale={gate.lr_scale} split_freeze_task={gate.k_freeze} "
        f"n_specific_layers={gate.k_specific} alloc_file={getattr(gate.args, 'd2_alloc_file', '') or '-'}",
        # Line 3: the number of persistent parameter groups, written into run.log (the first two
        # lines are unchanged; parsers anchor on line 1). It is the x-axis of the Pareto plot,
        # so it should never have to be recomputed by hand.
        f"[d2] persistent_param_groups={gate.persistent_groups} "
        f"(L={gate.num_layers} T={gate.num_tasks} "
        f"k={gate.k_effective()} K={gate.k_freeze if gate.mode in CAUSAL_MODES else 0})",
    ]


def run(train_loaders, val_loaders, test_loaders, ncls, args, device, threshold=None):
    if threshold is not None:
        raise ValueError("bilora_d2 does not use threshold (the layer allocation is set by --bioscore_split_mode).")
    gate = D2SharedAdapterGate(args, device)
    for line in startup_lines(gate, getattr(args, "dataset", "cifar100")):
        print(line)
    return run_bilora(train_loaders, val_loaders, test_loaders, ncls, args, device, gate=gate)


METHOD_SPEC = {"name": "bilora_d2", "needs_threshold": False, "needs_selector": False}
