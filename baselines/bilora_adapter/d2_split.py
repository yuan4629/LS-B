# -*- coding: utf-8 -*-
"""Pure allocation logic for D2 (brain-grounded shared/specific layer allocation).

Does not import torch, so tests can use it directly. The rho-based allocation itself is
implemented once, in analyze_d2_split.allocate() at the repository root; this module provides:
  - the mode vocabulary (must match the --bioscore_split_mode choices in core/cli.py;
    locked by a test);
  - the specific-layer sets of the static arms (all_shared / all_specific / random_split /
    fixed_depth / fixed_depth_l / fixed_depth_deep / fixed_window_w*);
  - reading an oracle allocation from a diag npz (via analyze_d2_split.allocate).

Unmapped modes always raise: a silent .get() fallback would quietly turn a new arm into an
old one.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

NUM_LAYERS = 12  # the BiLoRA backbone is a fixed ViT-B/16

# Allocation modes. fixed_depth_l = the shallow l layers are specific (l comes from
# --n_specific_layers and is part of the job name).
# "none" = not a D2 run; bilora_d2 raises on it.
D2_SPLIT_MODES = (
    "none",
    "ratio_causal",     # main arm: t<K is warm-up (all specific); at t=K the allocation is computed from the first K tasks' D_t, then frozen
    "ratio_oracle",     # analysis upper bound: allocation computed from --d2_alloc_file (npz), active from t=0
    "all_shared",       # all 12 layers shared (endpoint)
    "all_specific",     # all 12 layers task-specific ~ vanilla BiLoRA (endpoint)
    "random_split",     # k_specific layers drawn uniformly at random; redrawn per seed, fixed across tasks
    "fixed_depth",      # fixed-shallow heuristic: specific=[0..n_specific-1] (default [0,1,2,3])
                        # Note: same experiment as fixed_depth_l4; run_d2.bash only runs the latter
    "fixed_depth_l",    # grid sweep: shallow l layers specific, l = --n_specific_layers
    "fixed_depth_deep",     # deep n layers specific: specific=[num_layers-n .. num_layers-1].
                            # n=4 -> [8,9,10,11] (the DualPrompt-style "share the shallow layers"
                            # heuristic; on AugReg/iBOT it also equals the alpha/ESD top-4
                            # criterion, but not on DINO, whose alpha_hill top-4 is [7,8,9,10]);
                            # n=10 -> [2..11] (k ablation).
    # Sliding-window profile with fixed k=4: fixed_window_w<W> -> specific=[W, W+1, W+2, W+3].
    # Only w1..w7 exist: w0=[0,1,2,3] is fixed_depth_l4 and w8=[8,9,10,11] is
    # fixed_depth_deep(n=4); the two endpoints reuse those arms instead of new run directories.
    "fixed_window_w1", "fixed_window_w2", "fixed_window_w3", "fixed_window_w4",
    "fixed_window_w5", "fixed_window_w6", "fixed_window_w7",
    "best_approximation",   # capacity-accuracy Pareto ablation (exploratory): same procedure as
                            # ratio_causal (K all-specific warm-up tasks -> allocation from the
                            # first K tasks' D_t -> frozen), except that k=--n_specific_layers is
                            # swept over 1..8 and appears in the job name. A separate mode name
                            # keeps these runs out of the ratio_causal result set and makes the
                            # arm visible in every directory name and [gate] line.
                            # With k=4 it must reproduce ratio_causal exactly (acceptance test).
    "pinned",               # pinned allocation with an observation period (exploratory): same
                            # procedure as best_approximation (K all-specific warm-up tasks; rho is
                            # still computed at t=K and logged in layer_scores), except that the k
                            # layers committed at t=K come from --d2_pin_layers (deep=[8,9,10,11] /
                            # shallow=[0,1,2,3]) instead of the rho top-k.
                            # The pinned layers are part of the exp.py identity key; job name
                            # d2_pinned_k<k>f<K>_<deep|shallow>_<ds>[_bk]_s<seed>.
)

# Modes whose allocation only exists at run time (static_specific_layers returns None).
# `pinned` belongs here too: its committed layers are constants, but t<K is an all-12-layer
# warm-up, so there is no single per-task specific set to recompute statically.
DYNAMIC_MODES = ("ratio_causal", "ratio_oracle", "best_approximation", "pinned")
# Modes that follow "warm up K tasks (all specific) -> allocate from the first K tasks' D_t
# -> freeze". bilora_d2.py branches on this tuple, so a new arm of this kind is added here
# only. (Scattered `mode == "ratio_causal"` checks would let a new arm silently degrade into
# a static arm without warm-up while its [gate] lines still look normal.)
CAUSAL_MODES = ("ratio_causal", "best_approximation", "pinned")

def _repo_root():
    return Path(__file__).resolve().parents[2]


def _import_allocate():
    """Import analyze_d2_split from the repository root (the single implementation of the
    allocation algorithm)."""
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    import analyze_d2_split
    return analyze_d2_split


def persistent_param_groups(mode, k_specific, k_freeze, num_tasks, num_layers=NUM_LAYERS):
    """Number of persistent adapter parameter groups (the capacity measure).

    Slot rules (bilora_d2 role semantics):
      - task 0: one task slot for each of the k specific layers + one shared slot for each
        of the (L-k) shared layers = L;
      - task t>=1: one new task slot per specific layer = k;
      - during warm-up (causal arms, t<K) all 12 layers are specific, so each warm-up task
        creates L task slots; slots trained during warm-up keep their non-zero dW and are
        never cleared, so they count as persistent parameters.

    =>  static arms (K=0):  L + k(T-1)        = 9k+12   (L=12, T=10)
        causal arms:        L(K+1) + k(T-K-1) = 6k+48   (L=12, T=10, K=3)
    Check values: fixed_depth_l4 -> 48, ratio_causal(k=4,K=3) -> 72, all_specific -> 120,
    all_shared -> 12.
    """
    L, T = int(num_layers), int(num_tasks)
    k = validate_n_specific(k_specific, L)
    if T < 1:
        raise ValueError(f"num_tasks={num_tasks} is invalid (must be >= 1)")
    if mode == "all_shared":
        k = 0
    elif mode == "all_specific":
        k = L
    K = int(k_freeze) if mode in CAUSAL_MODES else 0
    if mode in CAUSAL_MODES and not 1 <= K < T:
        raise ValueError(f"split_freeze_task={k_freeze} out of range for a causal arm (valid: 1..{T - 1})")
    return L * (K + 1) + k * (T - K - 1)


def validate_n_specific(n_specific, num_layers=NUM_LAYERS):
    n = int(n_specific)
    if not (0 <= n <= num_layers):
        raise ValueError(f"n_specific_layers={n_specific} out of range (valid: 0..{num_layers})")
    return n


def parse_pin_layers(spec, k_specific=None, num_layers=NUM_LAYERS):
    """The only parser for --d2_pin_layers (shared by the core/cli parse-time check and the
    bilora_d2 gate).

    Only the canonical form is accepted: comma-separated, no spaces, strictly increasing, no
    duplicates, in range, e.g. "8,9,10,11". Non-canonical forms ("11,10,9,8" / "8, 9,10,11" /
    "8,8,9,10" / "08,9,10,11") raise ValueError instead of being sorted for the caller: the
    string goes verbatim into the exp.py identity key, and two spellings of one allocation
    would make the same run look like two identities (a resume would back up the old
    metrics.json and restart from scratch). If k_specific is given, the number of layers must
    equal k. Returns list[int].
    """
    s = "" if spec is None else str(spec)
    try:
        vals = [int(x) for x in s.split(",")]
    except ValueError:
        raise ValueError(f"--d2_pin_layers={spec!r} is not a comma-separated list of layer indices (canonical form e.g. 8,9,10,11)") from None
    if s != ",".join(str(v) for v in vals) or vals != sorted(set(vals)):
        raise ValueError(f"--d2_pin_layers={spec!r} is not in canonical form (strictly increasing, no duplicates, no spaces/leading zeros; "
                         f"e.g. {','.join(str(v) for v in sorted(set(vals)))})")
    nl = int(num_layers)
    if any(not 0 <= v < nl for v in vals):
        raise ValueError(f"--d2_pin_layers={spec!r} out of range (valid layer indices: 0..{nl - 1})")
    if k_specific is not None and len(vals) != int(k_specific):
        raise ValueError(f"--d2_pin_layers={spec!r} has {len(vals)} layers != n_specific_layers={k_specific}"
                         " (the pinned layers are the committed specific layers, so the counts must agree)")
    return vals


def random_split_layers(seed, n_specific, num_layers=NUM_LAYERS, alloc_seed=None):
    """random_split arm: k_specific layers drawn uniformly at random, redrawn per seed and
    fixed across the tasks of one run (the redraw unit is the seed, not the task).
    The seed-stream constant 60301 is offset from selectors.random_scores(100003+task) so the
    two streams never overlap.

    `alloc_seed` decouples the allocation draw from the training seed:
      - None (default): draw with the training seed (reproduces earlier runs bit for bit).
      - A (int): the allocation depends on A only.
    Why: when the draw and the training seed are tied one-to-one, the run noise of a fixed
    allocation cannot be identified; fixing A while varying the seed measures it directly.
    """
    n = validate_n_specific(n_specific, num_layers)
    s = int(seed) if alloc_seed is None else int(alloc_seed)
    rng = np.random.RandomState((s * 100003 + 60301) % (2 ** 31 - 1))
    pick = rng.choice(int(num_layers), size=n, replace=False) if n else np.array([], dtype=int)
    return sorted(int(x) for x in pick)


def static_specific_layers(mode, seed, n_specific, num_layers=NUM_LAYERS, alloc_seed=None,
                           alloc_file=None, backbone_tag=None):
    """Specific-layer set (sorted list) of a static arm; used to recompute and cross-check the
    allocation logged in run.log.

    Returns:
      - static arms -> sorted list[int]
      - dynamic arms (DYNAMIC_MODES) -> None (the allocation is produced by the host / oracle table)
      - none / unknown -> ValueError (no .get fallback)
    """
    # Note: this check must come before every return. Only random_split uses the draw seed;
    # passing it to another arm means the caller believes the allocation changed while nothing
    # happened (and the job name still carries the `a<seed>` suffix). An earlier version placed
    # it after the random_split branch, so all_shared/all_specific returned before reaching it.
    if alloc_seed is not None and mode != "random_split":
        raise ValueError(
            f"--d2_alloc_seed only applies to random_split, got mode={mode!r}. "
            "It only affects the random draw; passing it to another arm suggests the allocation changed when it did not.")
    if mode in DYNAMIC_MODES:
        return None
    if mode == "all_shared":
        return []
    if mode == "all_specific":
        return list(range(int(num_layers)))
    if mode == "random_split":
        return random_split_layers(seed, n_specific, num_layers, alloc_seed=alloc_seed)
    if mode in ("fixed_depth", "fixed_depth_l"):
        # fixed_depth = fixed-shallow heuristic (default n_specific=4 -> [0,1,2,3]);
        # fixed_depth_l = grid sweep with the same rule, l from --n_specific_layers (in the job name).
        n = validate_n_specific(n_specific, num_layers)
        return list(range(n))
    if mode == "fixed_depth_deep":
        # Mirror of fixed_depth: specific layers on the deep side, shared ones on the shallow side.
        # This is the side the field actually uses (DualPrompt, Table 6: sharing the first 1-2
        # layers is best), whereas the whole grid family sits on the other side.
        n = validate_n_specific(n_specific, num_layers)
        nl = int(num_layers)
        return list(range(nl - n, nl))
    if mode.startswith("fixed_window_w"):
        # Sliding-window arms: fixed k=4, only the position moves (the grid confounds position
        # with capacity). n_specific must therefore be exactly 4; any other value raises instead
        # of silently turning the window into a different capacity point.
        if mode not in D2_SPLIT_MODES:
            raise ValueError(f"{mode!r} is not in D2_SPLIT_MODES (valid: w1..w7)")
        n = validate_n_specific(n_specific, num_layers)
        if n != 4:
            raise ValueError(
                f"fixed_window_* arms require k=4 (got n_specific={n}): a window moves a fixed budget, "
                "so a different k is not the same profile; use fixed_depth_l/fixed_depth_deep to sweep capacity.")
        w = int(mode[len("fixed_window_w"):])
        if not 1 <= w <= int(num_layers) - 4 - 1:
            raise ValueError(f"window start w={w} out of range (valid: 1..{int(num_layers) - 5}; "
                             "w0/w8 are aliases of existing arms)")
        return list(range(w, w + 4))
    raise ValueError(
        f"bioscore_split_mode={mode!r} is not mapped (choices: {sorted(D2_SPLIT_MODES)})."
        " If you just added a new arm in core/cli.py, also add it to d2_split.D2_SPLIT_MODES and to this function; "
        "unmapped modes raise instead of silently falling back to another arm.")


def backbone_tag_from_args(args):
    """Infer the backbone name (augreg/ibot/dino) from --bilora_weights, used for the oracle
    npz key and for logging. Raises if inference fails: reading the wrong backbone's oracle
    allocation would silently invalidate the whole arm."""
    bw = str(getattr(args, "bilora_weights", "") or "").lower()
    if bw == "" or "augreg" in bw:
        return "augreg"
    if "ibot" in bw:
        return "ibot"
    if "dino" in bw:
        return "dino"
    raise ValueError(
        f"cannot infer the backbone from --bilora_weights={bw!r} (must contain augreg/ibot/dino, or be empty = augreg). "
        "The oracle allocation is read from the <backbone>_D_full key; guessing the key is worse than failing.")


def oracle_allocation(alloc_file, backbone_tag, k_specific, num_layers=NUM_LAYERS):
    """Compute the oracle allocation from --d2_alloc_file (a diag npz, e.g.
    outputs/atlas_v2_diag_k4_b10.npz).

    Only npz is accepted: the allocation is computed by analyze_d2_split.allocate from the raw
    D_full (k_freeze=None = all tasks) and never copied by hand, so every run is verifiable.
    Returns (specific, shared, sig); sig = first 12 hex chars of the file's sha1, logged in the
    [gate] line.
    """
    p = Path(alloc_file)
    if not p.is_file():
        raise FileNotFoundError(
            f"--d2_alloc_file not found: {alloc_file} (ratio_oracle needs an explicit diag npz; no path is guessed)")
    if p.suffix != ".npz":
        raise ValueError(
            f"--d2_alloc_file only accepts .npz (got {p.suffix!r}): the allocation must be computed by allocate() from D_full; "
            "hand-copied JSON tables are not accepted.")
    npz = np.load(str(p))
    key = f"{backbone_tag}_D_full"
    if key not in npz.files:
        raise KeyError(f"{alloc_file} has no key {key} (available: {sorted(npz.files)})")
    ad = _import_allocate()
    spec, shared, _rho = ad.allocate(npz[key], k_specific=int(k_specific), k_freeze=None)
    if len(spec) != int(k_specific):
        raise RuntimeError(f"oracle allocation returned {len(spec)} layers != k_specific={k_specific} (allocate contract violated)")
    sig = hashlib.sha1(p.read_bytes()).hexdigest()[:12]
    return spec, shared, f"npz:{sig}"


def causal_allocation(d_stack, k_specific, k_freeze, num_layers=NUM_LAYERS):
    """Causal-arm allocation: calls allocate on the raw D_t of the first k_freeze tasks only
    ([K,R,L]). d_stack must be exactly the K uncentred D_raw matrices cached during warm-up
    (same convention as *_D_full in the diag npz; allocate does its own two_way_center)."""
    D = np.asarray(d_stack, dtype=np.float64)
    if D.ndim != 3 or D.shape[0] != int(k_freeze):
        raise ValueError(
            f"causal allocation needs exactly K={k_freeze} D_t matrices, got shape {D.shape}: "
            "did D collection silently fail for a warm-up task? Refusing to allocate on missing data.")
    ad = _import_allocate()
    spec, shared, rho = ad.allocate(D, k_specific=int(k_specific), k_freeze=int(k_freeze))
    return spec, shared, rho
