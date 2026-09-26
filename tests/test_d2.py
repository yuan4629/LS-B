# -*- coding: utf-8 -*-
"""CPU-only tests for the D2 shared/specific adapter allocation (no GPU, no training).

Run:  python tests/test_d2.py        (or: pytest tests/test_d2.py)

Each group checks one design property; a failure means the property is not implemented:
  [1]  allocation: allocate() on the frozen diag npz reproduces the expected allocations
       (including the dino K=3 negative control)
  [2]  mode dispatch: CLI choices == implementation table; unknown modes raise
  [3]  identity key round trip: missing new fields fall back to defaults, so existing run
       outputs are still judged identical
  [4]  role semantics on a mock of fft.py: shared slots keep updating across tasks, specific
       slots are isolated per task, the shared delta-W is counted once, and BiLoRA's
       name-based unfreezing never releases the shared slots
  [5]  random_split: differs per seed, reproducible per seed, correct k
  [6]  gate x learner integration with a fake learner
  [8]  fixed_window_w* sliding-window arms
  [9]  best_approximation: bit-identical to ratio_causal at k=4, persistent-parameter
       formula, K=1 degeneration
  [10] dataset wiring (CUB, CIFAR-100, GRID_LS, unknown dataset names)
  [11] pinned allocation arm
  [12] ImageNet-R with T=20

Blocks that need inputs not included in the code release (the diag npz, completed run
outputs, datasets, the upstream BiLoRA checkout, bash) are skipped with a reason.
"""
import glob
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

OK, FAIL = "  ✓", "  ✗"
_fails = []
_skips = []

NPZ = "outputs/atlas_v2_diag_k4_b10.npz"
BILORA_DIR = os.path.join("baselines", "BiLoRA")
NPZ_REASON = f"needs {NPZ}, produced by baselines/bilora_adapter/atlas_v2_diag.py (not shipped)"
RUNS_REASON = "needs completed run outputs (not shipped)"
UPSTREAM_REASON = "needs the upstream BiLoRA checkout, run scripts/fetch_third_party.sh"
BASH_REASON = "needs bash"


def _dataset_reason(*names):
    return (f"needs the {' and '.join(names)} dataset{'s' if len(names) > 1 else ''}, "
            "see docs/DATASETS.md")


def _need_npz():
    return None if os.path.exists(NPZ) else NPZ_REASON


def _need_upstream():
    return None if os.path.isfile(os.path.join(BILORA_DIR, "utils", "schedulers.py")) else UPSTREAM_REASON


def _need_runs():
    return None if glob.glob("outputs/exp_bioscore_v2/*/metrics.json") else RUNS_REASON


def check(name, cond, extra=""):
    print((OK if cond else FAIL) + f" {name}" + (f"  [{extra}]" if extra else ""))
    if not cond:
        _fails.append(name)


def _skip(name, reason):
    """Record a block that cannot run because an input is not part of the release."""
    print(f"  - skip {name}: {reason}")
    _skips.append((name, reason))


def raises(name, exc, fn):
    try:
        fn()
    except exc as e:
        check(name, True, f"{type(e).__name__}: {str(e)[:60]}")
        return
    except Exception as e:  # noqa: BLE001 -- the exception type is part of the contract
        check(name, False, f"raised {type(e).__name__} (expected {exc.__name__})")
        return
    check(name, False, f"did not raise {exc.__name__}")


def _mk_args(**over):
    from core.cli import apply_preset, build_argparser
    a = apply_preset(build_argparser().parse_args(["--preset", "timm_cil_cifar100"]))
    for k, v in over.items():
        setattr(a, k, v)
    return a


# ---------------------------------------------------------------------------
def t1_allocation():
    print("\n[1] allocation: allocate(frozen npz) reproduces the expected allocations")
    from analyze_d2_split import allocate
    from baselines.bilora_adapter.d2_split import causal_allocation, oracle_allocation

    if _need_npz():
        _skip("t1", NPZ_REASON)
        return
    npz = np.load(NPZ)
    reg = {"augreg": [0, 2, 3, 7], "ibot": [3, 5, 6, 7], "dino": [0, 4, 5, 6]}  # expected allocations
    for bk, want in reg.items():
        spec, shared, _ = allocate(npz[f"{bk}_D_full"])
        check(f"{bk} oracle specific == {want}", spec == want, f"got {spec}")
        check(f"{bk} shared = complement, disjoint",
              sorted(spec + shared) == list(range(12)))
    for bk in ("augreg", "ibot"):  # K=3 matches the oracle exactly (J=1.00)
        specK, _, _ = allocate(npz[f"{bk}_D_full"], k_freeze=3)
        check(f"{bk} causal K=3 == oracle (J=1.00)", specK == reg[bk], f"got {specK}")
    # Negative control: dino K=3 vs oracle has J=0.60; an exact match would mean k_freeze is ignored.
    dK, _, _ = allocate(npz["dino_D_full"], k_freeze=3)
    j = len(set(dK) & set(reg["dino"])) / len(set(dK) | set(reg["dino"]))
    check("dino causal K=3 != oracle and J=0.60 (k_freeze really truncates to the first K tasks)",
          dK != reg["dino"] and abs(j - 0.60) < 1e-9, f"got {dK} J={j:.2f}")

    # Host-side wrapper: oracle_allocation(npz path) equals allocate() and carries a content hash.
    spec, shared, sig = oracle_allocation(NPZ, "augreg", 4)
    check("oracle_allocation(npz, augreg) == allocate", spec == reg["augreg"] and sig.startswith("npz:"),
          f"{spec} sig={sig}")
    # causal_allocation takes the raw D_t of the first 3 tasks (same as the online cache).
    specC, sharedC, rho = causal_allocation(npz["augreg_D_full"][:3], 4, 3)
    check("causal_allocation(raw D of the first 3 tasks) == expected", specC == reg["augreg"], f"got {specC}")
    check("causal_allocation returns rho for 12 layers", len(np.asarray(rho).ravel()) == 12)
    # Too few D_t or a bad input must raise: a silently failed warm-up collection must not be swallowed.
    raises("causal_allocation with 2 D_t raises ValueError", ValueError,
           lambda: causal_allocation(npz["augreg_D_full"][:2], 4, 3))
    raises("oracle_allocation on a missing file raises FileNotFoundError", FileNotFoundError,
           lambda: oracle_allocation("outputs/no_such.npz", "augreg", 4))
    raises("oracle_allocation rejects a hand-copied .json table", ValueError,
           lambda: oracle_allocation("outputs/atlas_v2_diag_k4_b10.json", "augreg", 4))
    raises("oracle_allocation with a missing backbone key raises KeyError", KeyError,
           lambda: oracle_allocation(NPZ, "clipzilla", 4))


# ---------------------------------------------------------------------------
def t2_mode_dispatch():
    print("\n[2] mode dispatch: cli choices == implementation table; unknown modes raise")
    from core.cli import build_argparser
    from baselines.bilora_adapter.d2_split import (
        D2_SPLIT_MODES, DYNAMIC_MODES, backbone_tag_from_args, static_specific_layers)

    p = build_argparser()
    act = next(a for a in p._actions if a.dest == "bioscore_split_mode")
    check("cli choices == D2_SPLIT_MODES (exact)", set(act.choices) == set(D2_SPLIT_MODES),
          f"cli^impl={sorted(set(act.choices) ^ set(D2_SPLIT_MODES))}")
    check("cli default is none (non-D2 runs unaffected)", act.default == "none")
    mact = next(a for a in p._actions if a.dest == "methods")
    check("--methods choices include bilora_d2", "bilora_d2" in mact.choices)
    from core.registry import REGISTRY
    check("registry registers bilora_d2", REGISTRY.get("bilora_d2") == "baselines.bilora_adapter.bilora_d2")

    for mode in D2_SPLIT_MODES:  # dispatch the whole vocabulary, none may be skipped
        if mode in DYNAMIC_MODES:
            check(f"{mode} -> dynamic (None)", static_specific_layers(mode, 0, 4) is None)
        elif mode == "none":
            raises("none -> ValueError (not an arm)", ValueError,
                   lambda: static_specific_layers("none", 0, 4))
        else:
            spec = static_specific_layers(mode, 0, 4)
            want_k = {"all_shared": 0, "all_specific": 12}.get(mode, 4)
            check(f"{mode} -> valid set with k={want_k}",
                  spec == sorted(spec) and len(spec) == want_k
                  and all(0 <= l < 12 for l in spec), f"got {spec}")
    raises("unknown mode brainz -> ValueError", ValueError,
           lambda: static_specific_layers("brainz", 0, 4))
    raises("n_specific out of range (13) -> ValueError", ValueError,
           lambda: static_specific_layers("random_split", 0, 13))

    # fixed_depth_deep mirrors fixed_depth. The generic loop only checks "k valid layers", not which
    # side they sit on, and the side is the whole point of this arm, so pin the exact sets here.
    check("fixed_depth_deep(n=4) -> [8,9,10,11] (shared = shallow 8 layers, DualPrompt-style)",
          static_specific_layers("fixed_depth_deep", 0, 4) == [8, 9, 10, 11],
          f"got {static_specific_layers('fixed_depth_deep', 0, 4)}")
    check("fixed_depth_deep(n=10) -> [2..11] (shared = shallow 2 layers)",
          static_specific_layers("fixed_depth_deep", 0, 10) == list(range(2, 12)),
          f"got {static_specific_layers('fixed_depth_deep', 0, 10)}")
    check("fixed_depth_deep and fixed_depth are mirrors (same n: union = all 12 layers, disjoint)",
          sorted(static_specific_layers("fixed_depth_deep", 0, 6)
                 + static_specific_layers("fixed_depth", 0, 6)) == list(range(12)))
    # alloc_seed only affects random_split; passing it to another arm would look like a changed allocation.
    raises("alloc_seed passed to fixed_depth_deep -> ValueError", ValueError,
           lambda: static_specific_layers("fixed_depth_deep", 0, 4, alloc_seed=7))
    # The default None must reproduce the training-seed draws, otherwise old runs lose their identity.
    check("alloc_seed=None reproduces the training-seed draw (old runs reproducible)",
          [static_specific_layers("random_split", s, 4) for s in (0, 1, 2)]
          == [[2, 5, 8, 11], [0, 4, 9, 10], [1, 6, 9, 10]])
    check("alloc_seed=A depends only on A (not on the training seed)",
          static_specific_layers("random_split", 99, 4, alloc_seed=0) == [2, 5, 8, 11]
          and static_specific_layers("random_split", 7, 4, alloc_seed=2) == [1, 6, 9, 10])

    # The gate fails fast at construction time (before any GPU work).
    from baselines.bilora_adapter.bilora_d2 import D2SharedAdapterGate
    dev = torch.device("cpu")
    raises("gate(none) construction raises ValueError", ValueError,
           lambda: D2SharedAdapterGate(_mk_args(bioscore_split_mode="none"), dev))
    raises("gate(ratio_causal, num_tasks<=K) raises ValueError (warm-up would cover the whole run)", ValueError,
           lambda: D2SharedAdapterGate(_mk_args(bioscore_split_mode="ratio_causal",
                                                num_tasks=3, split_freeze_task=3), dev))
    raises("gate(ratio_oracle, no alloc file) raises ValueError", ValueError,
           lambda: D2SharedAdapterGate(_mk_args(bioscore_split_mode="ratio_oracle",
                                                d2_alloc_file=""), dev))
    if os.path.exists(NPZ):
        g = D2SharedAdapterGate(_mk_args(bioscore_split_mode="ratio_oracle",
                                         d2_alloc_file=NPZ, bilora_weights=""), dev)
        check("gate(ratio_oracle, augreg) allocation is [0,2,3,7]",
              g._spec == [0, 2, 3, 7] and g._alloc_sig.startswith("npz:"), f"{g._spec}")
        g2 = D2SharedAdapterGate(_mk_args(bioscore_split_mode="ratio_oracle", d2_alloc_file=NPZ,
                                          bilora_weights="./pretrained/ibot_vitb16.pth"), dev)
        check("gate(ratio_oracle, ibot) picks the key by backbone: [3,5,6,7]", g2._spec == [3, 5, 6, 7],
              f"{g2._spec}")
    else:
        _skip("t2 gate(ratio_oracle) allocation", NPZ_REASON)
    ga = D2SharedAdapterGate(_mk_args(bioscore_split_mode="all_shared"), dev)
    check("gate(all_shared) k_effective=0 and shared=0..11",
          ga.k_effective() == 0 and ga._shared == list(range(12)))
    gl = D2SharedAdapterGate(_mk_args(bioscore_split_mode="fixed_depth_l", n_specific_layers=6), dev)
    check("gate(fixed_depth_l, l=6) specific=[0..5]", gl._spec == list(range(6)), f"{gl._spec}")
    check("backbone_tag: '' -> augreg / ibot path -> ibot",
          backbone_tag_from_args(_mk_args(bilora_weights="")) == "augreg"
          and backbone_tag_from_args(_mk_args(bilora_weights="./pretrained/ibot_vitb16.pth")) == "ibot")
    raises("backbone_tag: cannot infer -> ValueError", ValueError,
           lambda: backbone_tag_from_args(_mk_args(bilora_weights="./pretrained/weird.pth")))


# ---------------------------------------------------------------------------
def t3_identity_roundtrip():
    print("\n[3] identity key round trip: missing new fields -> defaults; existing v2 jobs judged identical")
    olds = sorted(glob.glob("outputs/exp_bioscore_v2/*/metrics.json"))
    if not olds:
        _skip("t3", RUNS_REASON)
        return
    import exp

    for k in ("bioscore_split_mode", "n_specific_layers", "shared_adapter_lr_scale",
              "split_freeze_task"):
        check(f"_IDENTITY_KEYS contains {k}", k in exp._IDENTITY_KEYS)
        check(f"_V2_KEY_DEFAULTS has a default for missing {k}", k in exp._V2_KEY_DEFAULTS)
    check("d2_alloc_file deliberately not in the key (paths differ across machines)",
          "d2_alloc_file" not in exp._IDENTITY_KEYS)

    check("existing v2 metrics.json present (real files, no synthetic stand-ins)", bool(olds),
          f"{len(olds)} files")
    new_defaults = {"bioscore_split_mode": "none", "n_specific_layers": 4,
                    "shared_adapter_lr_scale": 0.1, "split_freeze_task": 3}
    for mp in olds[:3] + olds[-1:]:
        with open(mp, encoding="utf-8") as f:
            old_cfg = json.load(f)["config"]
        check(f"old config has no D2 fields (precondition): {os.path.basename(os.path.dirname(mp))}",
              all(k not in old_cfg for k in new_defaults))
        cur = dict(old_cfg)
        cur.update(new_defaults)   # = vars(args) when the new code reruns the same command line
        check(f"identity judged identical: {os.path.basename(os.path.dirname(mp))}",
              exp._identity(old_cfg) == exp._identity(cur))
    # Counter-example: a config with D2 enabled must be judged different (else a d2 run would
    # overwrite the v2 metrics.json).
    with open(olds[0], encoding="utf-8") as f:
        old_cfg = json.load(f)["config"]
    cur = dict(old_cfg)
    cur.update(new_defaults)
    cur["bioscore_split_mode"] = "ratio_causal"
    check("counter-example: split_mode=ratio_causal judged different", exp._identity(old_cfg) != exp._identity(cur))
    cur2 = dict(old_cfg)
    cur2.update(new_defaults)
    cur2["n_specific_layers"] = 6
    check("counter-example: n_specific=6 judged different", exp._identity(old_cfg) != exp._identity(cur2))
    check("empty config: all four fields normalise to their defaults",
          all(exp._identity({})[k] == v for k, v in new_defaults.items()))

    # ---- cross-environment resume guard (old metrics.json files do not record their environment) ----
    import tempfile
    from pathlib import Path as _Path
    check("_ENV records python/torch/numpy/timm", all(exp._ENV.get(k) for k in ("python", "torch", "numpy", "timm")),
          f"{exp._ENV.get('python')}/torch{exp._ENV.get('torch')}/np{exp._ENV.get('numpy')}")
    check("env deliberately not in identity (a torch patch release must not invalidate every resume)",
          "env" not in exp._IDENTITY_KEYS)

    cur_full = dict(old_cfg)
    cur_full.update(new_defaults)

    def _fake(td, env, n_results):
        p = _Path(td) / "metrics.json"
        payload = {"config": old_cfg, "task_classes": [],
                   "results": [{"_run_name": "bilora_d2", "threshold": 1.0}] * n_results}
        if env is not None:
            payload["env"] = env
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    os.environ.pop("ALLOW_ENV_DRIFT", None)
    with tempfile.TemporaryDirectory() as td:
        raises("old run without env record + completed units -> cross-environment resume refused", RuntimeError,
               lambda: exp._load_resume(_fake(td, None, 2), cur_full))
    with tempfile.TemporaryDirectory() as td:
        _, d = exp._load_resume(_fake(td, None, 0), cur_full)
        check("no env but nothing done yet (fresh job) -> allowed", d == set())
    with tempfile.TemporaryDirectory() as td:
        _, d = exp._load_resume(_fake(td, exp._ENV, 2), cur_full)
        check("env matches the current one -> normal resume", len(d) == 1)
    with tempfile.TemporaryDirectory() as td:
        os.environ["ALLOW_ENV_DRIFT"] = "1"
        try:
            _, d = exp._load_resume(_fake(td, {"torch": "0.0.0"}, 2), cur_full)
            check("ALLOW_ENV_DRIFT=1 -> explicitly allowed (prints a warning, not silent)", len(d) == 1)
        finally:
            os.environ.pop("ALLOW_ENV_DRIFT", None)


# ---------------------------------------------------------------------------
# Minimal mock for [4]: reproduces only the part of fft.py's Attention_FFT contract that D2 uses
# (dim/n_frq/qkv.weight.device/coef_k/coef_v/select_pos/get_delta_w_k/v).
class _MockAttention(nn.Module):
    def __init__(self, dim=8, n_frq=6, n_tasks=4, num_heads=2):
        super().__init__()
        self.dim, self.n_frq, self.num_heads = dim, n_frq, num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.coef_k = nn.ParameterList([nn.Parameter(torch.zeros(n_frq)) for _ in range(n_tasks)])
        self.coef_v = nn.ParameterList([nn.Parameter(torch.zeros(n_frq)) for _ in range(n_tasks)])
        self.indices = [self.select_pos(t, dim) for t in range(n_tasks)]

    def select_pos(self, t, dim, seed=777):  # same formula as fft.py
        idx = torch.randperm(dim * dim, generator=torch.Generator().manual_seed(seed + t * 10))[: self.n_frq]
        return torch.stack([idx // dim, idx % dim], dim=0)

    def _dw(self, coef, task):
        F = torch.zeros(self.dim, self.dim)
        ind = self.indices[task]
        F[ind[0, :], ind[1, :]] = coef[task]
        return torch.fft.ifft2(F, dim=(-2, -1)).real * 300

    def get_delta_w_k(self, task, alpha=300):
        return self._dw(self.coef_k, task)

    def get_delta_w_v(self, task, alpha=300):
        return self._dw(self.coef_v, task)


class _MockNet(nn.Module):
    def __init__(self, num_layers=4, **kw):
        super().__init__()
        blocks = []
        for _ in range(num_layers):
            b = nn.Module()
            b.attn = _MockAttention(**kw)
            blocks.append(b)
        enc = nn.Module()
        enc.blocks = nn.ModuleList(blocks)
        self.image_encoder = enc


def _bilora_style_freeze_then_unfreeze(net, task):
    """Mimic BiLoRA._train's freeze and name-based unfreeze (including its substring matching)."""
    for _n, p in net.named_parameters():
        p.requires_grad_(False)
    for n, p in net.named_parameters():
        if f"coef_k.{task}" in n or f"coef_v.{task}" in n:
            p.requires_grad_(True)


def t4_role_semantics():
    print("\n[4] role semantics (mock): shared slots update across tasks / specific slots isolated / "
          "shared delta-W counted once")
    from baselines.bilora_adapter.bilora_d2 import (
        apply_d2_roles, build_split_optimizer, create_shared_slots,
        effective_delta_weights, shared_delta_w)

    torch.manual_seed(0)
    NL = 4
    net = _MockNet(num_layers=NL)
    spec, shared = [0, 1], [2, 3]

    # Counter-example first: the slots do not exist yet, so using layer 2 as shared must raise.
    raises("apply_d2_roles on a layer without slots raises RuntimeError", RuntimeError,
           lambda: apply_d2_roles(net, 0, spec, NL))
    params = create_shared_slots(net, shared, NL)
    check("shared slots = 2 layers x (k,v) = 4 parameters, zero-initialised",
          len(params) == 4 and all(float(p.abs().sum()) == 0.0 for p in params))
    check("shared slots registered in named_parameters (saved in state_dict)",
          sum(1 for n, _ in net.named_parameters() if "d2_shared_coef" in n) == 4)
    raises("building the slots twice raises RuntimeError", RuntimeError, lambda: create_shared_slots(net, shared, NL))
    raises("specific layer with a shared slot raises RuntimeError (allocation-drift guard)", RuntimeError,
           lambda: apply_d2_roles(net, 0, [0, 2], NL))

    # BiLoRA's name-based unfreezing must never release the shared slots (substring match on "coef_k.<t>").
    _bilora_style_freeze_then_unfreeze(net, 0)
    check("BiLoRA name matching does not unfreeze the shared slots",
          all(not p.requires_grad for p in params))
    apply_d2_roles(net, 0, spec, NL)
    a0, a2 = net.image_encoder.blocks[0].attn, net.image_encoder.blocks[2].attn
    check("after roles: specific-layer task slot trainable / shared-layer task slot frozen / shared slot trainable",
          a0.coef_k[0].requires_grad and not a2.coef_k[0].requires_grad
          and a2.d2_shared_coef_k.requires_grad)

    # Optimizer: two groups, shared group lr = lr x 0.1; shared params in the base group must raise.
    base = [p for p in net.parameters() if p.requires_grad and id(p) not in {id(x) for x in params}]
    opt = build_split_optimizer("adam", base, params, lr=1e-2, weight_decay=0.0, lr_scale=0.1)
    check("param group1 lr = 1e-3 (= 1e-2 x 0.1)", abs(opt.param_groups[1]["lr"] - 1e-3) < 1e-12)
    raises("shared params in the base group raise RuntimeError", RuntimeError,
           lambda: build_split_optimizer("adam", base + params[:1], params, 1e-2, 0.0, 0.1))
    raises("unknown optimizer raises ValueError", ValueError,
           lambda: build_split_optimizer("rmsprop", base, params, 1e-2, 0.0, 0.1))

    # The loss target must be a non-uniform matrix: with delta-W starting at 0, a constant target gives a
    # spatially constant upstream gradient, orthogonal to every non-DC Fourier basis, so only the (0,0)
    # coefficient would get a gradient. That is a degenerate test loss, not an implementation issue (in
    # the real host delta-W enters attention via F.linear(x, .) and gets gradients on the full spectrum).
    g = torch.Generator().manual_seed(1234)
    Tk = [torch.randn(8, 8, generator=g) for _ in range(NL)]
    Tv = [torch.randn(8, 8, generator=g) for _ in range(NL)]

    def steps(optimizer, task, n=5):
        for _ in range(n):
            loss = 0.0
            for lid in range(NL):
                wk, wv = effective_delta_weights(net.image_encoder.blocks[lid].attn, task)
                loss = loss + ((wk - Tk[lid]) ** 2).sum() + ((wv - Tv[lid]) ** 2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    flat = lambda ps: torch.cat([p.detach().reshape(-1) for p in ps]).clone()
    sh0 = flat(params)
    steps(opt, task=0)
    sh1 = flat(params)
    check("task0: shared slots updated (delta > 0)", float((sh1 - sh0).norm()) > 0)
    check("task0: specific-layer task slot updated", float(a0.coef_k[0].abs().sum()) > 0)
    check("task0: shared-layer task slot stays 0 (freeze works)", float(a2.coef_k[0].abs().sum()) == 0.0)
    spec0_snapshot = a0.coef_k[0].detach().clone()

    # task 1: mimic BiLoRA's per-task freeze -> unfreeze -> roles -> rebuild optimizer
    _bilora_style_freeze_then_unfreeze(net, 1)
    check("the blanket freeze also freezes the shared slots (real _train behaviour; the role step must restore them)",
          not a2.d2_shared_coef_k.requires_grad)
    apply_d2_roles(net, 1, spec, NL)
    base1 = [p for p in net.parameters() if p.requires_grad and id(p) not in {id(x) for x in params}]
    opt1 = build_split_optimizer("adam", base1, params, 1e-2, 0.0, 0.1)
    steps(opt1, task=1)
    sh2 = flat(params)
    check("task1: shared slots keep updating (never frozen per task)", float((sh2 - sh1).norm()) > 0)
    check("task1: task0's specific slot unchanged bit for bit (task isolation)",
          torch.equal(a0.coef_k[0].detach(), spec0_snapshot))
    check("task1: shared-layer task1 slot still 0", float(a2.coef_k[1].abs().sum()) == 0.0)

    # Shared delta-W counted once: shared-layer task slots are all 0, so the effective delta-W equals the
    # shared delta-W itself for any task.
    w_t0 = effective_delta_weights(a2, 0)[0]
    w_t3 = effective_delta_weights(a2, 3)[0]
    w_sh = shared_delta_w(a2, a2.d2_shared_coef_k)
    check("shared delta-W independent of the task (task=0 and task=3 bit-identical)", torch.equal(w_t0, w_t3))
    check("and exactly 1x the shared delta-W (not added once per task)", torch.allclose(w_t0, w_sh, atol=0, rtol=0))
    check("shared delta-W nonzero (the slot actually learned something)", float(w_sh.abs().sum()) > 0)
    # Specific layers have no shared slot: effective delta-W = sum of the task slots (original semantics).
    w_spec = effective_delta_weights(a0, 1)[0]
    w_manual = a0.get_delta_w_k(0) + a0.get_delta_w_k(1)
    check("specific layer = original sum of task slots (no shared term)", torch.allclose(w_spec, w_manual, atol=0, rtol=0))
    raises("delta-W for task<0 raises ValueError", ValueError, lambda: effective_delta_weights(a0, -1))


# ---------------------------------------------------------------------------
def t5_random_split():
    print("\n[5] random_split: differs per seed, reproducible per seed, correct k")
    from baselines.bilora_adapter.d2_split import random_split_layers, static_specific_layers

    sets = {}
    for sd in range(10):
        s = random_split_layers(sd, 4)
        sets[sd] = tuple(s)
        if not (len(s) == 4 and s == sorted(s) and all(0 <= l < 12 for l in s)):
            check(f"seed {sd} set valid", False, f"{s}")
            return
    check("sets for 10 seeds all valid (k=4, sorted, in range)", True)
    check("same seed twice gives the same set (reproducible)",
          random_split_layers(3, 4) == list(sets[3]))
    check("seeds 0/1/2 give pairwise different sets (the per-seed redraw really draws)",
          len({sets[0], sets[1], sets[2]}) == 3, f"{sets[0]} {sets[1]} {sets[2]}")
    check("10 seeds give at least 6 distinct sets", len(set(sets.values())) >= 6,
          f"{len(set(sets.values()))} distinct")
    check("static_specific_layers('random_split') and the direct call share one source",
          static_specific_layers("random_split", 7, 4) == list(sets[7]))
    check("k=0 -> empty set / k=12 -> all layers",
          random_split_layers(0, 0) == [] and random_split_layers(0, 12) == list(range(12)))


# ---------------------------------------------------------------------------
def t6_gate_learner_integration():
    print("\n[6] gate x learner integration (fake learner): warm-up -> freeze -> build slots -> rebuild optimizer")
    reason = _need_npz() or _need_upstream()
    if reason:
        _skip("t6", reason)
        return
    import contextlib
    import io

    from baselines.bilora_adapter.bilora_d2 import D2SharedAdapterGate, startup_lines

    br = os.path.join("baselines", "BiLoRA")   # _rebuild_optimizer needs utils.schedulers from BiLoRA
    if br not in sys.path:
        sys.path.insert(0, br)

    npz = np.load(NPZ)
    args = _mk_args(bioscore_split_mode="ratio_causal", num_tasks=10, split_freeze_task=3,
                    n_specific_layers=4, shared_adapter_lr_scale=0.1)
    gate = D2SharedAdapterGate(args, torch.device("cpu"))
    # Stub for the causal arm's online collection: inject the raw D of the first 3 tasks from the npz
    # (the collection pipeline itself is not exercised here).
    d_iter = iter(np.asarray(npz["augreg_D_full"][:3], dtype=np.float64))

    def _stub_collect(_loader):
        gate._d_cache.append(next(d_iter))
        gate._alloc_sig = "deadbeefdeadbeef"

    gate._collect_warmup_d = _stub_collect

    class _FakeLearner:  # only the attributes the gate uses (_cur_task/_network/optim/lr/run_epoch)
        def __init__(self, network):
            self._network = network
            self._cur_task = -1
            self.optim, self.run_epoch = "adam", 2
            self.init_lr = self.lrate = 5e-4
            self.init_weight_decay = self.weight_decay = 0.0

    net = _MockNet(num_layers=12, dim=8, n_frq=6, n_tasks=10)
    ln = _FakeLearner(net)
    buf = io.StringIO()
    same_opt = {}
    for t in range(5):
        ln._cur_task = t
        _bilora_style_freeze_then_unfreeze(net, t)
        base_opt = torch.optim.Adam(net.parameters(), lr=5e-4)
        with contextlib.redirect_stdout(buf):
            opt, sch = gate.on_task_start(ln, None, base_opt, None)
        same_opt[t] = opt is base_opt
        pre = gate.shared_snapshot()
        if gate._shared_params:                      # simulate training: move the shared params -> delta-norm > 0
            with torch.no_grad():
                for p in gate._shared_params:
                    p.add_(0.01)
        with contextlib.redirect_stdout(buf):
            gate.on_task_end(ln, pre)
        if t == 3:
            check("t=3 rebuilt optimizer has two groups and shared lr = task lr x 0.1",
                  len(opt.param_groups) == 2 and abs(opt.param_groups[1]["lr"] - 5e-5) < 1e-12)
            check("t=3 scheduler rebuilt with the new optimizer (CosineSchedule, two base_lrs)",
                  type(sch).__name__ == "CosineSchedule" and len(sch.base_lrs) == 2
                  and abs(sch.base_lrs[1] - 5e-5) < 1e-12)
    check("warm-up tasks (0..2) keep _train's optimizer; t>=3 uses the rebuilt one",
          same_opt[0] and same_opt[1] and same_opt[2] and not same_opt[3] and not same_opt[4],
          f"{same_opt}")
    check("from t=3 the causal allocation = expected [0,2,3,7]", gate._spec == [0, 2, 3, 7], f"{gate._spec}")
    check("shared slots = 8 shared layers x (k,v) = 16 parameters", len(gate._shared_params) == 16)
    raises("task index jump (5->7) raises RuntimeError", RuntimeError,
           lambda: (setattr(ln, "_cur_task", 7),
                    gate.on_task_start(ln, None, None, None)))
    raises("startup_lines with an unknown dataset raises ValueError", ValueError,
           lambda: startup_lines(gate, "cifar10"))


def t8_window_arms():
    """Sliding-window arms fixed_window_w*: vocabulary / parsing / range / forced k / alias rule."""
    print("\n[t8] fixed_window_w* sliding-window arms")
    from baselines.bilora_adapter.d2_split import (
        D2_SPLIT_MODES, static_specific_layers)

    wins = [m for m in D2_SPLIT_MODES if m.startswith("fixed_window_w")]
    check("t8.A vocabulary contains exactly w1..w7", wins == [f"fixed_window_w{i}" for i in range(1, 8)],
          f"got {wins}")
    check("t8.A alias rule: w0/w8 not in the vocabulary",
          "fixed_window_w0" not in D2_SPLIT_MODES and "fixed_window_w8" not in D2_SPLIT_MODES,
          "the endpoints must reuse fixed_depth_l4 / fixed_depth_deep, not become new arms")
    for w in range(1, 8):
        got = static_specific_layers(f"fixed_window_w{w}", seed=0, n_specific=4)
        check(f"t8.B w{w} = [{w}..{w + 3}]", got == list(range(w, w + 4)), f"got {got}")
    # Three semantic guards: k != 4 / out of range / stray alloc_seed must raise, not be accepted silently.
    try:
        static_specific_layers("fixed_window_w3", 0, 6)
        check("t8.C k!=4 raises", False, "n_specific=6 accepted silently")
    except ValueError:
        check("t8.C k!=4 raises", True, "")
    try:
        static_specific_layers("fixed_window_w8", 0, 4)
        check("t8.C w8 rejected (alias endpoint)", False, "w8 accepted silently")
    except ValueError:
        check("t8.C w8 rejected (alias endpoint)", True, "")
    try:
        static_specific_layers("fixed_window_w3", 0, 4, alloc_seed=7)
        check("t8.C stray alloc_seed raises", False, "window arm accepted alloc_seed")
    except ValueError:
        check("t8.C stray alloc_seed raises", True, "")
    # cli vocabulary matches d2_split exactly (same rule as t2, restated for the new arms).
    from core.cli import build_argparser
    ch = None
    for a in build_argparser()._actions:
        if a.dest == "bioscore_split_mode":
            ch = list(a.choices)
    check("t8.E cli choices contain all 7 windows", ch is not None and all(w in ch for w in wins),
          f"cli choices={ch}")


def t9_best_approximation():
    """Exploratory arm `best_approximation` (capacity-accuracy Pareto ablation).

    The core is the equivalence test t9.E: at k=4 this arm must give bit-identical results to the
    main arm `ratio_causal`. The host dispatches on `mode in CAUSAL_MODES`; a refactor from `==` to
    `in` that misses one site silently turns the new arm into a static arm without warm-up while the
    [gate] lines and delta-norms still look normal. Only an equivalence test catches that; the final
    accuracy does not.
    """
    print("\n[t9] best_approximation (capacity Pareto ablation arm)")
    import contextlib
    import io

    from baselines.bilora_adapter.bilora_d2 import D2SharedAdapterGate, startup_lines
    from baselines.bilora_adapter.d2_split import (
        CAUSAL_MODES, D2_SPLIT_MODES, DYNAMIC_MODES, persistent_param_groups,
        static_specific_layers)

    ARM = "best_approximation"

    # ---- t9.A vocabulary ----------------------------------------------------
    check("t9.A in the D2_SPLIT_MODES vocabulary", ARM in D2_SPLIT_MODES)
    check("t9.A in DYNAMIC_MODES (allocation produced at run time)", ARM in DYNAMIC_MODES)
    check("t9.A in CAUSAL_MODES (warm-up -> freeze pipeline)", ARM in CAUSAL_MODES)
    # The pinned arm follows the same pipeline and is in this tuple too (tested in t11).
    check("t9.A CAUSAL_MODES is exactly (ratio_causal, best_approximation, pinned)",
          tuple(CAUSAL_MODES) == ("ratio_causal", ARM, "pinned"), f"got {tuple(CAUSAL_MODES)}")
    from core.cli import build_argparser
    ch = None
    for a in build_argparser()._actions:
        if a.dest == "bioscore_split_mode":
            ch = list(a.choices)
    check("t9.A cli choices include the arm", ch is not None and ARM in ch)
    check("t9.A static_specific_layers returns None for the arm (dynamic)",
          static_specific_layers(ARM, seed=0, n_specific=4) is None)

    # ---- t9.B persistent parameter groups -----------------------------------
    # Expected (L=12, T=10): fixed_depth_l -> 9l+12; ratio_causal(k=4,K=3) -> 72;
    # all_specific -> 120; all_shared -> 12; this arm -> 6k+48.
    for l in range(0, 13):
        got = persistent_param_groups("fixed_depth_l", l, 3, 10)
        check(f"t9.B grid l={l} -> 9l+12 = {9 * l + 12}", got == 9 * l + 12, f"got {got}")
    check("t9.B all_shared -> 12", persistent_param_groups("all_shared", 4, 3, 10) == 12)
    check("t9.B all_specific -> 120", persistent_param_groups("all_specific", 4, 3, 10) == 120)
    check("t9.B ratio_causal(k=4,K=3) -> 72",
          persistent_param_groups("ratio_causal", 4, 3, 10) == 72)
    for k in range(1, 9):
        got = persistent_param_groups(ARM, k, 3, 10)
        check(f"t9.B arm k={k} -> 6k+48 = {6 * k + 48}", got == 6 * k + 48, f"got {got}")
    check("t9.B arm at k=12 equals all_specific = 120 (the two formulas meet at the endpoint)",
          persistent_param_groups(ARM, 12, 3, 10) == 120)
    check("t9.B K=0 reduces the causal formula to the grid formula (K is the only source of overhead)",
          persistent_param_groups("fixed_depth_l", 4, 0, 10) == 48)
    raises("t9.B K out of range raises ValueError", ValueError,
           lambda: persistent_param_groups(ARM, 4, 10, 10))
    raises("t9.B k out of range raises ValueError", ValueError,
           lambda: persistent_param_groups(ARM, 13, 3, 10))

    # ---- t9.C K-axis capacity -----------------------------------------------
    # Run names carry both k and K (`k<k>f<K>`): the K-axis point k4f1 and the acceptance point k4f3
    # differ in capacity (P(k=4,K) = 8K+48) and would overwrite each other if only k were in the name.
    _pk = [persistent_param_groups(ARM, 4, KK, 10) for KK in (1, 2, 3)]
    check("t9.C the three K-axis points have distinct capacities [56, 64, 72]", _pk == [56, 64, 72], f"got {_pk}")

    # ---- t9.D construction guards, as for ratio_causal ----------------------
    raises("t9.D num_tasks <= K raises ValueError (warm-up would cover the whole run)", ValueError,
           lambda: D2SharedAdapterGate(_mk_args(bioscore_split_mode=ARM, num_tasks=3,
                                                split_freeze_task=3, n_specific_layers=4),
                                       torch.device("cpu")))
    raises("t9.D split_freeze_task<1 raises ValueError", ValueError,
           lambda: D2SharedAdapterGate(_mk_args(bioscore_split_mode=ARM, num_tasks=10,
                                                split_freeze_task=0, n_specific_layers=4),
                                       torch.device("cpu")))

    # ---- t9.H K=1 degeneration: pin down the known behaviour ------------------
    # With K=1, layer_stats sees T=1, so E==0 and rho==0 identically; the allocation degenerates to the
    # natural argsort order [0,1,2,3], identical to fixed_depth_l4. This makes K=1 vs fixed_depth_l4 a
    # same-allocation comparison that differs only by one warm-up task. If rho at T=1 were ever
    # "fixed", that comparison would silently stop holding, so the test guards it.
    import numpy as _np
    from analyze_d2_split import layer_stats as _ls, allocate as _al
    _rng = _np.random.default_rng(12345)
    _D = _rng.normal(size=(3, 40, 12))
    _m, _s_, _rho = _ls(_D[:1])
    check("t9.H K=1 -> rho identically 0 (E==0 when T=1, independent of the data)",
          float(_np.max(_np.abs(_rho))) == 0.0, f"got max|rho|={float(_np.max(_np.abs(_rho)))}")
    _spec1, _, _ = _al(_D, 4, k_freeze=1)
    check("t9.H K=1 -> allocation degenerates to [0,1,2,3]", _spec1 == [0, 1, 2, 3], f"got {_spec1}")
    check("t9.H K=1 allocation identical to fixed_depth_l4 (precondition of the K=1 vs l4 comparison)",
          _spec1 == static_specific_layers("fixed_depth_l", 0, 4),
          f"got {_spec1} vs {static_specific_layers('fixed_depth_l', 0, 4)}")
    _spec3, _, _ = _al(_D, 4, k_freeze=3)
    check("t9.H K=3 does not degenerate (rho nonzero -> data-driven allocation, not the natural order)",
          float(_np.max(_np.abs(_ls(_D[:3])[2]))) > 0.0, "rho is all zero at K=3 too -> not limited to K=1")
    check("t9.H K=1 and K=3 capacities differ (56 vs 72, difference 2x(L-k)=16)",
          [persistent_param_groups(ARM, 4, K, 10) for K in (1, 3)] == [56, 72])

    # ---- t9.E/F/G end to end: bit-identical to the main arm at k=4; k=6 capacity point consistent ----
    reason = _need_npz() or _need_upstream()
    if reason:
        _skip("t9.E-G", reason)
        return
    br = os.path.join("baselines", "BiLoRA")
    if br not in sys.path:
        sys.path.insert(0, br)
    npz = np.load(NPZ)

    class _FakeLearner:
        def __init__(self, network):
            self._network = network
            self._cur_task = -1
            self.optim, self.run_epoch = "adam", 2
            self.init_lr = self.lrate = 5e-4
            self.init_weight_decay = self.weight_decay = 0.0

    def _drive(mode, k, n_task=5):
        """Run the first n_task tasks of a fake CL loop and return (gate, stdout text). Two calls differ
        only in mode/k: same npz D_raw for the first 3 tasks, same mock network, same optimizers."""
        args = _mk_args(bioscore_split_mode=mode, num_tasks=10, split_freeze_task=3,
                        n_specific_layers=k, shared_adapter_lr_scale=0.1)
        gate = D2SharedAdapterGate(args, torch.device("cpu"))
        d_iter = iter(np.asarray(npz["augreg_D_full"][:3], dtype=np.float64))

        def _stub(_loader):
            gate._d_cache.append(next(d_iter))
            gate._alloc_sig = "deadbeefdeadbeef"

        gate._collect_warmup_d = _stub
        net = _MockNet(num_layers=12, dim=8, n_frq=6, n_tasks=10)
        ln = _FakeLearner(net)
        buf = io.StringIO()
        for t in range(n_task):
            ln._cur_task = t
            _bilora_style_freeze_then_unfreeze(net, t)
            base_opt = torch.optim.Adam(net.parameters(), lr=5e-4)
            with contextlib.redirect_stdout(buf):
                gate.on_task_start(ln, None, base_opt, None)
            pre = gate.shared_snapshot()
            if gate._shared_params:
                with torch.no_grad():
                    for p in gate._shared_params:
                        p.add_(0.01)
            with contextlib.redirect_stdout(buf):
                gate.on_task_end(ln, pre)
        return gate, buf.getvalue()

    g_ref, t_ref = _drive("ratio_causal", 4)
    g_new, t_new = _drive(ARM, 4)

    check("t9.E k=4: allocation identical to the main arm", g_new._spec == g_ref._spec == [0, 2, 3, 7],
          f"new {g_new._spec} / main {g_ref._spec}")
    check("t9.E k=4: shared layers identical to the main arm", g_new._shared == g_ref._shared)
    check("t9.E per-task selected_layers identical to the main arm (including 3 all-12-layer warm-up tasks)",
          g_new.selected_layers == g_ref.selected_layers)
    check("t9.E per-task layer_scores (rho) identical to the main arm",
          np.allclose(np.asarray(g_new.layer_scores), np.asarray(g_ref.layer_scores)))
    check("t9.E same number of shared slots as the main arm (8 layers x k,v = 16)",
          len(g_new._shared_params) == len(g_ref._shared_params) == 16)
    check("t9.E log identical except for the arm name (src=<mode> is the only allowed difference)",
          t_new.replace(ARM, "ratio_causal") == t_ref)

    g6, t6txt = _drive(ARM, 6)
    check("t9.F k=6: exactly 6 specific / 6 shared layers",
          len(g6._spec) == 6 and len(g6._shared) == 6, f"{g6._spec}")
    check("t9.F k=6: shared slots = 6 layers x (k,v) = 12 parameters", len(g6._shared_params) == 12)
    check("t9.F k=6: warm-up is still 3 tasks (K independent of k)",
          [len(s) for s in g6.selected_layers[:3]] == [12, 12, 12])
    fields = g6.extra_result_fields()
    check("t9.F k=6: extra_result_fields records persistent param groups = 84",
          fields["d2_persistent_param_groups"] == 84, f"got {fields.get('d2_persistent_param_groups')}")
    check("t9.F k=6: records split_freeze_task=3 (None only for static arms)",
          fields["d2_split_freeze_task"] == 3)

    # ---- t9.G startup lines: the third line matches the formula ----------------
    lines = startup_lines(g6, "imagenet_r")
    check("t9.G line 3 prints persistent_param_groups=84",
          len(lines) == 3 and "persistent_param_groups=84" in lines[2], f"{lines[-1]}")
    check("t9.G line 3 has K = 3 (static arms print K=0)", "K=3)" in lines[2], f"{lines[2]}")


def t10_dataset_wiring():
    """Dataset wiring: CUB branch, CIFAR-100 fix, GRID_LS, unknown dataset names raise."""
    print("\n[t10] dataset wiring: cub / c100 / GRID_LS / unknown values")
    import contextlib
    import io
    import os as _os
    import re as _re
    import shutil as _sh
    import tempfile as _tf
    import types as _ty
    from core.cli import build_argparser, apply_preset
    from core.data import build_split, proto_loader_from
    from baselines.bilora_adapter.bilora_d2 import _DS_TOKEN

    p = build_argparser()
    ch = {a.dest: list(a.choices or []) for a in p._actions if a.dest in ("dataset", "preset")}
    check("t10.A --dataset choices include cub", "cub" in ch.get("dataset", []), f"{ch.get('dataset')}")
    check("t10.A --preset choices include timm_cil_cub", "timm_cil_cub" in ch.get("preset", []), f"{ch.get('preset')}")
    a = apply_preset(p.parse_args(["--preset", "timm_cil_cub", "--dataset", "cub"]))
    check("t10.A timm_cil_cub -> cub / 10 tasks x 20 classes",
          (a.dataset, a.num_tasks, a.classes_per_task) == ("cub", 10, 20),
          f"{(a.dataset, a.num_tasks, a.classes_per_task)}")
    check("t10.A timm_cil_cub -> default backbone timm", a.backbone == "timm", f"{a.backbone}")
    with contextlib.redirect_stderr(io.StringIO()):
        raises("t10.A `--dataset c100` (the old run_d2 c100 argument) is still rejected by argparse", SystemExit,
               lambda: p.parse_args(["--dataset", "c100"]))

    check("t10.B _DS_TOKEN: cub -> cub (dataset token in the run.log startup line)", _DS_TOKEN.get("cub") == "cub",
          f"{_DS_TOKEN}")
    raises("t10.C build_split raises on an unknown dataset (no silent fallback to CIFAR-100)", ValueError,
           lambda: build_split(_ty.SimpleNamespace(dataset="imagenet_a")))
    fake_loader = _ty.SimpleNamespace(dataset=_ty.SimpleNamespace(indices=[0]))
    raises("t10.C proto_loader_from raises on an unknown dataset", ValueError,
           lambda: proto_loader_from(fake_loader, _ty.SimpleNamespace(
               dataset="imagenet_a", data_root="./__no_such_root__", batch_size=2, num_workers=0)))

    txt = open(_os.path.join(ROOT, "run_d2.bash"), encoding="utf-8").read()
    check("t10.E run_d2.bash c100 branch sets DATASET=cifar100 explicitly",
          _re.search(r'^\s*c100\)\s+DS_ENV=\(PRESET=timm_cil_cifar100 DATASET=cifar100\);\s+DS_SUF="_c100"', txt, _re.M) is not None)
    check("t10.E run_d2.bash cub branch binds preset / dataset / suffix together",
          _re.search(r'^\s*cub\)\s+DS_ENV=\(PRESET=timm_cil_cub DATASET=cub\);\s+DS_SUF="_cub"', txt, _re.M) is not None)
    check("t10.E run_d2.bash GRID points can be narrowed by GRID_LS, default still 2 4 6 8 10",
          "for L in ${GRID_LS:-2 4 6 8 10}; do" in txt)

    from baselines.bilora_adapter.cub_data import ensure_bilora_cub, split_fingerprint
    tmp = _tf.mkdtemp()
    raises("t10.F ensure_bilora_cub: no cub under data_root -> raises", FileNotFoundError,
           lambda: ensure_bilora_cub(_ty.SimpleNamespace(data_root=tmp)))
    for split, classes in (("train", ["001.a", "002.b"]), ("test", ["001.a", "003.c"])):
        for c in classes:
            _os.makedirs(_os.path.join(tmp, "cub", split, c), exist_ok=True)
            open(_os.path.join(tmp, "cub", split, c, "x.jpg"), "wb").close()
    tr = split_fingerprint(_os.path.join(tmp, "cub", "train"))
    te = split_fingerprint(_os.path.join(tmp, "cub", "test"))
    check("t10.F split_fingerprint counts class folders and files; differing class folders are visible",
          tr[0] == ["001.a", "002.b"] and tr[1] == 2 and te[1] == 2 and tr[0] != te[0], f"{tr[:2]} / {te[:2]}")
    _sh.rmtree(tmp, ignore_errors=True)
    if _os.path.isdir(_os.path.join("data", "cub", "train")):
        root = ensure_bilora_cub(_ty.SimpleNamespace(data_root="./data"))
        check("t10.G real data/cub passes ensure_bilora_cub (APER split) and resolves to data/cub",
              _os.path.realpath(root) == _os.path.realpath(_os.path.join("data", "cub")))
    else:
        _skip("t10.G real data/cub", _dataset_reason("cub"))


def t11_pinned():
    """Pinned-allocation arm `pinned` (exploratory).

    Same pipeline as best_approximation with k=4, K=3; the only difference is that the 4 layers
    committed at t=K come from the registered constant --d2_pin_layers instead of the rho top-4:
      (a) with an input whose rho top-4 differs from the pinned layers, the gate uses the pinned layers
          from t=K on, while rho is still computed and logged;
      (b) for t<K the gate / log / optimizer are bit-identical to best_approximation;
      (c) the pinned layers enter the identity key: a pinned run != the best_approximation run with the
          same k/K/seed, and deep vs shallow differ only in that key; without the new argument the
          identity of existing metrics.json files is unchanged;
      (d) run_d2.bash STAGE=PINNED dry-run naming (exp_timm.bash stubbed, no training) and the
          exp_timm.bash pass-through.
    """
    print("\n[t11] pinned (pinned allocation + observation period)")
    import contextlib
    import io
    import shutil as _sh
    import subprocess as _sp
    import tempfile as _tf

    import exp
    from analyze_d2_split import allocate
    from baselines.bilora_adapter.bilora_d2 import D2SharedAdapterGate, startup_lines
    from baselines.bilora_adapter.d2_split import (
        CAUSAL_MODES, D2_SPLIT_MODES, DYNAMIC_MODES, parse_pin_layers, persistent_param_groups)
    from core.cli import build_argparser

    ARM, BA, K = "pinned", "best_approximation", 3
    REG = {"deep": [8, 9, 10, 11], "shallow": [0, 1, 2, 3]}          # the two registered pinned sets
    dev = torch.device("cpu")
    here = ROOT

    # ---- t11.A vocabulary / capacity -------------------------------------------
    check("t11.A in D2_SPLIT_MODES, DYNAMIC_MODES and CAUSAL_MODES",
          ARM in D2_SPLIT_MODES and ARM in DYNAMIC_MODES and ARM in CAUSAL_MODES)
    _ch = next(a for a in build_argparser()._actions if a.dest == "bioscore_split_mode").choices
    check("t11.A cli choices include the arm", ARM in _ch)
    check("t11.A P(k=4,K=3,T=10)=72, same as best_approximation_k4f3",
          persistent_param_groups(ARM, 4, K, 10) == 72 == persistent_param_groups(BA, 4, K, 10))

    # ---- t11.B parsing and construction guards ---------------------------------
    check("t11.B parse_pin_layers accepts the canonical form", parse_pin_layers("8,9,10,11", 4) == REG["deep"])
    for _bad in ("11,10,9,8", "8, 9,10,11", "8,8,9,10", "08,9,10,11", "8,9,10,12", "", "8,9,10"):
        raises(f"t11.B parse_pin_layers(k=4) rejects {_bad!r}", ValueError,
               lambda b=_bad: parse_pin_layers(b, 4))
    _p = build_argparser()
    check("t11.B argparse default d2_pin_layers=None (existing arms get None in vars(args))",
          _p.parse_args([]).d2_pin_layers is None)
    check("t11.B argparse accepts the canonical form and keeps the string as is",
          _p.parse_args(["--d2_pin_layers", "0,1,2,3"]).d2_pin_layers == "0,1,2,3")
    with contextlib.redirect_stderr(io.StringIO()):
        raises("t11.B argparse rejects the non-canonical 11,10,9,8 at parse time", SystemExit,
               lambda: _p.parse_args(["--d2_pin_layers", "11,10,9,8"]))

    def _args(mode, pin=None, **kw):
        base = dict(bioscore_split_mode=mode, num_tasks=10, split_freeze_task=K,
                    n_specific_layers=4, shared_adapter_lr_scale=0.1, d2_pin_layers=pin)
        base.update(kw)
        return _mk_args(**base)

    raises("t11.B gate(pinned, no pinned layers) raises ValueError (no default guessed)", ValueError,
           lambda: D2SharedAdapterGate(_args(ARM), dev))
    raises("t11.B gate(pinned, number of pinned layers != k) raises ValueError", ValueError,
           lambda: D2SharedAdapterGate(_args(ARM, "8,9,10,11", n_specific_layers=2), dev))
    raises("t11.B gate(pinned, num_tasks <= K) raises ValueError (same guard as the causal arm)", ValueError,
           lambda: D2SharedAdapterGate(_args(ARM, "8,9,10,11", num_tasks=K), dev))
    for _o in (BA, "ratio_causal", "fixed_depth_l", "all_shared"):
        raises(f"t11.B gate({_o}) with pinned layers raises ValueError (pinning must not leak into other arms)",
               ValueError, lambda o=_o: D2SharedAdapterGate(_args(o, "8,9,10,11"), dev))

    # ---- (c) identity key -------------------------------------------------------
    ident = exp._identity
    a_ba, a_dp, a_sh = _args(BA), _args(ARM, "8,9,10,11"), _args(ARM, "0,1,2,3")
    check("t11.E (c) d2_pin_layers is in exp._IDENTITY_KEYS", "d2_pin_layers" in exp._IDENTITY_KEYS)
    i_ba, i_dp, i_sh = ident(vars(a_ba)), ident(vars(a_dp)), ident(vars(a_sh))
    check("t11.E (c) a pinned run and the best_approximation run with the same k/K/seed differ in identity",
          i_dp != i_ba)
    check("t11.E (c) raw pinned string in the identity: deep='8,9,10,11', best_approximation=None",
          i_dp["d2_pin_layers"] == "8,9,10,11" and i_ba["d2_pin_layers"] is None)
    _d1 = sorted(k for k in i_dp if i_dp[k] != i_sh[k])
    check("t11.E (c) deep and shallow identities differ ONLY in d2_pin_layers (without it they would coincide)",
          _d1 == ["d2_pin_layers"], f"differing keys {_d1}")
    _d2 = sorted(k for k in i_dp if i_dp[k] != i_ba[k])
    check("t11.E (c) pinned vs best_approximation identities differ exactly in {bioscore_split_mode, d2_pin_layers}",
          _d2 == ["bioscore_split_mode", "d2_pin_layers"], f"differing keys {_d2}")
    # The two BA k4f3 s0 runs (AugReg / iBOT) are listed explicitly instead of globbing `_inr*`, so that
    # runs with other backbones are never picked up.
    _olds = ([p for p in ("outputs/exp_d2/d2_best_approximation_k4f3_inr_s0/metrics.json",
                          "outputs/exp_d2/d2_best_approximation_k4f3_inr_ibot_s0/metrics.json") if os.path.exists(p)]
             + sorted(glob.glob("outputs/exp_bioscore_v2/*/metrics.json"))[:1])
    if not _olds:
        _skip("t11.E existing metrics.json", RUNS_REASON)
    else:
        check("t11.E existing metrics.json present (BA k4f3 s0 for two backbones + one v2 job; read-only)",
              len(_olds) == 3, f"{len(_olds)} files")
        for _mp in _olds:
            with open(_mp, encoding="utf-8") as f:
                _oc = json.load(f)["config"]
            check(f"t11.E default unchanged: {os.path.basename(os.path.dirname(_mp))} has no such key; "
                  "identity unchanged after adding the argparse default None",
                  "d2_pin_layers" not in _oc and ident(_oc) == ident(dict(_oc, d2_pin_layers=None)))
        with open(_olds[0], encoding="utf-8") as f:
            _oc = json.load(f)["config"]
        check("t11.E counter-example: old BA config changed only to pinned + pinned layers -> judged different "
              "(no resume onto its metrics.json)",
              ident(_oc) != ident(dict(_oc, bioscore_split_mode=ARM, d2_pin_layers="8,9,10,11")))

    # ---- (d) run_d2.bash STAGE=PINNED dry-run naming + exp_timm.bash pass-through ----
    _head = open(os.path.join(here, "run_d2.bash"), encoding="utf-8").read().split("set -euo pipefail")[0]
    check("t11.G header comment documents STAGE=PINNED usage and PIN_SETS", "STAGE=PINNED" in _head and "PIN_SETS" in _head)
    _drop = ("STAGE", "BK", "DATASET", "SEEDS", "PIN_SETS", "ONLY", "FAST", "N_SPECIFIC", "FREEZE_T",
             "BA_KS", "BA_KFS", "GRID_LS", "D2_PIN_LAYERS", "BASE_OUT", "JOB_NAME", "PYTHON_BIN",
             "NTASKS", "CLASSES_PER_TASK", "NUM_TASKS")   # T=20 knobs (t12): keep the outer env out of the dry runs
    d = _tf.mkdtemp()
    try:
        _sh.copy(os.path.join(here, "run_d2.bash"), os.path.join(d, "run_d2.bash"))
        with open(os.path.join(d, "exp_timm.bash"), "w", encoding="utf-8", newline="\n") as f:
            f.write('#!/usr/bin/env bash\n'
                    'echo "RUN|${JOB_NAME}|${BIOSCORE_SPLIT_MODE}|${N_SPECIFIC_LAYERS}|'
                    '${SPLIT_FREEZE_TASK}|${D2_PIN_LAYERS:-}|${SEED}|${BILORA_WEIGHTS}"\n')

        def _dry(**kw):
            env = {k: v for k, v in os.environ.items() if k not in _drop}
            env.update(kw)
            pr = _sp.run(["bash", "run_d2.bash"], cwd=d, env=env, capture_output=True, text=True, timeout=60)
            return pr.returncode, [l.split("|")[1:] for l in pr.stdout.splitlines() if l.startswith("RUN|")]

        rc, runs = _dry(STAGE="PINNED", BK="augreg", DATASET="inr", PIN_SETS="deep shallow", SEEDS="0 1")
        _want = [["d2_pinned_k4f3_deep_inr_s0", ARM, "4", "3", "8,9,10,11", "0"],
                 ["d2_pinned_k4f3_shallow_inr_s0", ARM, "4", "3", "0,1,2,3", "0"],
                 ["d2_pinned_k4f3_deep_inr_s1", ARM, "4", "3", "8,9,10,11", "1"],
                 ["d2_pinned_k4f3_shallow_inr_s1", ARM, "4", "3", "0,1,2,3", "1"]]
        check("t11.G (d) STAGE=PINNED AugReg: job name / arm / k / K / pinned layers / seed as registered "
              "(seed-major order)",
              rc == 0 and [r[:6] for r in runs] == _want, f"rc={rc} {runs}")
        rc, runs = _dry(STAGE="PINNED", BK="ibot", DATASET="inr", PIN_SETS="deep", SEEDS="2")
        check("t11.G (d) STAGE=PINNED iBOT: d2_pinned_k4f3_deep_inr_ibot_s2 with iBOT weights",
              rc == 0 and len(runs) == 1
              and runs[0][:6] == ["d2_pinned_k4f3_deep_inr_ibot_s2", ARM, "4", "3", "8,9,10,11", "2"]
              and "ibot" in runs[0][6], f"rc={rc} {runs}")
        rc, runs = _dry(STAGE="PINNED", BK="augreg", DATASET="inr", PIN_SETS="shallow deep", SEEDS="0",
                        N_SPECIFIC="2", FREEZE_T="1")
        # the set name is the 4th "_" field of d2_pinned_k4f3_<set>_inr_s<seed>
        _cross = bool(runs) and all(
            r[0].split("_")[3] in REG and r[4] == ",".join(str(x) for x in REG[r[0].split("_")[3]])
            for r in runs)
        check("t11.G (d) N_SPECIFIC/FREEZE_T do not affect this branch (still k4f3); "
              "launched pinned layers match the registered sets",
              rc == 0 and len(runs) == 2 and all(r[2:4] == ["4", "3"] for r in runs) and _cross,
              f"rc={rc} {runs}")
        rc, runs = _dry(STAGE="PINNED", BK="augreg", DATASET="inr", PIN_SETS="deep middle", SEEDS="0")
        check("t11.G (d) PIN_SETS with an unregistered name -> nonzero exit and no run started (list validated first)",
              rc != 0 and runs == [], f"rc={rc} {runs}")
        rc, runs = _dry(STAGE="PINNED", BK="augreg", DATASET="inr", SEEDS="0")
        check("t11.G (d) missing PIN_SETS -> nonzero exit and no run", rc != 0 and runs == [], f"rc={rc} {runs}")
        rc, runs = _dry(STAGE="BESTAPPROX", BK="augreg", DATASET="inr", BA_KS="4", BA_KFS="3", SEEDS="0")
        check("t11.G default behaviour: BESTAPPROX naming unchanged, no pinned layers",
              rc == 0 and [r[:6] for r in runs] == [["d2_best_approximation_k4f3_inr_s0", BA, "4", "3", "", "0"]],
              f"rc={rc} {runs}")
        rc, runs = _dry(STAGE="ENDPOINT", BK="ibot", DATASET="inr", SEEDS="0")
        check("t11.G default behaviour: ENDPOINT naming of the three arms unchanged, no pinned layers",
              rc == 0 and [(r[0], r[4]) for r in runs] == [("d2_all_shared_inr_ibot_s0", ""),
                                                          ("d2_all_specific_inr_ibot_s0", ""),
                                                          ("d2_random_split_inr_ibot_s0", "")],
              f"rc={rc} {runs}")
        # exp_timm.bash: --d2_pin_layers is passed only when D2_PIN_LAYERS is set (python stubbed, cwd in a temp dir)
        py = os.path.join(d, "fakepy")
        with open(py, "w", encoding="utf-8", newline="\n") as f:
            f.write('#!/usr/bin/env bash\nif [ "$1" = "-" ]; then cat > /dev/null; exit 0; fi\n'
                    'printf "%s\\n" "$@"\n')
        os.chmod(py, 0o755)

        def _argv(**kw):
            env = {k: v for k, v in os.environ.items() if k not in _drop}
            env.update(PYTHON_BIN=py, BASE_OUT=os.path.join(d, "out"), JOB_NAME="j", METHODS="bilora_d2", **kw)
            pr = _sp.run(["bash", os.path.join(here, "exp_timm.bash")], cwd=d, env=env,
                         capture_output=True, text=True, timeout=60)
            return pr.returncode, pr.stdout.splitlines()

        rc1, l1 = _argv(BIOSCORE_SPLIT_MODE=ARM, D2_PIN_LAYERS="8,9,10,11")
        _i = l1.index("--d2_pin_layers") if "--d2_pin_layers" in l1 else -1
        check("t11.G (d) exp_timm.bash: D2_PIN_LAYERS set -> passes --d2_pin_layers 8,9,10,11",
              rc1 == 0 and _i >= 0 and l1[_i + 1:_i + 2] == ["8,9,10,11"], f"rc={rc1}")
        rc2, l2 = _argv(BIOSCORE_SPLIT_MODE=BA)
        check("t11.G default behaviour: D2_PIN_LAYERS unset -> no --d2_pin_layers on the command line",
              rc2 == 0 and "--bioscore_split_mode" in l2 and "--d2_pin_layers" not in l2, f"rc={rc2}")
    except FileNotFoundError as e:
        if shutil.which("bash") is None:
            _skip("t11.G run_d2.bash / exp_timm.bash dry runs", BASH_REASON)
        else:
            check("t11.G dry-run prerequisites present", False, f"{e}")
    finally:
        _sh.rmtree(d, ignore_errors=True)

    # ---- (a)(b) fake CL loop over 10 tasks: needs the diag npz and the upstream BiLoRA ----
    reason = _need_npz() or _need_upstream()
    if reason:
        _skip("t11.C/t11.D", reason)
        return
    br = os.path.join("baselines", "BiLoRA")
    if br not in sys.path:
        sys.path.insert(0, br)
    D3 = np.asarray(np.load(NPZ)["augreg_D_full"][:K], dtype=np.float64)
    rho_top4 = allocate(D3, 4, k_freeze=K)[0]
    check("t11.C precondition: the input's rho top-4 differs from both deep and shallow (else (a) is vacuous)",
          rho_top4 != REG["deep"] and rho_top4 != REG["shallow"], f"rho top-4={rho_top4}")

    class _FakeLearner:
        def __init__(self, network):
            self._network = network
            self._cur_task = -1
            self.optim, self.run_epoch = "adam", 2
            self.init_lr = self.lrate = 5e-4
            self.init_weight_decay = self.weight_decay = 0.0

    def _drive(mode, pin=None):
        """Fake CL loop as in t9 over all 10 tasks; the arms differ only in mode / pinned layers."""
        args = _args(mode, pin)
        gate = D2SharedAdapterGate(args, dev)
        d_iter = iter(D3)

        def _stub(_loader):
            gate._d_cache.append(next(d_iter))
            gate._alloc_sig = "deadbeefdeadbeef"

        gate._collect_warmup_d = _stub
        net = _MockNet(num_layers=12, dim=8, n_frq=6, n_tasks=10)
        ln = _FakeLearner(net)
        buf, same_opt = io.StringIO(), []
        for t in range(10):
            ln._cur_task = t
            _bilora_style_freeze_then_unfreeze(net, t)
            base_opt = torch.optim.Adam(net.parameters(), lr=5e-4)
            with contextlib.redirect_stdout(buf):
                opt, _ = gate.on_task_start(ln, None, base_opt, None)
            same_opt.append(opt is base_opt)
            pre = gate.shared_snapshot()
            if gate._shared_params:
                with torch.no_grad():
                    for p_ in gate._shared_params:
                        p_.add_(0.01)
            with contextlib.redirect_stdout(buf):
                gate.on_task_end(ln, pre)
        return gate, buf.getvalue(), same_opt

    g_ba, t_ba, o_ba = _drive(BA)
    g_dp, t_dp, o_dp = _drive(ARM, "8,9,10,11")
    g_sh, _, _ = _drive(ARM, "0,1,2,3")

    # ---- (a) from t=K the gate uses the pinned layers; rho still computed and logged ----
    check("t11.C (a) deep: specific at t=3..9 is always [8,9,10,11] (not the rho top-4)",
          len(g_dp.selected_layers) == 10 and all(s == REG["deep"] for s in g_dp.selected_layers[K:]),
          f"t=K got {g_dp.selected_layers[K]}")
    check("t11.C (a) shallow: specific at t=3..9 is always [0,1,2,3]",
          all(s == REG["shallow"] for s in g_sh.selected_layers[K:]), f"t=K got {g_sh.selected_layers[K]}")
    check("t11.C (a) control: with the same input best_approximation takes the rho top-4 for t>=K",
          all(s == rho_top4 for s in g_ba.selected_layers[K:]))
    check("t11.C (a) shared = complement of the pinned layers; shared slots 8 layers x (k,v) = 16",
          g_dp._shared == [0, 1, 2, 3, 4, 5, 6, 7] and len(g_dp._shared_params) == 16)
    check("t11.C (a) the t=K line reads: [d2] pinned_layers=[8, 9, 10, 11] rho_top4=<rho top-4>",
          t_dp.count("[d2] pinned_layers=") == 1
          and f"[d2] pinned_layers=[8, 9, 10, 11] rho_top4={rho_top4}\n" in t_dp)
    check("t11.C (a) that line is printed between the [gate] lines of task 2 and task 3",
          t_dp.index("[gate] task 2:") < t_dp.index("[d2] pinned_layers=") < t_dp.index("[gate] task 3:"))
    check("t11.C (a) rho still computed and logged: layer_scores for t>=K identical to best_approximation",
          g_dp._rho is not None and g_dp.layer_scores[K:] == g_ba.layer_scores[K:])

    # ---- (b) t<K identical to best_approximation ---------------------------------
    _pre_dp = t_dp.split("[d2] pinned_layers=")[0]
    _pre_ba = t_ba.split("[gate] task 3:")[0]
    check("t11.D (b) log for t<K identical to best_approximation except the src arm name",
          "[gate] task 2:" in _pre_ba
          and _pre_dp.replace("src=pinned(", "src=best_approximation(") == _pre_ba)
    check("t11.D (b) selected_layers / shared_layers / layer_scores for t<K identical",
          g_dp.selected_layers[:K] == g_ba.selected_layers[:K]
          and g_dp.shared_layers[:K] == g_ba.shared_layers[:K]
          and g_dp.layer_scores[:K] == g_ba.layer_scores[:K])
    check("t11.D (b) optimizer: t<K keeps _train's, t>=K uses the rebuilt one (same for both arms)",
          o_dp == o_ba == [True] * K + [False] * (10 - K), f"pinned={o_dp} ba={o_ba}")
    check("t11.D (b) D_t cached during warm-up identical (same rho input)",
          len(g_dp._d_cache) == K and all(np.array_equal(x, y) for x, y in zip(g_dp._d_cache, g_ba._d_cache)))
    _sl_dp, _sl_ba = startup_lines(g_dp, "imagenet_r"), startup_lines(g_ba, "imagenet_r")
    check("t11.D startup lines identical to best_approximation except the mode, and P=72",
          [x.replace("mode=pinned ", "mode=best_approximation ") for x in _sl_dp] == _sl_ba
          and "persistent_param_groups=72 (L=12 T=10 k=4 K=3)" in _sl_dp[2], f"{_sl_dp}")
    _f = g_dp.extra_result_fields()
    check("t11.D recorded fields: mode=pinned, k=4, K=3, P=72",
          (_f["d2_split_mode"], _f["d2_k_specific"], _f["d2_split_freeze_task"],
           _f["d2_persistent_param_groups"]) == (ARM, 4, K, 72), f"{_f}")
    check("t11.D default behaviour: best_approximation prints no pinned_layers line and gate._pin is None",
          "pinned_layers" not in t_ba and g_ba._pin is None)


def t12_t20():
    """ImageNet-R with T=20 (exploratory): 20 tasks x 10 classes per task.

      (a) defaults unchanged: with NTASKS unset and NTASKS=10 the dry-run stdout, job names, the full
          environment passed to exp_timm.bash and the exp_timm.bash command line are identical; names,
          output root and command line (no --classes_per_task) are as before;
      (b) T=20 names and commands: 7 configurations x seeds 0-2 = d2_<arm>_inr_t20_s<seed>, output root
          outputs/exp_d2_t20, --classes_per_task 10 right after --num_tasks 20, everything else as for
          the same arm at T=10; FAST adds _fast;
      (c) presets do not override explicit values; on IN-R/CUB num_tasks x classes_per_task != 200
          raises; re-parsing the CMD line of real runs keeps the identity key;
      (d) host config: _build_cfg gives total_sessions=20, init_cls=increment=10, epochs still 40/20;
          DataManager 20x10 with the same class order as our build_split; task-layout cross-check
          (model slots / heads / DataManager) with positive and negative cases, done once before the
          first task;
      (e) persistent parameter groups of the 7 configurations at T=20 (host formula and gate startup line);
      (g) run_d2.bash rejects unregistered usage: NTASKS=20 outside inr, NTASKS not in {10,20},
          unregistered configurations / STAGE / backbone / output root / caller-set class count;
      (h) read-out batches: with fewer than 10 batches collect_batches takes what is there (no error, no
          cycling), drive_matrix records the counts and the host logs them per task; on the real IN-R
          split the first 3 tasks at T=20 have 1443/1142/1748 images -> 12/9/14 loader batches.
    """
    print("\n[t12] ImageNet-R T=20")
    import contextlib
    import io
    import shutil as _sh
    import subprocess as _sp
    import tempfile as _tf
    import types as _ty

    import exp
    from baselines.bilora_adapter.bilora import _build_cfg
    from baselines.bilora_adapter.bilora_d2 import (
        D2SharedAdapterGate, startup_lines, task_layout_line, warmup_readout_line)
    from baselines.bilora_adapter.d2_split import persistent_param_groups
    from core.cli import apply_preset, build_argparser
    from core.data import build_split
    from model_m.common.bioscore_v2 import BioScoreV2Calculator as _V2
    from torch.utils.data import DataLoader, TensorDataset

    dev = torch.device("cpu")
    here = ROOT
    INR = ("--preset", "timm_cil_imagenet_r", "--dataset", "imagenet_r")
    # The 7 T=20 configurations: job token -> (bioscore_split_mode, k, K, P)
    W2 = {"all_shared": ("all_shared", 0, 0, 12), "fixed_depth_l2": ("fixed_depth_l", 2, 0, 50),
          "best_approximation_k2f3": ("best_approximation", 2, 3, 80), "fixed_depth_l4": ("fixed_depth_l", 4, 0, 88),
          "best_approximation_k4f3": ("best_approximation", 4, 3, 112), "fixed_depth_l6": ("fixed_depth_l", 6, 0, 126),
          "all_specific": ("all_specific", 12, 0, 240)}

    def _quiet(fn):
        def w():
            with contextlib.redirect_stdout(io.StringIO()):
                return fn()
        return w

    # ---- (e) P: host formula / gate startup line == expected table ---------------------------------
    for tok, (mode, k, K, P) in W2.items():
        host = persistent_param_groups(mode, k, 3, 20)
        g = D2SharedAdapterGate(_mk_args(bioscore_split_mode=mode, num_tasks=20, classes_per_task=10,
                                         n_specific_layers=(k if mode in ("fixed_depth_l", "best_approximation") else 4),
                                         split_freeze_task=3), dev)
        line3 = startup_lines(g, "imagenet_r")[2]
        check(f"t12.E {tok}: P(T=20) host formula = gate = expected {P}",
              host == g.persistent_groups == P, f"host {host} / gate {g.persistent_groups}")
        check(f"t12.E {tok}: startup line reads persistent_param_groups={P} (L=12 T=20 k={k} K={K})",
              line3 == f"[d2] persistent_param_groups={P} (L=12 T=20 k={k} K={K})", line3)
    check("t12.E control: the same 7 configurations at T=10 give P = 12/30/60/48/72/66/120 (T enters only as a parameter)",
          [persistent_param_groups(m_, k_, 3, 10) for (m_, k_, _K, _P) in W2.values()] == [12, 30, 60, 48, 72, 66, 120])

    # ---- (c) presets do not override explicit values; T x classes != 200 raises ----------------------
    p = build_argparser()

    def _res(*argv):
        a = apply_preset(p.parse_args(list(argv)))
        return a.num_tasks, a.classes_per_task

    check("t12.C IN-R default -> 10 tasks x 20 classes", _res(*INR) == (10, 20), f"{_res(*INR)}")
    check("t12.C IN-R explicit --num_tasks 10 (always passed by exp_timm.bash) -> still 10 x 20",
          _res(*INR, "--num_tasks", "10") == (10, 20))
    check("t12.C IN-R explicit --num_tasks 20 --classes_per_task 10 -> 20 x 10 (the preset does not reset the explicit 10)",
          _res(*INR, "--num_tasks", "20", "--classes_per_task", "10") == (20, 10))
    for _b, _why in ((("--num_tasks", "20"), "20x20=400: T=20 without the class count"),
                     (("--num_tasks", "20", "--classes_per_task", "20"), "20x20=400"),
                     (("--classes_per_task", "10"), "10x10=100: only half of the dataset"),
                     (("--num_tasks", "5", "--classes_per_task", "20"), "5x20=100")):
        raises(f"t12.C IN-R {' '.join(_b)} -> ValueError ({_why})", ValueError, lambda b=_b: _res(*INR, *b))
    raises("t12.C dataset=imagenet_r with the cifar preset (10x10) -> ValueError (judged by dataset, not preset)", ValueError,
           lambda: _res("--preset", "timm_cil_cifar100", "--dataset", "imagenet_r"))
    check("t12.C CUB default 10 x 20 unchanged", _res("--preset", "timm_cil_cub", "--dataset", "cub") == (10, 20))
    raises("t12.C CUB --classes_per_task 10 -> ValueError (also a fixed 200-class split)", ValueError,
           lambda: _res("--preset", "timm_cil_cub", "--dataset", "cub", "--classes_per_task", "10"))
    check("t12.C CIFAR default 10 x 10; smoke --num_tasks 2 -> 2 x 10 (CIFAR need not cover all classes)",
          _res() == (10, 10) and _res("--num_tasks", "2") == (2, 10))
    # Real existing runs (read-only: the CMD line of run.log and the config in metrics.json; no accuracies)
    if not os.path.isdir(os.path.join("outputs", "exp_d2")):
        _skip("t12.C real runs", RUNS_REASON)
    else:
        for _n in ("d2_all_shared_inr_s0", "d2_random_split_inr_ibot_s1", "d2_fixed_depth_l6_inr_s2",
                   "d2_best_approximation_k4f3_inr_ibot_s0"):
            _rd = os.path.join("outputs", "exp_d2", _n)
            try:
                with open(os.path.join(_rd, "run.log"), encoding="utf-8", errors="replace") as f:
                    _cmd = next(l for l in f if l.startswith("CMD: "))
                with open(os.path.join(_rd, "metrics.json"), encoding="utf-8") as f:
                    _oc = json.load(f)["config"]
            except (OSError, StopIteration, KeyError, ValueError) as e:
                check(f"t12.C real run {_n} present (CMD line and config only)", False, f"{type(e).__name__}")
                continue
            _a = apply_preset(p.parse_args(_cmd.split()[2:]))
            check(f"t12.C real run {_n}: re-parsing the CMD line with the current parser -> same "
                  "num_tasks/classes_per_task and identical identity key",
                  (_a.num_tasks, _a.classes_per_task) == (_oc["num_tasks"], _oc["classes_per_task"])
                  and exp._identity(vars(_a)) == exp._identity(_oc),
                  f"{(_a.num_tasks, _a.classes_per_task)} vs {(_oc['num_tasks'], _oc['classes_per_task'])}")

    # ---- (d) host config / DataManager / class order / task-layout cross-check ----------------------
    a20 = apply_preset(p.parse_args([*INR, "--num_tasks", "20", "--classes_per_task", "10"]))
    a10 = apply_preset(p.parse_args([*INR, "--num_tasks", "10"]))
    _src_b = io.open(os.path.join(here, "baselines", "bilora_adapter", "bilora.py"), encoding="utf-8").read()
    check("t12.D host does not read BiLoRA's mimg20_bilora.json (its 50/50 epochs must not leak in)", "mimg20" not in _src_b)
    dm20 = None
    _missing = [n for n, pth in (("imagenet-r", os.path.join("data", "imagenet-r")),
                                 ("imagenet-r-bilora", os.path.join("data", "imagenet-r-bilora", "train")))
                if not os.path.isdir(pth)]
    _ds_reason = _dataset_reason(*_missing) if _missing else _need_upstream()
    if _ds_reason:
        _skip("t12.D host config / DataManager", _ds_reason)
    else:
        # ensure_bilora_imagenet_r stats ~30k files to verify the source (~85 s per call on a slow bind
        # mount). That is not what this check tests (real runs still call it), so it is replaced by the
        # materialised directory for these two calls and restored right after.
        import baselines.bilora_adapter.inr_data as _inr
        _orig_ensure = _inr.ensure_bilora_imagenet_r
        _inr.ensure_bilora_imagenet_r = lambda args: os.path.join(args.data_root, "imagenet-r-bilora")
        try:
            c20, c10 = _build_cfg(a20, dev), _build_cfg(a10, dev)
        finally:
            _inr.ensure_bilora_imagenet_r = _orig_ensure
        _k = ("dataset", "total_sessions", "init_cls", "increment", "init_epoch", "epochs")
        check("t12.D _build_cfg(T=20): imagenet_r / total_sessions=20 / init_cls=increment=10 / epochs 40/20",
              tuple(c20[x] for x in _k) == ("imagenet_r", 20, 10, 10, 40, 20), f"{tuple(c20[x] for x in _k)}")
        check("t12.D _build_cfg(T=10) unchanged: total_sessions=10 / init_cls=increment=20 / epochs 40/20",
              tuple(c10[x] for x in _k) == ("imagenet_r", 10, 20, 20, 40, 20), f"{tuple(c10[x] for x in _k)}")
        br = os.path.join("baselines", "BiLoRA")
        if br not in sys.path:
            sys.path.insert(0, br)
        from utils.data_manager import DataManager
        dm20 = DataManager(c20["dataset"], c20["shuffle"], 0, c20["init_cls"], c20["increment"], c20)
        _sizes = sorted({dm20.get_task_size(t) for t in range(dm20.nb_tasks)})
        check("t12.D host DataManager(T=20): 20 tasks of 10 classes, class order 0..199 "
              "(shuffle=False -> independent of the seed)",
              dm20.nb_tasks == 20 and _sizes == [10] and c20["shuffle"] is False
              and list(dm20._class_order) == list(range(200)), f"nb={dm20.nb_tasks} sizes={_sizes}")
        _a20q = apply_preset(p.parse_args([*INR, "--num_tasks", "20", "--classes_per_task", "10", "--num_workers", "0"]))
        with contextlib.redirect_stdout(io.StringIO()):
            _tc = build_split(_a20q)[3]
        from torchvision.datasets.folder import find_classes
        check("t12.D our build_split(T=20) task_classes match the host DataManager (task t = classes [10t, 10t+10)) "
              "and the class folders are identical on both sides -> same class ids",
              _tc == [list(dm20._class_order[10 * t:10 * t + 10]) for t in range(20)]
              and find_classes(os.path.join("data", "imagenet-r"))[0]
              == find_classes(os.path.join(c20["data_path"], "train"))[0], f"{len(_tc)} tasks")

    class _Net(_MockNet):
        def __init__(self, n_tasks, n_cls):
            super().__init__(num_layers=12, dim=8, n_frq=6, n_tasks=n_tasks)
            self.classifier_pool = nn.ModuleList([nn.Linear(4, n_cls) for _ in range(n_tasks)])
            self.classifier_pool_backup = nn.ModuleList([nn.Linear(4, n_cls) for _ in range(n_tasks)])

    g20 = D2SharedAdapterGate(_mk_args(bioscore_split_mode="best_approximation", num_tasks=20, classes_per_task=10,
                                       n_specific_layers=4, split_freeze_task=3), dev)
    _dmf = _ty.SimpleNamespace(nb_tasks=20, get_task_size=lambda t: 10)
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        g20.confirm_task_layout(_ty.SimpleNamespace(_network=_Net(20, 10)), _dmf)
    check("t12.D task-layout cross-check (20-slot network + 20x10 DataManager): exactly one line, all fields right, no error",
          _buf.getvalue() == task_layout_line(20, 10, 20, 10, 20, 20, 10) + "\n", _buf.getvalue().strip())
    if dm20 is not None:
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            g20.confirm_task_layout(_ty.SimpleNamespace(_network=_Net(20, 10)), dm20)
        check("t12.D task-layout cross-check also passes with the real host DataManager(T=20)",
              _buf.getvalue() == task_layout_line(20, 10, 20, 10, 20, 20, 10) + "\n", _buf.getvalue().strip())
    raises("t12.D counter-example: the model has only 10 task slots (total_sessions not passed to SiNet) -> RuntimeError",
           RuntimeError, _quiet(lambda: g20.confirm_task_layout(_ty.SimpleNamespace(_network=_Net(10, 10)), _dmf)))
    raises("t12.D counter-example: 20 classes per head -> RuntimeError", RuntimeError,
           _quiet(lambda: g20.confirm_task_layout(_ty.SimpleNamespace(_network=_Net(20, 20)), _dmf)))
    raises("t12.D counter-example: DataManager still 10 tasks x 20 classes (class count not passed to the host) "
           "-> RuntimeError", RuntimeError,
           _quiet(lambda: g20.confirm_task_layout(_ty.SimpleNamespace(_network=_Net(20, 10)),
                                                  _ty.SimpleNamespace(nb_tasks=10, get_task_size=lambda t: 20))))

    class _FakeBiLoRA:
        def __init__(self, cfg):
            self._cur_task = -1
            self._network = _Net(20, 10)
            self.seen = []

        def incremental_train(self, data_manager):
            self._cur_task += 1
            self.seen.append(self._cur_task)

    _L = g20.make_learner_cls(_FakeBiLoRA)({})
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        for _ in range(3):
            _L.incremental_train(_dmf)
    check("t12.D learner hook: layout cross-check once before the first task, then the parent incremental_train as usual",
          _buf.getvalue().count("[d2] task_layout ") == 1 and _L.seen == [0, 1, 2], f"{_L.seen}")
    _fft = os.path.join(BILORA_DIR, "models", "fft.py")
    if not os.path.isfile(_fft):
        _skip("t12.D fft.py source evidence", UPSTREAM_REASON)
    else:
        _src_fft = io.open(_fft, encoding="utf-8").read()
        check("t12.D source evidence: SiNet's task slots and both classifier pools use total_sessions, "
              "classes per head use init_cls",
              'n_tasks=args["total_sessions"]' in _src_fft
              and _src_fft.count('for i in range(args["total_sessions"])') == 2
              and 'self.class_num = args["init_cls"]' in _src_fft)

    # ---- (h) read-out batches -----------------------------------------------------------------------
    def _loader(n, with_idx=False):
        x, y = torch.zeros(n, 1), torch.zeros(n, dtype=torch.long)
        ds = TensorDataset(torch.arange(n), x, y) if with_idx else TensorDataset(x, y)
        return DataLoader(ds, batch_size=128, shuffle=True)      # as the host train_loader: drop_last=False

    _fake = _ty.SimpleNamespace(args=_ty.SimpleNamespace(seed=0))
    for _n, _want in ((1443, (10, 1280)), (1142, (9, 1142)), (1748, (10, 1280)), (2585, (10, 1280))):
        _ld = _loader(_n)
        _st = torch.get_rng_state()
        _out = _V2.collect_batches(_fake, _ld, 10)
        _got = (len(_out), int(sum(len(b) for b in _out)))
        check(f"t12.H collect_batches: {_n} images x bs128 -> {_want[0]} batches / {_want[1]} images "
              "(fewer than 10 is fine, no cycling; global RNG restored)",
              _got == _want and torch.equal(_st, torch.get_rng_state()), f"got {_got}")
    _rng = np.random.default_rng(7)
    _fake.collect_batches = lambda loader, mb=10: _V2.collect_batches(_fake, loader, mb)
    _fake._pool_std = lambda batches: None
    _fake.aggregate = lambda s: (_rng.normal(size=(5, 12)), np.ones(5))
    _fake.roi_names = [f"r{i}" for i in range(5)]
    _dt, _info = _V2.drive_matrix(_fake, _loader(1142), 10)
    check("t12.H drive_matrix bookkeeping: info has n_batches=9 / n_images=1142, D_raw as before",
          (_info.get("n_batches"), _info.get("n_images")) == (9, 1142) and np.asarray(_info["D_raw"]).shape == (5, 12),
          f"{(_info.get('n_batches'), _info.get('n_images'))}")
    g_ro = D2SharedAdapterGate(_mk_args(bioscore_split_mode="best_approximation", num_tasks=20, classes_per_task=10,
                                        n_specific_layers=4, split_freeze_task=3), dev)
    g_ro._calc_obj = _ty.SimpleNamespace(drive_matrix=lambda loader, mb: _V2.drive_matrix(_fake, loader, mb))
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        for _n in (1443, 1142, 1748):
            g_ro._collect_warmup_d(_loader(_n, with_idx=True))
    check("t12.H host logs one read-out line per task: task 0/1/2 = 10/10 1280, 9/10 1142, 10/10 1280; "
          "D_t cached as usual",
          _buf.getvalue().splitlines() == [warmup_readout_line(0, 10, 10, 1280), warmup_readout_line(1, 9, 10, 1142),
                                           warmup_readout_line(2, 10, 10, 1280)] and len(g_ro._d_cache) == 3,
          _buf.getvalue().strip())
    g_bad = D2SharedAdapterGate(_mk_args(bioscore_split_mode="best_approximation", num_tasks=20, classes_per_task=10,
                                         n_specific_layers=4, split_freeze_task=3), dev)
    g_bad._calc_obj = _ty.SimpleNamespace(drive_matrix=lambda loader, mb: (None, {"D_raw": np.zeros((5, 12))}))
    raises("t12.H calculator without n_batches/n_images -> host raises RuntimeError (no guessed counts)", RuntimeError,
           lambda: g_bad._collect_warmup_d(_loader(10, with_idx=True)))
    if dm20 is not None:
        _cnt = []
        for t in range(3):
            _lo = sum(dm20._increments[:t])
            _cnt.append(len(dm20.get_dataset(np.arange(_lo, _lo + dm20.get_task_size(t)), source="train", mode="train")))
        check("t12.H real IN-R split (host DataManager, T=20): first 3 tasks have [1443, 1142, 1748] training images "
              "-> [12, 9, 14] loader batches -> [10, 9, 10] read out",
              _cnt == [1443, 1142, 1748] and [-(-c // 128) for c in _cnt] == [12, 9, 14], f"{_cnt}")
    else:
        _skip("t12.H real IN-R split", _ds_reason)

    # ---- (a)(b)(g) run_d2.bash dry runs (exp_timm.bash stubbed, no training) + exp_timm.bash command line (python stubbed) ----
    _drop = ("STAGE", "BK", "DATASET", "SEEDS", "PIN_SETS", "ONLY", "FAST", "N_SPECIFIC", "FREEZE_T", "LR_SCALE",
             "BA_KS", "BA_KFS", "GRID_LS", "WINDOWS", "DEEPN", "D2_ALLOC_FILE",
             "D2_PIN_LAYERS", "ALLOC_SEED_TAG", "ATLAS_SEED", "BASE_OUT", "JOB_NAME", "PYTHON_BIN",
             "NTASKS", "CLASSES_PER_TASK", "NUM_TASKS")
    d = _tf.mkdtemp()
    try:
        _sh.copy(os.path.join(here, "run_d2.bash"), os.path.join(d, "run_d2.bash"))
        with open(os.path.join(d, "exp_timm.bash"), "w", encoding="utf-8", newline="\n") as f:
            f.write('#!/usr/bin/env bash\n'
                    'mkdir -p envdump\n'
                    'echo "RUN|${JOB_NAME}|${BASE_OUT}|${NUM_TASKS}|${CLASSES_PER_TASK-<unset>}|'
                    '${BIOSCORE_SPLIT_MODE}|${N_SPECIFIC_LAYERS}|${SPLIT_FREEZE_TASK}|${SEED}"\n'
                    'env -0 | LC_ALL=C sort -z | grep -z -v -e "^_=" -e "^SHLVL=" -e "^PWD=" -e "^OLDPWD=" '
                    '-e "^NTASKS=" > "envdump/${JOB_NAME}.env"\n')
        py = os.path.join(d, "fakepy")
        with open(py, "w", encoding="utf-8", newline="\n") as f:
            f.write('#!/usr/bin/env bash\nif [ "$1" = "-" ]; then cat > /dev/null; exit 0; fi\nprintf "%s\\n" "$@"\n')
        os.chmod(py, 0o755)

        def _dry(**kw):
            env = {k_: v_ for k_, v_ in os.environ.items() if k_ not in _drop}
            env.update(kw)
            ed = os.path.join(d, "envdump")
            _sh.rmtree(ed, ignore_errors=True)
            pr = _sp.run(["bash", "run_d2.bash"], cwd=d, env=env, capture_output=True, text=True, timeout=120)
            runs = [l.split("|")[1:] for l in pr.stdout.splitlines() if l.startswith("RUN|")]
            dumps = {}
            if os.path.isdir(ed):
                for fn in sorted(os.listdir(ed)):
                    with open(os.path.join(ed, fn), "rb") as f:
                        dumps[fn[:-len(".env")]] = f.read().decode("utf-8", errors="replace")
            return pr.returncode, pr.stdout, runs, dumps

        def _argv(dump):
            env = {}
            for rec in dump.split("\0"):
                if "=" in rec:
                    k_, _, v_ = rec.partition("=")
                    env[k_] = v_
            env["PYTHON_BIN"] = py
            pr = _sp.run(["bash", os.path.join(here, "exp_timm.bash")], cwd=d, env=env,
                         capture_output=True, text=True, timeout=60)
            lines = pr.stdout.splitlines()
            return pr.returncode, (lines[:lines.index("")] if "" in lines else lines)

        # (a) defaults unchanged
        CASES10 = [dict(STAGE="ENDPOINT", BK="augreg", DATASET="inr", SEEDS="0"),
                   dict(STAGE="ENDPOINT", BK="ibot", DATASET="c100", SEEDS="1", FAST="1"),
                   dict(STAGE="MAIN", BK="augreg", DATASET="inr", SEEDS="0"),
                   dict(STAGE="GRID", BK="augreg", DATASET="inr", SEEDS="0"),
                   dict(STAGE="GRID", BK="dino", DATASET="cub", SEEDS="2", GRID_LS="4 6 8"),
                   dict(STAGE="CALIB", BK="augreg", DATASET="inr", SEEDS="0 1"),
                   dict(STAGE="RAND2", BK="ibot", DATASET="inr", SEEDS="0"),
                   dict(STAGE="WINDOW", BK="augreg", DATASET="inr", SEEDS="0", WINDOWS="1 7"),
                   dict(STAGE="DEEP", BK="augreg", DATASET="inr", SEEDS="0", DEEPN="4"),
                   dict(STAGE="BESTAPPROX", BK="augreg", DATASET="inr", SEEDS="0"),
                   dict(STAGE="PINNED", BK="ibot", DATASET="inr", SEEDS="0", PIN_SETS="deep shallow")]
        _same, _n, _badf, _t20, _ref = True, 0, [], [], {}
        for c in CASES10:
            r_u, r_10 = _dry(**c), _dry(NTASKS="10", **c)
            _same &= (r_u[0] == r_10[0] == 0 and bool(r_u[2]) and r_u[1] == r_10[1] and r_u[3] == r_10[3])
            _n += len(r_u[2])
            _badf += [r[0] for r in r_u[2] if (r[1], r[2], r[3]) != ("./outputs/exp_d2", "10", "<unset>")]
            _t20 += [r[0] for r in r_u[2] if "_t20" in r[0]]
            _ref.update(r_u[3])
        check(f"t12.A {len(CASES10)} STAGE x backbone x dataset combinations: stdout, job names and the full environment "
              "passed to exp_timm.bash identical for NTASKS unset and NTASKS=10",
              _same and _n > 0, f"{_n} runs")
        check("t12.A defaults: output root ./outputs/exp_d2, NUM_TASKS=10, CLASSES_PER_TASK unset, no _t20 in names",
              not _badf and not _t20, f"{_badf} {_t20}")
        _, _, _r1, _ = _dry(STAGE="GRID", BK="dino", DATASET="cub", SEEDS="2", GRID_LS="4 6 8")
        _, _, _r2, _ = _dry(STAGE="CALIB", BK="augreg", DATASET="inr", SEEDS="0 1")
        _, _, _r3, _ = _dry(STAGE="ENDPOINT", BK="ibot", DATASET="c100", SEEDS="1", FAST="1")
        check("t12.A default naming anchors (GRID x dino x cub / CALIB _a suffix / c100 x iBOT smoke) unchanged",
              [r[0] for r in _r1] == ["d2_fixed_depth_l4_cub_dino_s2", "d2_fixed_depth_l6_cub_dino_s2",
                                      "d2_fixed_depth_l8_cub_dino_s2"]
              and [r[0] for r in _r2] == ["d2_random_split_inr_s1_a0", "d2_random_split_inr_s0_a2",
                                          "d2_random_split_inr_s1_a2"]
              and [r[0] for r in _r3] == ["d2_all_shared_c100_ibot_s1_fast", "d2_all_specific_c100_ibot_s1_fast",
                                          "d2_random_split_c100_ibot_s1_fast"], f"{_r1} {_r2} {_r3}")
        _ok_argv, _nargv = True, 0
        for c in (CASES10[0], CASES10[9], CASES10[10]):
            du, d10 = _dry(**c)[3], _dry(NTASKS="10", **c)[3]
            for job in du:
                ru, r10 = _argv(du[job]), _argv(d10[job])
                _nargv += 1
                _i = ru[1].index("--num_tasks") if "--num_tasks" in ru[1] else -1
                _ok_argv &= (ru[0] == r10[0] == 0 and ru[1] == r10[1] and "--classes_per_task" not in ru[1]
                             and _i >= 0 and ru[1][_i + 1:_i + 3] == ["10", "--max_train_steps"])
        check("t12.A exp_timm.bash command line identical for NTASKS unset and =10; --num_tasks 10 is directly "
              "followed by --max_train_steps (nothing inserted)",
              _ok_argv and _nargv == 8, f"{_nargv} lines (ENDPOINT 3 + BESTAPPROX 3 + PINNED 2)")

        # (b) T=20 names and commands
        T20 = dict(BK="augreg", DATASET="inr", NTASKS="20")
        want21 = sorted(f"d2_{tok}_inr_t20_s{s}" for tok in W2 for s in (0, 1, 2))
        _names, _rcs, _fields, _modes = [], [], True, True
        for c in (dict(STAGE="BESTAPPROX", BA_KS="4", BA_KFS="3"), dict(STAGE="GRID", GRID_LS="4 6"),
                  dict(STAGE="ENDPOINT", ONLY="all_specific"), dict(STAGE="BESTAPPROX", BA_KS="2", BA_KFS="3"),
                  dict(STAGE="GRID", GRID_LS="2"), dict(STAGE="ENDPOINT", ONLY="all_shared")):
            rc, _, runs, _ = _dry(SEEDS="0 1 2", **T20, **c)
            _rcs.append(rc)
            for r in runs:
                _names.append(r[0])
                _fields &= (r[1], r[2], r[3]) == ("./outputs/exp_d2_t20", "20", "10")
                tok = r[0][len("d2_"):r[0].index("_inr_t20")]
                mode, k, K, _P = W2[tok]
                if mode in ("fixed_depth_l", "best_approximation"):
                    _modes &= (r[4], r[5]) == (mode, str(k)) and (mode != "best_approximation" or r[6] == str(K))
                else:
                    _modes &= r[4] == mode
        check("t12.B the 7 T=20 configurations x seeds 0-2 = 21 job names d2_<arm>_inr_t20_s<seed>, all accepted",
              _rcs == [0] * 6 and sorted(_names) == want21, f"rc={_rcs} {sorted(set(_names) ^ set(want21))}")
        check("t12.B every run: output root ./outputs/exp_d2_t20, NUM_TASKS=20, CLASSES_PER_TASK=10; "
              "arm / k / K match the name",
              _fields and _modes)
        rc, _, runs, _ = _dry(STAGE="GRID", GRID_LS="2 4 6", SEEDS="0", FAST="1", **T20)
        check("t12.B FAST=1 smoke at T=20: d2_fixed_depth_l{2,4,6}_inr_t20_s0_fast, output root still exp_d2_t20",
              rc == 0 and [r[0] for r in runs] == [f"d2_fixed_depth_l{l}_inr_t20_s0_fast" for l in (2, 4, 6)]
              and all(r[1] == "./outputs/exp_d2_t20" for r in runs), f"rc={rc} {runs}")
        for _tok, c in (("all_shared", dict(STAGE="ENDPOINT", ONLY="all_shared")),
                        ("best_approximation_k4f3", dict(STAGE="BESTAPPROX", BA_KS="4", BA_KFS="3"))):
            du = _dry(SEEDS="0", BK="augreg", DATASET="inr", **c)[3][f"d2_{_tok}_inr_s0"]
            d20 = _dry(SEEDS="0", **T20, **c)[3][f"d2_{_tok}_inr_t20_s0"]
            rc10, av10 = _argv(du)
            rc20, av20 = _argv(d20)
            exp20 = list(av10)
            _i = exp20.index("--num_tasks")
            exp20[_i + 1] = "20"
            exp20[_i + 2:_i + 2] = ["--classes_per_task", "10"]
            _j = exp20.index("--output_dir")
            exp20[_j + 1] = f"./outputs/exp_d2_t20/d2_{_tok}_inr_t20_s0"
            check(f"t12.B {_tok}: T=20 command line = T=10 command line of the same arm with --num_tasks 20, "
                  "--classes_per_task 10 inserted right after, output dir under exp_d2_t20, everything else identical",
                  rc10 == rc20 == 0 and av20 == exp20 and "--num_tasks" in av20
                  and av20[av20.index("--num_tasks"):av20.index("--num_tasks") + 4]
                  == ["--num_tasks", "20", "--classes_per_task", "10"], f"{av20}")
            s10 = {x for x in du.split("\0") if x}
            s20 = {x for x in d20.split("\0") if x}
            check(f"t12.B {_tok}: environment passed to exp_timm.bash differs only in JOB_NAME / BASE_OUT / NUM_TASKS "
                  "and the added CLASSES_PER_TASK=10",
                  s20 - s10 == {f"JOB_NAME=d2_{_tok}_inr_t20_s0", "BASE_OUT=./outputs/exp_d2_t20", "NUM_TASKS=20",
                                "CLASSES_PER_TASK=10"}
                  and s10 - s20 == {f"JOB_NAME=d2_{_tok}_inr_s0", "BASE_OUT=./outputs/exp_d2", "NUM_TASKS=10"},
                  f"+{sorted(s20 - s10)} -{sorted(s10 - s20)}")

        # (g) reject unregistered usage. The needles match messages printed by run_d2.bash; the
        # non-ASCII ones are kept verbatim because they must match that script's output.
        for c, why, needle in (
                (dict(DATASET="c100", NTASKS="20", STAGE="ENDPOINT", ONLY="all_shared"), "NTASKS=20 on c100", "only supported for DATASET=inr"),
                (dict(DATASET="cub", NTASKS="20", STAGE="ENDPOINT", ONLY="all_shared"), "NTASKS=20 on cub", "only supported for DATASET=inr"),
                (dict(DATASET="c100", NTASKS="15", STAGE="ENDPOINT", ONLY="all_shared"), "NTASKS=15 on c100", "is not supported: only 10"),
                (dict(DATASET="inr", NTASKS="15", STAGE="ENDPOINT", ONLY="all_shared"), "NTASKS=15 on inr", "is not supported: only 10"),
                (dict(DATASET="inr", NTASKS="5", STAGE="GRID", GRID_LS="2"), "NTASKS=5 on inr", "is not supported: only 10"),
                (dict(DATASET="inr", NTASKS="20", STAGE="ENDPOINT"), "T=20 ENDPOINT without ONLY (would start random_split)", "random_split"),
                (dict(DATASET="inr", NTASKS="20", STAGE="GRID", GRID_LS="2 4 8"), "T=20 GRID with the unregistered l=8", "fixed_depth_l8"),
                (dict(DATASET="inr", NTASKS="20", STAGE="GRID"), "T=20 GRID with the default l list 2..10", "fixed_depth_l8"),
                (dict(DATASET="inr", NTASKS="20", STAGE="BESTAPPROX", BA_KS="4"), "T=20 BESTAPPROX with the default K list (includes k4f1)", "best_approximation_k4f1"),
                (dict(DATASET="inr", NTASKS="20", STAGE="PINNED", PIN_SETS="deep"), "T=20 STAGE=PINNED", "STAGE=PINNED"),
                (dict(DATASET="inr", NTASKS="20", STAGE="ENDPOINT", ONLY="all_shared", BK="ibot"), "T=20 with BK=ibot", "BK=ibot"),
                (dict(DATASET="inr", NTASKS="20", STAGE="ENDPOINT", ONLY="all_shared", BASE_OUT="./outputs/exp_d2"), "T=20 BASE_OUT pointing at the T=10 root", "points at the T=10 output root"),
                (dict(DATASET="inr", STAGE="ENDPOINT", ONLY="all_shared", BASE_OUT="./outputs/exp_d2_t20/"), "T=10 BASE_OUT pointing at the T=20 root", "points at the T=20 output root"),
                (dict(DATASET="inr", NTASKS="20", STAGE="ENDPOINT", ONLY="all_shared", CLASSES_PER_TASK="10"), "caller-set CLASSES_PER_TASK", "callers may not set it")):
            env = dict(BK="augreg", SEEDS="0")
            env.update(c)
            rc, out, runs, _ = _dry(**env)
            check(f"t12.G rejected: {why} -> nonzero exit, no run started, reason printed", rc != 0 and runs == [] and needle in out,
                  f"rc={rc} runs={len(runs)}")
    except FileNotFoundError as e:
        if shutil.which("bash") is None:
            _skip("t12.A/B/G run_d2.bash / exp_timm.bash dry runs", BASH_REASON)
        else:
            check("t12 dry-run prerequisites present", False, f"{e}")
    finally:
        _sh.rmtree(d, ignore_errors=True)


TESTS = (t1_allocation, t2_mode_dispatch, t3_identity_roundtrip, t4_role_semantics, t5_random_split,
         t6_gate_learner_integration, t8_window_arms, t9_best_approximation, t10_dataset_wiring,
         t11_pinned, t12_t20)


# ---------------------------------------------------------------------------
# pytest entry points: each runs one tN function and fails if it recorded a failed check.
# Blocks inside a function that need an unshipped input are recorded as skips and printed;
# functions that need such an input as a whole are skipped up front.
def _run_as_test(fn, *requirements):
    import pytest
    for req in requirements:
        reason = req()
        if reason:
            pytest.skip(reason)
    n = len(_fails)
    fn()
    new = _fails[n:]
    assert not new, f"{len(new)} failed check(s): {new}"


def test_t1_allocation():
    _run_as_test(t1_allocation, _need_npz)


def test_t2_mode_dispatch():
    _run_as_test(t2_mode_dispatch)


def test_t3_identity_roundtrip():
    _run_as_test(t3_identity_roundtrip, _need_runs)


def test_t4_role_semantics():
    _run_as_test(t4_role_semantics)


def test_t5_random_split():
    _run_as_test(t5_random_split)


def test_t6_gate_learner_integration():
    _run_as_test(t6_gate_learner_integration, _need_npz, _need_upstream)


def test_t8_window_arms():
    _run_as_test(t8_window_arms)


def test_t9_best_approximation():
    _run_as_test(t9_best_approximation)


def test_t10_dataset_wiring():
    _run_as_test(t10_dataset_wiring)


def test_t11_pinned():
    _run_as_test(t11_pinned)


def test_t12_t20():
    _run_as_test(t12_t20)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for _t in TESTS:
        _t()
    print()
    if _skips:
        print(f"{len(_skips)} block(s) skipped:")
        for _name, _reason in _skips:
            print(f"  - {_name}: {_reason}")
    if _fails:
        print(f"✗ {len(_fails)} check(s) failed:")
        for f in _fails:
            print(f"  - {f}")
        sys.exit(1)
    print("✓ all non-skipped checks passed")
