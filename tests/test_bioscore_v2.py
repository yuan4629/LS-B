# -*- coding: utf-8 -*-
"""CPU smoke tests for the pure-math part of BioScore v2 (no GPU, atlas or fMRI data needed).

Run from the repository root:  python tests/test_bioscore_v2.py   (or: pytest tests/test_bioscore_v2.py)
Note: this file must not live in model_m/common/: Python puts the script's directory at
sys.path[0], and the selectors.py there would shadow the standard-library selectors module
and break the socket -> torch import chain.

Each check corresponds to a claim of the design; a failure means the claim is not
implemented, not that the test is broken.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402

from model_m.common.bioscore_v2 import (  # noqa: E402
    diagnose_verdict,
    identify_tasks,
    interaction_energy,
    selection_reproducibility,
    reachable_topk_sets,
    select_layers_submodular,
    two_way_center,
)

OK, FAIL = "  ✓", "  ✗"
_fails = []


def check(name, cond, extra=""):
    print((OK if cond else FAIL) + f" {name}" + (f"  [{extra}]" if extra else ""))
    if not cond:
        _fails.append(name)


def t_centering():
    print("\n[1] Two-way centering: main effects removed, only the interaction remains")
    rng = np.random.RandomState(0)
    a, b = rng.rand(12), rng.rand(12)
    D_add = a[:, None] + b[None, :]                      # purely additive
    Dt, _ = two_way_center(D_add)
    check("additive matrix is ~0 after centering", np.abs(Dt).max() < 1e-10, f"max={np.abs(Dt).max():.2e}")
    check("interaction energy of an additive matrix ~ 0", interaction_energy(D_add) < 1e-12,
          f"E={interaction_energy(D_add):.2e}")

    D = D_add + rng.randn(12, 12) * 0.5
    Dt, _ = two_way_center(D)
    check("general matrix: row/column means ~ 0 after centering",
          max(np.abs(Dt.mean(0)).max(), np.abs(Dt.mean(1)).max()) < 1e-10)
    check("interaction energy > 0.1 when an interaction is present", interaction_energy(D) > 0.1,
          f"E={interaction_energy(D):.3f}")

    # An outer product (the v1 structure) is NOT zero after centering -- guards against
    # mistaking this diagnostic for a rank test.
    # Closed form: D = a b^T => D~ = (a - mean(a))(b - mean(b))^T => E = |a~|^2 |b~|^2 / |D - mean(D)|^2.
    # Asserting the closed form (not an arbitrary threshold) tests the implementation itself.
    D_out = np.outer(a, b)
    at, bt = a - a.mean(), b - b.mean()
    e_ana = (at @ at) * (bt @ bt) / ((D_out - D_out.mean()) ** 2).sum()
    e_num = interaction_energy(D_out)
    check("outer-product interaction energy = closed form and > 0 (so not a rank test)",
          e_num > 1e-3 and abs(e_num - e_ana) < 1e-10, f"num={e_num:.4f} ana={e_ana:.4f}")
    # When the mean dominates the variation (typical of a deep-heavy atlas) the value is small,
    # consistent with "main effects dominate".
    check("outer product with a dominant mean: interaction energy < 0.2", e_num < 0.2, f"E={e_num:.4f}")


def t_reachable():
    print("\n[2] v1 reachable top-k sets: a deep-heavy atlas structurally cannot leave a fixed selection")
    # Simulated atlas: every ROI's layer profile leans deep, only the strength differs (roughly proportional shapes)
    L = 12
    base = np.exp(np.linspace(-2.0, 1.5, L))
    rng = np.random.RandomState(1)
    A_deep = np.stack([base * (0.8 + 0.4 * rng.rand()) for _ in range(12)])
    n, top = reachable_topk_sets(A_deep, k=4, n_samples=4000, seed=0)
    check("roughly proportional atlas: reachable top-4 sets <= 3", n <= 3, f"n={n}")
    check("and the most frequent set has share > 0.9", top[0][1] > 0.9, f"{top[0][0]} {top[0][1]:.3f}")

    # Control: ROI peaks at different layers (a real hierarchy) -> many more reachable sets
    A_div = np.stack([np.exp(-0.5 * ((np.arange(L) - p) / 1.5) ** 2)
                      for p in np.linspace(0, L - 1, 12)])
    n2, _ = reachable_topk_sets(A_div, k=4, n_samples=4000, seed=0)
    check("atlas with spread peaks: clearly more reachable sets", n2 > 3 * max(n, 1), f"n={n2} vs {n}")


def t_submodular_diversity():
    print("\n[3] Submodular coverage: avoid redundant adjacent layers, spread across the hierarchy")
    L, R = 12, 6
    # 6 ROIs peak at layers 0/2/4/7/9/11; layers 8, 9, 10 are highly similar (redundant cluster)
    peaks = [0, 2, 4, 7, 9, 11]
    D = np.stack([np.exp(-0.5 * ((np.arange(L) - p) / 1.2) ** 2) for p in peaks])
    D[:, 8:11] = D[:, 9][:, None]        # make layers 8/9/10 fully redundant
    Dt, main = two_way_center(D)

    sel, gains = select_layers_submodular(Dt, k=4)
    print(f"      submodular selection = {sel}")
    top4 = sorted(np.argsort(-Dt.mean(0))[:4].tolist())
    print(f"      top-4 of the same matrix (independent scores) = {top4}")
    dup = sum(1 for x in (8, 9, 10) if x in sel)
    check("redundant cluster {8,9,10}: at most 1 layer selected", dup <= 1, f"{dup} selected")
    check("span of the selected layers >= 6", max(sel) - min(sel) >= 6, f"span={max(sel) - min(sel)}")
    adj = sum(1 for a, b in zip(sel, sel[1:]) if b - a == 1)
    check("adjacent layer pairs <= 1", adj <= 1, f"adj={adj}")
    check("returns k layers", len(sel) == 4)
    check("len(gains) = L and every selected layer has gain > 0",
          len(gains) == L and all(gains[l] > 0 for l in sel))


def t_task_variation():
    print("\n[4] Task conditioning: D changes -> selection changes; interference penalty -> staggered across tasks")
    L, R = 12, 6
    rng = np.random.RandomState(7)
    base = np.stack([np.exp(-0.5 * ((np.arange(L) - p) / 1.5) ** 2)
                     for p in [0, 2, 4, 7, 9, 11]])
    sels = set()
    for t in range(6):                                  # 6 "tasks" with different ROI involvement
        w = rng.dirichlet(np.ones(R))
        D = base * w[:, None] * 3.0 + rng.randn(R, L) * 0.05
        Dt, _ = two_way_center(D)
        s, _ = select_layers_submodular(Dt, k=4, roi_w=w)
        sels.add(tuple(s))
    check("6 tasks give >= 3 distinct selections", len(sels) >= 3, f"{len(sels)} distinct: {sorted(sels)}")

    # Interference penalty pushes down layers used by the previous task (same D, same ROI profile -> only gamma acts)
    D = base * 3.0
    Dt, _ = two_way_center(D)
    w = np.ones(R) / R
    s0, _ = select_layers_submodular(Dt, k=3, roi_w=w)
    s1, _ = select_layers_submodular(Dt, k=3, roi_w=w, gamma=5.0,
                                     history=[(s0, w)], cur_w=w)
    print(f"      task 0 selects {s0} -> task 1 (gamma=5) selects {s1}")
    check("gamma>0: the new task avoids layers used by the old task", set(s0) != set(s1))
    s1b, _ = select_layers_submodular(Dt, k=3, roi_w=w, gamma=0.0,
                                      history=[(s0, w)], cur_w=w)
    check("gamma=0: falls back to task 0's selection (the penalty is controlled by gamma)", set(s1b) == set(s0))


def t_invariance():
    print("\n[5] Invariance: task-irrelevant components must not change the selection")
    L, R = 12, 6
    rng = np.random.RandomState(3)
    raw = np.stack([np.exp(-0.5 * ((np.arange(L) - p) / 1.5) ** 2)
                    for p in [0, 2, 4, 7, 9, 11]])

    def pipeline(D):
        Dt, _ = two_way_center(D)
        return select_layers_submodular(Dt, k=4)[0]

    s_ref = pipeline(raw)
    # Contract 1: global scaling of D~ does not change the selection (the objective is positively
    # homogeneous of degree 1). v1 failed exactly because per-layer scale dominated its ranking.
    Dt, _ = two_way_center(raw)
    check("global scaling of D~ does not change the selection",
          select_layers_submodular(Dt * 137.0, k=4)[0] == s_ref)
    # Note: "D~ is shift-invariant" is deliberately not tested: after two-way centering the grand mean
    # of D~ is always 0, so a shifted D~ is not a valid input, and relu(D~) (preference above its own
    # mean) relies on that zero mean. What must hold is that the whole pipeline ignores
    # task-irrelevant components -- the three checks below.

    # Contract 2: any per-layer task-irrelevant additive preference ("deep layers score higher") is removed by centering
    per_layer = rng.randn(L) * 5.0
    check("per-layer additive preference does not change the selection", pipeline(raw + per_layer[None, :]) == s_ref,
          f"{pipeline(raw + per_layer[None, :])} vs {s_ref}")
    # Contract 3: per-ROI overall strength ("V1 naturally responds more") is removed as well
    per_roi = rng.randn(R) * 5.0
    check("per-ROI additive strength does not change the selection", pipeline(raw + per_roi[:, None]) == s_ref)
    # Contract 4: a global shift (different reference baseline) is removed
    check("global shift does not change the selection", pipeline(raw + 9.0) == s_ref)

    # Negative control: a real interaction change MUST change the selection; otherwise the three
    # checks above would only show insensitivity to everything.
    perturbed = raw.copy()
    perturbed[0] = np.exp(-0.5 * ((np.arange(L) - 11) / 1.5) ** 2)   # ROI0 peak moves 0 -> 11
    check("an interaction change does change the selection", pipeline(perturbed) != s_ref,
          f"{pipeline(perturbed)} vs {s_ref}")


def t_diag_decision_rule():
    """[7] Can the decision rule of atlas_v2_diag.py tell "task signal present" from "pure noise"?

    The rule drives a stop/continue decision, so it must separate the two worlds on synthetic
    data. Three worlds, each read with the same pipeline as the real one
    (two-way centering -> task centering -> split-half reproducibility):

      W1 domain shift dominates + real task signal: shared component x50, task signal identical in both halves
      W2 domain shift dominates + pure noise: task residuals redrawn independently for each half
      W3 no shared component: sanity check, the task signal is directly visible

    Two metrics that would wrongly reject a real signal (keep these assertions):
      - exact position-wise agreement is too strict: with a planted r=0.80 signal (W1) it is only 2/10.
      - the number of distinct selections does not separate the worlds: noise is also 10/10
        distinct after task centering.
    -> The primary criterion must be task identification (R x L cells per task, enough power),
       not distinct counts and not exact agreement.
    """
    print("\n[7] Decision rule: separates 'task signal drowned out' from 'residual is sampling noise'")
    rng = np.random.RandomState(7)
    R, L, T, k = 12, 12, 10, 4

    def pipeline(D_half_a, D_half_b):
        A = np.stack([two_way_center(d)[0] for d in D_half_a])
        B = np.stack([two_way_center(d)[0] for d in D_half_b])
        dA, dB = A - A.mean(0), B - B.mean(0)
        selA = [select_layers_submodular(dA[t], k)[0] for t in range(T)]
        selB = [select_layers_submodular(dB[t], k)[0] for t in range(T)]
        hits, _T, p_id, _rm, _rx = identify_tasks(dA, dB)
        j_match, j_mis, p_rep, exact = selection_reproducibility(selA, selB)
        return dict(distinct=len({tuple(s) for s in selA}), hits=hits, p_id=p_id,
                    p_rep=p_rep, exact=exact, j_match=j_match, j_mis=j_mis)

    def _v(w):
        return diagnose_verdict(w["p_id"], w["p_rep"], w["j_match"], w["j_mis"])

    shared = rng.randn(R, L) * 50.0                      # domain shift (e.g. CIFAR vs NSD): shared by all tasks
    sig = rng.randn(T, R, L)                             # real task signal: identical in both halves
    a1 = shared + sig + rng.randn(T, R, L) * 0.5
    b1 = shared + sig + rng.randn(T, R, L) * 0.5
    w1 = pipeline(a1, b1)
    # Control without task centering: fully dominated by the shared component -> selection nearly task-invariant
    raw = [select_layers_submodular(two_way_center(a1[t])[0], k)[0] for t in range(T)]
    check("W1 without task centering => nearly invariant across tasks", len({tuple(s) for s in raw}) <= 2,
          f"distinct={len({tuple(s) for s in raw})}/{T}")
    check("W1 primary test says signal", w1["hits"] >= 8 and w1["p_id"] < 0.05,
          f"identified {w1['hits']}/{T} p={w1['p_id']:.1e}")
    check("W1 combined verdict = proceed", _v(w1) == "proceed",
          f"{_v(w1)}  J {w1['j_match']:.2f}vs{w1['j_mis']:.2f} p={w1['p_rep']:.3f}")

    # W2: residuals redrawn independently for each half = sampling noise
    w2 = pipeline(shared + rng.randn(T, R, L), shared + rng.randn(T, R, L))
    check("W2 primary test says no signal", w2["hits"] <= 3 and w2["p_id"] > 0.05,
          f"identified {w2['hits']}/{T} p={w2['p_id']:.2f}")
    # Key point: the secondary test on W2 gives p=0.089 < alpha_repro=0.10, a false positive on its own.
    # The combined verdict must still be stop -- this is why the primary test decides first;
    # reversing the order would proceed on noise.
    check("W2 combined verdict = stop (despite the secondary false positive)", _v(w2) == "stop",
          f"{_v(w2)}  secondary p={w2['p_rep']:.3f} (<0.10, false positive)")

    # Middle zone: signal present but buried in noise -> neither stop nor proceed
    wm = pipeline(shared + sig + rng.randn(T, R, L) * 1.5,
                  shared + sig + rng.randn(T, R, L) * 1.5)
    check("middle zone combined verdict = more_data (neither stop nor proceed)", _v(wm) == "more_data",
          f"{_v(wm)}  identified {wm['hits']}/{T} selection reproducibility p={wm['p_rep']:.3f}")

    # Two metrics that look usable but are not -- kept as counter-examples so the rule is not reverted
    check("counter-example: distinct count cannot separate the worlds (noise 'varies' too)", w1["distinct"] >= 5 and w2["distinct"] >= 5,
          f"W1={w1['distinct']} W2={w2['distinct']}")
    check("counter-example: exact agreement would reject a real signal", w1["exact"] <= (T * 2) // 3,
          f"W1 has a real signal but exact agreement is only {w1['exact']}/{T}")

    # W3: sanity -- without a shared component the task signal is visible without centering
    a3 = sig + rng.randn(T, R, L) * 0.5
    raw3 = [select_layers_submodular(two_way_center(a3[t])[0], k)[0] for t in range(T)]
    check("W3 without domain shift: selection varies across tasks anyway", len({tuple(s) for s in raw3}) >= 5,
          f"distinct={len({tuple(s) for s in raw3})}/{T}")


def t_selector_rng_hygiene():
    """[10] Selection sampling must be 1) reproducible and 2) leave the global RNGs untouched.

    1) The selector reads a train-mode loader (RandomResizedCrop + HFlip) whose augmentation draws
       from the global RNG; without its own seed, two runs with the same atlas and code gave D~
       differing by 1.4-1.8% (relative) and per-task selections agreeing in only 6-7/10 tasks.
    2) If the selector consumes global random numbers, the later training trajectory diverges
       (runs with identical per-task selections still differed by ~0.42pp ACC). Restoring the
       state makes the selector transparent to training.
    """
    print("\n[10] Selection sampling: deterministic and leaves the global RNG untouched")
    import random as _r

    import torch as _t

    from model_m.common.bioscore_v2 import BioScoreV2Calculator

    class _Args:
        seed = 0

    class _Holder:
        args = _Args()

    cb = BioScoreV2Calculator.collect_batches.__get__(_Holder(), _Holder)

    class _AugLoader:
        """Mimics a train loader with random augmentation: batch content comes from the **global** RNG."""

        def __iter__(self):
            for _ in range(12):
                yield (_t.rand(2, 3), _t.zeros(2))

    # Negative control first: show this loader really depends on the global RNG, otherwise 1) tests nothing
    _t.manual_seed(1)
    r1 = next(iter(_AugLoader()))[0]
    _t.manual_seed(2)
    r2 = next(iter(_AugLoader()))[0]
    check("counter-example: this loader depends on the global RNG (assertion is non-trivial)", not _t.equal(r1, r2))

    # 1) Determinism: different global states on entry must yield the same data
    _t.manual_seed(1234)
    a = cb(_AugLoader(), 10)
    _t.manual_seed(9999)
    b = cb(_AugLoader(), 10)
    check("same batches under different global RNG states (reproducible)",
          len(a) == len(b) == 10 and all(_t.equal(x, y) for x, y in zip(a, b)))

    # 2) No side effects: all three global RNG states must be bit-identical before and after the call
    _t.manual_seed(4321)
    bt, bnp, bpy = _t.get_rng_state(), np.random.get_state(), _r.getstate()
    cb(_AugLoader(), 10)
    anp = np.random.get_state()
    check("torch global RNG unchanged", _t.equal(bt, _t.get_rng_state()))
    check("numpy global RNG unchanged",
          np.array_equal(bnp[1], anp[1]) and bnp[2] == anp[2])
    check("python random unchanged", bpy == _r.getstate())


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 64)
    print("BioScore v2 pure-math smoke tests")
    print("=" * 64)
    t_centering()
    t_reachable()
    t_submodular_diversity()
    t_task_variation()
    t_invariance()
    t_diag_decision_rule()
    t_selector_rng_hygiene()
    print("\n" + "=" * 64)
    if _fails:
        print(f"FAIL: {len(_fails)} checks failed: {_fails}")
        sys.exit(1)
    print("ALL PASSED")


# ---- pytest entry points: each runs one t_* group and fails if it recorded a failed check ----
def _run_group(fn):
    n0 = len(_fails)
    fn()
    new = _fails[n0:]
    assert not new, f"{fn.__name__}: {len(new)} failed checks: {new}"


def test_centering():
    _run_group(t_centering)


def test_reachable():
    _run_group(t_reachable)


def test_submodular_diversity():
    _run_group(t_submodular_diversity)


def test_task_variation():
    _run_group(t_task_variation)


def test_invariance():
    _run_group(t_invariance)


def test_diag_decision_rule():
    _run_group(t_diag_decision_rule)


def test_selector_rng_hygiene():
    _run_group(t_selector_rng_hygiene)


if __name__ == "__main__":
    main()
