# -*- coding: utf-8 -*-
"""analyze_d2_split.py -- offline readouts and allocation generator for D2 (brain-grounded
shared/specific layer allocation).

This module is the single reference implementation of the allocation: other code (the D2 host,
tests) imports allocate() rather than re-implementing it. Printed readouts: P2.1 = Spearman of
m / rho with depth, P2.2 = cross-backbone Spearman of m, P2.4 = split-half reliability.

Algorithm (rho criterion):
  D~_t = two_way_center(D_t);  C = mean_t D~_t;  E_t = D~_t - C
  m[l] = (1/T) sum_t sum_r E_t[r,l]^2   (task-residual mass)
  s[l] = sum_r C[r,l]^2                 (shared mass)
  rho[l] = m[l]/s[l]; top-k_specific layers by descending rho -> task-specific pool, rest -> shared pool
  Causal deployment: use only the first K tasks' D_t (main arm K=3).

Usage:
  python analyze_d2_split.py                      # all readouts + per-backbone allocation + K stability
  python analyze_d2_split.py --npz <path>         # other data source (e.g. the IN-R diag npz)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Non-ASCII output raises UnicodeEncodeError on cp936 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_m.common.bioscore_v2 import two_way_center  # noqa: E402

K_SPECIFIC = 4      # number of task-specific layers
K_FREEZE = 3        # tasks used by the causal allocation (with K=3, augreg/ibot match the oracle, J=1.00)


def _center(D):
    out = two_way_center(D)
    return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float64)


def layer_stats(D_stack):
    """D_stack [T,R,L] -> (m[L], s[L], rho[L]); see the module docstring."""
    Dt = np.stack([_center(D_stack[t]) for t in range(D_stack.shape[0])])
    C = Dt.mean(axis=0)
    E = Dt - C
    m = (E ** 2).sum(axis=1).mean(axis=0)
    s = (C ** 2).sum(axis=0)
    rho = m / (s + 1e-12)
    return m, s, rho


def allocate(D_stack, k_specific=K_SPECIFIC, k_freeze=None):
    """Return (specific_layers sorted list, shared_layers sorted list, rho).
    k_freeze=None -> oracle (all tasks); k_freeze=K -> causal (first K tasks only)."""
    D = D_stack if k_freeze is None else D_stack[:k_freeze]
    _, _, rho = layer_stats(D)
    spec = allocate_from_scores(rho, k_specific)
    shared = sorted(int(x) for x in range(len(rho)) if x not in set(spec))
    return spec, shared, rho


def allocate_from_scores(score, k_specific=K_SPECIFIC):
    """Per-layer scores -> specific layer set (sorted list). Shared top-k for the rho arm and the
    file-based score control arms.

    Polarity: descending top-k -> task-specific pool. It is a separate function (also called by
    allocate()) so that a polarity inversion (e.g. "top-4 = shared", which would make a control arm
    a guaranteed-to-lose strawman) would flip the main arm too and be caught by the existing tests.
    """
    sc = np.asarray(score, dtype=np.float64).ravel()
    k = int(k_specific)
    if not 0 <= k <= sc.size:
        raise ValueError(f"k_specific={k} exceeds score length {sc.size}")
    if not np.all(np.isfinite(sc)):
        raise ValueError("score vector contains nan/inf; refusing to allocate on invalid scores")
    order = np.argsort(-sc)
    return sorted(int(x) for x in order[:k])


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def jaccard(x, y):
    x, y = set(x), set(y)
    return len(x & y) / len(x | y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/atlas_v2_diag_k4_b10.npz")
    args = ap.parse_args()
    npz = np.load(args.npz)
    backbones = sorted({k.split("_")[0] for k in npz.files if k.endswith("_D_full")})
    if not backbones:
        raise SystemExit(f"{args.npz} has no *_D_full keys: not a diag output, or the key names changed (refusing to continue)")

    depth = np.arange(12)
    m_all = {}
    print(f"=== D2 offline readouts ({args.npz}; rho criterion) ===")
    for bk in backbones:
        D = npz[f"{bk}_D_full"]
        m, s, rho = layer_stats(D)
        m_all[bk] = m
        spec_o, shared_o, _ = allocate(D)
        print(f"\n[{bk}] T={D.shape[0]}")
        print("  m%  = " + " ".join(f"{100*x/m.sum():5.1f}" for x in m))
        print("  rho = " + " ".join(f"{x:7.4f}" for x in rho))
        print(f"  P2.1 Spearman(m,depth)={spearman(m, depth):+.3f}  Spearman(rho,depth)={spearman(rho, depth):+.3f}")
        print(f"  oracle: specific={spec_o} shared={shared_o}")
        # split-half reliability (if the npz contains A/B halves)
        ka, kb = f"{bk}_D_A", f"{bk}_D_B"
        if ka in npz.files and kb in npz.files:
            mA, _, rA = layer_stats(npz[ka])
            mB, _, rB = layer_stats(npz[kb])
            sA, _, _ = allocate(npz[ka])
            sB, _, _ = allocate(npz[kb])
            print(f"  P2.4 split-half: Pearson(mA,mB)={pearson(mA,mB):.3f}  Spearman(rhoA,rhoB)={spearman(rA,rB):.3f}  top4 overlap={len(set(sA)&set(sB))}/4")
        # stability of the causal allocation w.r.t. K
        stab = []
        for K in (2, 3, 4, 5):
            sK, _, _ = allocate(D, k_freeze=K)
            stab.append(f"K={K}:{sK} J={jaccard(sK, spec_o):.2f}")
        print("  K-freeze: " + " | ".join(stab))
        print(f"  -> main-arm allocation (causal, K={K_FREEZE}): specific={allocate(D, k_freeze=K_FREEZE)[0]}")

    if len(backbones) >= 2:
        print("\nP2.2 cross-backbone Spearman of m:")
        for i, a in enumerate(backbones):
            for b in backbones[i + 1:]:
                print(f"  {a}-{b}: {spearman(m_all[a], m_all[b]):+.3f}")


if __name__ == "__main__":
    main()
