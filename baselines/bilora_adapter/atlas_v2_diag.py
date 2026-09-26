# -*- coding: utf-8 -*-
"""BioScore v2 diagnostic: is the per-task residual delta_t signal or sampling noise? (forward only)

## Why this is needed

With two-way (ROI, layer) centering the interaction energy of D~_t is clearly non-zero, yet almost
every task selects the same layer set:

    D~_t ~ D~_shared + delta_t ,   ||delta_t|| << ||D~_shared||

The z-score reference pool is NSD, while the task images come from the CL dataset:

    z_t[n,l] = ( s_t[n,l] - mu_ref[n,l] ) / ( sd_ref[n,l] + eps )

sd_ref is the std across 10 NSD sub-pools, i.e. the sampling standard error of the estimator
(1280 images per pool -> ~2% relative), while the numerator contains the CL-data <-> NSD domain
shift (tens of percent). The domain shift therefore spans tens of sigmas, whereas tasks (same image
statistics, different class subsets) differ by ~1 sigma: SNR ~1:50. Two-way centering removes the
ROI and layer main effects but not the ROI x layer interaction of the domain shift itself, so most
of the measured interaction energy is domain shift, not task interaction. A task-agnostic
ROI x layer component can drive the selection just like a task-agnostic per-layer factor can.

## Why not apply the fix directly

The obvious fix is a third centering axis (task): D^_t = D~_t - mean_{t'<t} D~_{t'} (causal,
CL-legal, on the [12,12] matrix, no extra forward passes). But if delta_t is sampling noise,
task centering makes every task pick a distinct set while the selections are pure noise.
So this script first tests whether delta_t is reproducible.

## Method

Each task's batches are split (interleaved) into disjoint halves A/B. Moments are accumulated so
that full-sample moments = A moments + B moments exactly (std is non-linear, so stds are never
averaged). One forward pass thus yields A-half, B-half and full-sample estimates. Then:

  1) three-way variance decomposition: tau = between-task / (between-task + shared)
  2) primary test, task identification: identify each B-half task from the A-half residuals
     (144 cells per task, null = Binomial(1/T)). Answers whether task information exists.
     Synthetic calibration: split-half correlation as low as 0.31 still gives 10/10 hits.
  3) selection after task centering (oracle global mean = ceiling; causal running mean =
     deployable)
  4) secondary test, selection reproducibility: Jaccard between the sets picked from the two
     independent halves vs a mismatched-permutation null. Answers whether the information is
     stable enough to drive selection; a different question from 2), judged separately.
  5) numerical health per layer: sd_ref / |z| / column mass of relu(D~). Distinguishes
     "layer 0 is naturally the most distinct" from "sd_ref collapsed on a ~1e-6-scale layer and
     z blew up" (min(S) = 0 is otherwise very common).

The verdict has three branches rather than two: synthetic calibration (tests/test_bioscore_v2.py
[7]) shows a real middle zone where task information clearly exists (identification 10/10) but the
SNR is too low for the 4-of-12 greedy to converge (reproducibility ~ chance). There the right
action is to raise the SNR, neither stopping nor task-centering.

Two pitfalls found by the synthetic calibration:
  - exact position-wise agreement is too strict: a planted r=0.80 signal still agrees exactly in
    only 2/10 tasks. Use Jaccard only.
  - the reproducibility null must not be uniform random k-subsets: a method that prefers a few
    layers lowers that baseline and turns noise into "signal". Use a task-label permutation
    (keeps the marginal selection distribution of both halves).

Note: with BATCHES != 10 the cached atlas ref_mu/ref_sd still come from 10-batch sub-pools and are
not recomputed. This does not affect the diagnostic: task centering D~_t - mean_t D~ cancels the
mu_ref term and sd_ref only acts as a per-(n,l) weight; more task-side samples shrink the noise by
sqrt(n) while the signal is unchanged, which is exactly the comparison "raise BATCHES" needs.

Run:  python baselines/bilora_adapter/atlas_v2_diag.py
      ONLY=augreg  K=2  BATCHES=20  python ...
"""
import itertools
import json
import os
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from baselines.bilora_adapter.atlas_v2_export import (  # noqa: E402
    BACKBONES, _Strip, _args, _jaccard, _task_loaders, atlas_signature,
)
from model_m.common.bioscore_v2 import (  # noqa: E402
    diagnose_verdict, identify_tasks, interaction_energy, pearson as _pearson,
    select_layers_submodular, selection_reproducibility,
    spearman_brown as _spearman_brown, two_way_center,
)

NUM_LAYERS = 12


def _sel(Dc, k):
    """Submodular greedy on an already-centered matrix (gamma=0: only the information in D, no interference penalty)."""
    return select_layers_submodular(Dc, k)[0]


def collect(name, bk, device, k, n_batches):
    """One forward pass: per-task raw [R,L] matrices (D_full, D_A, D_B) plus per-layer z statistics."""
    from model_m.common.bioscore_v2 import get_bioscore_v2
    from model_m.common.timm_backbone import ModifiedTimmViT

    args = _args(bk)
    args.selector_batches = n_batches
    if not os.path.exists(args.timm_weights):
        print(f"[G1b:{name}] ✗ weights missing {args.timm_weights}, skipping.")
        return None
    backbone = ModifiedTimmViT(model_name=args.timm_model, weights_file=args.timm_weights,
                               pretrained=True, weights_format=args.timm_weights_format).to(device)
    calc = get_bioscore_v2(args, device, backbone)
    loaders = _task_loaders(args, args.num_tasks)

    full, ha, hb, zstat = [], [], [], []
    for t, ld in enumerate(loaders):
        batches = calc.collect_batches(_Strip(ld), n_batches)
        if len(batches) < 4:
            raise RuntimeError(f"[G1b:{name}] task {t} has only {len(batches)} batches; split-half is meaningless.")
        # Interleaved rather than first/second-half split: more robust to residual ordering effects.
        nA, s1A, s2A = calc.pool_moments(batches[0::2])
        nB, s1B, s2B = calc.pool_moments(batches[1::2])
        sA = calc.std_from_moments(nA, s1A, s2A)
        sB = calc.std_from_moments(nB, s1B, s2B)
        sF = calc.std_from_moments(nA + nB, s1A + s1B, s2A + s2B)   # exactly the full-sample std
        full.append(calc.aggregate(sF)[0])
        ha.append(calc.aggregate(sA)[0])
        hb.append(calc.aggregate(sB)[0])
        z = ((sF - calc.ref_mu) / (calc.ref_sd + 1e-8)).abs()
        zstat.append(z.median(dim=0).values.cpu().numpy())
        print(f"[G1b:{name}] task {t}: n={nA + nB} (A={nA}/B={nB}) "
              f"interaction_energy={interaction_energy(full[-1]):.3f}")

    ref_mu = calc.ref_mu.cpu().numpy()
    ref_sd = calc.ref_sd.cpu().numpy()
    return dict(name=name, k=k, D_full=np.stack(full), D_A=np.stack(ha), D_B=np.stack(hb),
                z_med=np.stack(zstat), ref_mu=ref_mu, ref_sd=ref_sd,
                roi_names=list(calc.roi_names), n_img=nA + nB,
                # Compare with the atlas_v2_export.py dump: a different fingerprint means a different
                # atlas, and per-task selections must not be cross-referenced between the two.
                atlas_sig=atlas_signature(calc))


def analyse(c):
    """Pure numpy, no GPU. Prints and returns a JSON-serialisable result dict."""
    name, k = c["name"], c["k"]
    T = c["D_full"].shape[0]
    Dt = np.stack([two_way_center(D)[0] for D in c["D_full"]])          # [T,R,L]
    Dbar = Dt.mean(0)
    common = float((Dbar ** 2).sum())
    between = float(((Dt - Dbar) ** 2).sum(axis=(1, 2)).mean())
    tau = between / (between + common + 1e-30)

    print(f"\n--- [{name}] 1) three-way variance decomposition (D~ two-way centered) ---")
    print(f"  shared component ||Dbar||^2          = {common:.4g}")
    print(f"  between-task     mean_t||D~_t-Dbar||^2 = {between:.4g}")
    print(f"  tau = between/(between+shared)       = {tau:.4f}   <- fraction of D~ that is task signal")

    # 2) Split-half reliability + task identification (primary test): apply the same oracle
    #    task centering to each half, then compare residuals.
    A = np.stack([two_way_center(D)[0] for D in c["D_A"]])
    B = np.stack([two_way_center(D)[0] for D in c["D_B"]])
    dA, dB = A - A.mean(0), B - B.mean(0)
    r = _pearson(dA, dB)
    r_sb = 2 * r / (1 + r) if r > -1 else -1.0                          # Spearman-Brown -> full-sample reliability
    per_task = [_pearson(dA[t], dB[t]) for t in range(T)]
    hits, nT, p_id, r_mat, r_mis = identify_tasks(dA, dB)
    print(f"\n--- [{name}] 2) is the task residual delta_t signal or noise? ---")
    print(f"  [primary] task identification, A half -> B half = {hits}/{nT}  (chance 1/{nT}, binomial p={p_id:.2e})")
    print(f"            matched corr {r_mat:.3f} vs mismatched {r_mis:.3f}")
    print(f"  split-half reliability corr(delta^A,delta^B) = {r:.3f}   Spearman-Brown full-sample = {r_sb:.3f}")
    print(f"  per-task corr = {[round(x, 2) for x in per_task]}")
    if 0 < r < 0.9:
        need = next((n for n in (2, 4, 8, 16, 32) if _spearman_brown(r, n) >= 0.8), None)
        print(f"  reaching full-sample reliability 0.8 needs about {need}x the current batch count"
              if need else "  even 32x batches cannot reach reliability 0.8 (signal itself too weak)")

    # 3) Selection after task centering: oracle (global mean, non-causal) = ceiling;
    #    causal (running mean) = deployable.
    sel_oracle = [_sel(Dt[t] - Dbar, k) for t in range(T)]
    sel_causal = []
    for t in range(T):
        ref = Dt[:t].mean(0) if t > 0 else np.zeros_like(Dt[0])         # t=0 has no history -> plain D~
        sel_causal.append(_sel(Dt[t] - ref, k))
    jo = [_jaccard(a, b) for a, b in itertools.combinations(sel_oracle, 2)]
    jc = [_jaccard(a, b) for a, b in itertools.combinations(sel_causal, 2)]
    print(f"\n--- [{name}] 3) selection after task centering ---")
    for t in range(T):
        print(f"  task {t}: oracle={sel_oracle[t]}  causal={sel_causal[t]}")
    print(f"  oracle distinct={len({tuple(s) for s in sel_oracle})}/{T} mean Jaccard={np.mean(jo):.3f}")
    print(f"  causal distinct={len({tuple(s) for s in sel_causal})}/{T} mean Jaccard={np.mean(jc):.3f}")

    # 4) Selection reproducibility (secondary test): task information may exist yet be too
    #    unstable to drive a stable layer set.
    selA = [_sel(dA[t], k) for t in range(T)]
    selB = [_sel(dB[t], k) for t in range(T)]
    j_match, j_mis, p_rep, exact = selection_reproducibility(selA, selB)
    print(f"\n--- [{name}] 4) selection reproducibility (secondary: stable enough to drive selection?) ---")
    for t in range(T):
        print(f"  task {t}: A half={selA[t]}  B half={selB[t]}  J={_jaccard(selA[t], selB[t]):.2f}")
    print(f"  matched Jaccard={j_match:.3f} vs mismatched={j_mis:.3f}  excess={j_match - j_mis:+.3f}  "
          f"permutation p={p_rep:.4f}   (exact match {exact}/{T}; too strict, not used as a criterion)")
    if hits >= max(3, nT // 2) and p_rep > 0.1:
        print("  Note: task information exists (2 passes) but selection does not reproduce -> SNR too low to drive selection, not absence of signal.")

    # 5) Numerical health: min(S)=0 is very common, so first rule out sd_ref collapse blowing up z.
    rel_sd = c["ref_sd"] / (np.abs(c["ref_mu"]) + 1e-12)
    relu_mass = np.maximum(Dt, 0).sum(axis=(0, 1))
    relu_mass = relu_mass / (relu_mass.sum() + 1e-30)
    print(f"\n--- [{name}] 5) numerical health (per layer, L={c['ref_mu'].shape[1]}) ---")
    print(f"  ref_mu      median = {np.array2string(np.median(c['ref_mu'], 0), precision=3, max_line_width=200)}")
    print(f"  ref_sd/mu   median = {np.array2string(np.median(rel_sd, 0), precision=4, max_line_width=200)}"
          "   <- relative sampling SE; should be similar across layers (~2%); a layer near 0 is numerically ill-posed")
    print(f"  |z|         median = {np.array2string(np.median(c['z_med'], 0), precision=1, max_line_width=200)}"
          "   <- values >> 3 everywhere mean z is inflated by the CL-data <-> NSD domain shift (expected)")
    print(f"  relu(D~) column mass = {np.array2string(relu_mass, precision=3, max_line_width=200)}")

    return dict(backbone=name, k=k, T=T, tau=tau, common=common, between=between,
                ident_hits=hits, ident_p=p_id, ident_r_match=r_mat, ident_r_mismatch=r_mis,
                split_half_r=r, split_half_r_sb=r_sb, per_task_r=per_task,
                sel_oracle=sel_oracle, sel_causal=sel_causal,
                distinct_oracle=len({tuple(s) for s in sel_oracle}),
                distinct_causal=len({tuple(s) for s in sel_causal}),
                repro_match=j_match, repro_mismatch=j_mis, repro_p=p_rep, repro_exact=exact,
                rel_sd_median=np.median(rel_sd, 0).tolist(),
                z_median=np.median(c["z_med"], 0).tolist(),
                relu_mass=relu_mass.tolist())


def verdict(res):
    """Three-branch verdict. 2) (task identification) is the primary criterion; 4) (selection
    reproducibility) only decides whether the information is usable.

    Deliberately not binary: synthetic calibration shows a real middle zone where task information
    clearly exists (identification 10/10) but the SNR is too low for the 4-of-12 greedy to converge
    (reproducibility ~ chance). There the right action is to raise the SNR, neither stop nor task-center.
    """
    print("\n" + "=" * 74)
    for r in res.values():
        if r is None:
            continue
        n, T = r["backbone"], r["T"]
        hits, p_id, p_rep = r["ident_hits"], r["ident_p"], r["repro_p"]
        branch = diagnose_verdict(p_id, p_rep, r["repro_match"], r["repro_mismatch"])
        if branch == "stop":
            v = ("STOP: the A-half ROI x layer residuals cannot identify the B-half task (~chance) -> "
                 "the NSD-encoder drive matrix carries no task information. Do NOT apply task centering: "
                 "it would make every task pick a distinct set while selecting sampling noise. "
                 "Report as a negative result (reachable-set plot + u[l] curve + Delta R^2 oracle).")
        elif branch == "proceed":
            v = ("PROCEED: task information exists and is stable enough to drive selection, but is swamped by "
                 "the CL-data <-> NSD domain shift -> apply task centering (reference = running mean of past "
                 "in-domain tasks instead of NSD) and rerun the export.")
        else:
            need = next((m for m in (2, 4, 8, 16, 32)
                         if _spearman_brown(max(r["split_half_r"], 1e-6), m) >= 0.8), None)
            v = (f"LOW-SNR: task identification {hits}/{T} is significant (p={p_id:.1e}) but selection does not reproduce "
                 f"(permutation p={p_rep:.3f}) -> signal exists, SNR is insufficient. Try in order: "
                 f"1) rerun this diagnostic with BATCHES raised {need or '32'}x; 2) use k=2 (fewer picks are more stable); "
                 "3) coarser ROI aggregation (merge V1-V4 / higher areas). Only if none moves r is it noise.")
        print(f"[{n}] tau={r['tau']:.4f}  task_ident={hits}/{T}(p={p_id:.1e})  r_SB={r['split_half_r_sb']:.2f}  "
              f"sel_repro J={r['repro_match']:.3f}vs{r['repro_mismatch']:.3f}(p={p_rep:.3f})")
        print(f"      {v}")
    print("=" * 74)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    only = os.environ.get("ONLY", "").strip().lower()
    k = int(os.environ.get("K", 4))
    nb = int(os.environ.get("BATCHES", 10))
    res, raw = {}, {}
    for name, bk in BACKBONES.items():
        if only and only != name:
            continue
        try:
            c = collect(name, bk, device, k, nb)
            if c is None:
                continue
            raw[name] = c
            res[name] = analyse(c)
            res[name]["atlas_sig"] = c["atlas_sig"]   # cross-script consistency check, see atlas_signature
        except Exception:
            traceback.print_exc()
            print(f"[G1b:{name}] ✗ failed (most likely atlas_v2.pt missing / fMRI data not present).")
    verdict(res)
    sigs = {n: r["atlas_sig"]["sha1"] for n, r in res.items() if r.get("atlas_sig")}
    if sigs:
        print("\n[atlas fingerprint] " + "  ".join(f"{n}={s}" for n, s in sigs.items()))
        print("  -> must equal atlas_sig.sha1 of the same backbone in atlas_v2_gate_k*.json; "
              "otherwise per-task selections of the two runs must not be cross-referenced.")

    dump = os.environ.get("DUMP", f"./outputs/atlas_v2_diag_k{k}_b{nb}.json")
    os.makedirs(os.path.dirname(dump) or ".", exist_ok=True)
    with open(dump, "w", encoding="utf-8") as f:
        json.dump(dict(k=k, batches=nb, result=res), f, ensure_ascii=False, indent=2)
    # D matrices go to a separate npz so offline re-analysis (other k, gamma, centering) needs no GPU.
    npz = dump.replace(".json", ".npz")
    np.savez_compressed(npz, **{f"{n}_{key}": raw[n][key] for n in raw
                                for key in ("D_full", "D_A", "D_B", "ref_mu", "ref_sd", "z_med")})
    print(f"[dump] {dump}\n[dump] {npz}  <- D matrices for offline re-analysis (no GPU needed)")


if __name__ == "__main__":
    main()
