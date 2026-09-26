# -*- coding: utf-8 -*-
"""Offline pre-training checks for BioScore v2 (forward only; no CL model is trained).

g0() -- structural diagnosis of v1 (zero cost)
Enumerates how many distinct top-k sets the v1 score `score = A_roi_layer^T . v` (v >= 0 is the task
profile) can produce over the whole non-negative cone. If very few sets are reachable and one of them
covers most of the cone volume, a fixed top-k across tasks/backbones is structurally forced rather than
a coincidence: no task profile can escape it.

g1() -- does v2 vary with task / backbone (~1 GPU-h, forward only)
Runs v2 layer selection on the 10 tasks of the chosen dataset for each backbone and checks:
  1) across tasks: >= 5 distinct sets out of 10 and mean pairwise Jaccard < 0.7
  2) across backbones: |S_a & S_b| <= k/2 (descriptive only, see verdict())
  3) non-degenerate: S is neither first-k nor last-k, and min(S) is not simply shallower
It also reports the per-task non-additive (interaction) energy: if ~0, the atlas ROI x layer
interaction is empty and submodular coverage has nothing to use.

Prerequisites
The atlas plmodel.ckpt files must exist under `outputs/bioscore_atlas/<fingerprint>/`.
v2 re-extracts the rich atlas (per-voxel parameters + val R^2 + reference distribution) from the ckpt;
the PLModel is not retrained.
Note: the fingerprint uses --atlas_seed (default 0), not the CL seed.

Run:  python baselines/bilora_adapter/atlas_v2_export.py
      ONLY=ibot python ...            one backbone only
      K=2 python ...                  other budget (default 4)
      GAMMA=0.5 python ...            with interference penalty (checks whether tasks spread out more)
"""
import collections
import itertools
import json
import os
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from core.cli import apply_preset, build_argparser  # noqa: E402
from baselines.bilora_adapter.bilora import _build_cfg, _load_bilora  # noqa: E402
from model_m.common.bioscore_v2 import get_bioscore_v2, reachable_topk_sets  # noqa: E402

NUM_LAYERS = 12
BACKBONES = {
    "augreg": dict(timm_model="vit_base_patch16_224.augreg_in21k",
                   timm_weights="./pretrained/vit_b16_augreg_in21k.npz",
                   timm_weights_format="augreg_npz"),
    "ibot": dict(timm_model="vit_base_patch16_224",
                 timm_weights="./pretrained/ibot_vitb16.pth",
                 timm_weights_format="ibot_pth"),
    # Third backbone: controls the "objective and data both change" confound. DINO and iBOT are both
    # IN1k self-supervised -> their difference isolates the SSL objective; augreg is IN21k supervised
    # -> objective and data both differ. Isolating supervised vs self-supervised would need a
    # supervised IN1k checkpoint, which --timm_weights_format does not support.
    "dino": dict(timm_model="vit_base_patch16_224",
                 timm_weights="./pretrained/dino_vitbase16_pretrain.pth",
                 timm_weights_format="dino_pth"),
}


def _args(bk):
    # DATASET=inr (or imagenet_r) switches to ImageNet-R task images, so the rho allocation can be
    # computed on IN-R with the same pipeline (point DUMP=... at a *_inr file name so the CIFAR dump
    # is not overwritten). Default/empty = CIFAR-100; unknown values raise instead of falling back.
    ds = os.environ.get("DATASET", "").strip().lower()
    if ds in ("", "c100", "cifar100"):
        preset, dataset, cpt = "timm_cil_cifar100", "cifar100", 10
    elif ds in ("inr", "imagenet_r"):
        preset, dataset, cpt = "timm_cil_imagenet_r", "imagenet_r", 20
    elif ds == "cub":   # CUB: point DUMP at a *_cub file name so other datasets' dumps are not overwritten
        preset, dataset, cpt = "timm_cil_cub", "cub", 20
    else:
        raise ValueError(f"DATASET={ds!r} not recognised (choices: empty/c100/cifar100/inr/imagenet_r/cub)")
    a = apply_preset(build_argparser().parse_args(["--preset", preset, "--dataset", dataset]))
    a.data_root, a.fmri_data_dir = "./data", "./subj01"
    a.seed, a.num_workers, a.batch_size = 0, 0, 128
    a.num_tasks, a.classes_per_task = 10, cpt
    a.max_train_steps = a.max_eval_steps = 0
    a.selector_batches = 10
    a.allow_mock_atlas = False           # real-atlas check enabled
    a.bioscore_cache_dir = "./outputs/bioscore_atlas"
    a.atlas_seed = int(os.environ.get("ATLAS_SEED", 0))
    a.bioscore_gamma = float(os.environ.get("GAMMA", 0.0))
    # Same switch as the training runs: "none" reproduces the two-way-centering-only failure mode
    # (almost every task selects the same set).
    a.bioscore_task_center = os.environ.get("BIOSCORE_TASK_CENTER", "running")
    a.__dict__.update(bk)
    return a


def _task_loaders(args, n_tasks):
    """Per-task training loaders from BiLoRA's own DataManager, identical to the training runs
    (same normalisation, same shuffle); otherwise a different selection could be a loader artefact."""
    DataManager, _B, _c = _load_bilora(args.timm_weights or None)
    cfg = _build_cfg(args, torch.device("cpu"))
    dm = DataManager(cfg["dataset"], cfg["shuffle"], cfg["seed"], cfg["init_cls"], cfg["increment"], cfg)
    out = []
    for t in range(n_tasks):
        lo, hi = t * args.classes_per_task, (t + 1) * args.classes_per_task
        ds = dm.get_dataset(np.arange(lo, hi), source="train", mode="train")
        g = torch.Generator().manual_seed(args.seed)
        out.append(DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g))
    return out


class _Strip:
    """BiLoRA's DummyDataset yields (idx, x, y); v2 only needs x."""

    def __init__(self, ld):
        self._ld = ld

    def __iter__(self):
        for b in self._ld:
            yield (b[1], b[2]) if len(b) == 3 else b


def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def _anchor_layers(sels, rate=0.8):
    """Layers selected in >= rate of tasks are anchors; slots taken by anchors add no cross-task variation."""
    if not sels:
        return []
    cnt = collections.Counter(l for s in sels for l in s)
    return sorted(l for l, c in cnt.items() if c >= rate * len(sels))


def atlas_signature(calc):
    """Content fingerprint of this atlas. Must match across scripts/runs, otherwise selections
    cannot be cross-referenced.

    The task-specific residual is only 5-10% of ||D~||, so any tiny atlas difference (e.g. the top-q
    voxel boundary flipping under non-deterministic GPU reductions) lands exactly on the signal being
    selected; two runs with slightly different atlases agreed on only ~5-6/10 per-task selections.
    Writing the fingerprint into every dump makes such mismatches detectable immediately.
    """
    import hashlib

    h = hashlib.sha1()
    for attr in ("ref_mu", "ref_sd", "v_q", "v_sel_layer"):
        v = getattr(calc, attr, None)
        if v is None:
            continue
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    return dict(sha1=h.hexdigest()[:16],
                roi_names=list(getattr(calc, "roi_names", [])),
                n_voxel=int(getattr(calc, "v_q").shape[0]) if getattr(calc, "v_q", None) is not None else -1,
                atlas_path=str(getattr(calc, "atlas_v2_path", "") or ""))


def g0(name, args, device):
    """Enumerate reachable v1 top-k sets. Reads the v1 atlas.pt (A_roi_layer); skips if absent."""
    from model_m.common.selectors import _backbone_fingerprint

    # The fingerprint only checks whether type(backbone).__name__ contains "Timm", so an empty
    # stand-in class suffices (no need to build a ViT). The class name must be ModifiedTimmViT.
    fp = _backbone_fingerprint(args, type("ModifiedTimmViT", (), {})())
    p = pathlib.Path(args.bioscore_cache_dir) / fp / "atlas.pt"
    if not p.exists():
        print(f"[G0:{name}] no v1 atlas ({p}), skipping structural diagnosis.")
        return None
    A = torch.load(str(p), map_location="cpu")["A_roi_layer"].numpy()
    k = int(os.environ.get("K", 4))
    n, top = reachable_topk_sets(A, k, n_samples=20000, seed=0)
    print(f"\n[G0:{name}] number of top-{k} sets reachable by the v1 score over the non-negative cone = {n}")
    for t, frac in top[:5]:
        print(f"        {list(t)}  cone volume fraction {frac:.3f}")
    print(f"[G0:{name}] reading: small n and a dominant first set -> v1's fixed selection is structural, not a coincidence.")
    return dict(n_reachable=n, top=[(list(t), f) for t, f in top[:8]])


def g1(name, bk, device, k):
    args = _args(bk)
    if not os.path.exists(args.timm_weights):
        print(f"[G1:{name}] ✗ weights missing {args.timm_weights}, skipping.")
        return None
    from model_m.common.timm_backbone import ModifiedTimmViT

    backbone = ModifiedTimmViT(model_name=args.timm_model, weights_file=args.timm_weights,
                               pretrained=True, weights_format=args.timm_weights_format).to(device)
    calc = get_bioscore_v2(args, device, backbone)
    loaders = _task_loaders(args, args.num_tasks)

    sels, inters, profiles, hist, rfracs, d_hist = [], [], [], [], [], []
    # Running mean for the third (task) centering axis. The state must live here, not in calc:
    # calc is memoised globally per backbone fingerprint and would leak across CL sequences.
    # Mirrors bioscore_gate._v2_call; if the two diverge, this offline check and the training
    # runs no longer test the same method.
    d_sum, d_n = None, 0
    for t, ld in enumerate(loaders):
        ref = (d_sum / d_n) if d_n else None      # t=0 has no history -> D~ without task centering (cold start, reported as is)
        sel, _gains, info = calc.select(_Strip(ld), k, args.selector_batches,
                                        history=hist, task_ref=ref)
        dc = info.get("D_centered")               # = D~_t before task centering
        if dc is not None:
            d_hist.append(np.asarray(dc))
            d_sum = dc.copy() if d_sum is None else d_sum + dc
            d_n += 1
        hist.append((sel, info["roi_w"]))
        sels.append(sel)
        inters.append(info["interaction"])
        rfracs.append(float(info.get("residual_frac", 1.0)))
        profiles.append([float(v) for v in info["roi_w"]])
        top_roi = info["roi_names"][int(np.argmax(info["roi_w"]))] if len(info["roi_w"]) else "?"
        print(f"[G1:{name}] task {t}: S={sel}  interaction_energy={info['interaction']:.3f}  top_ROI={top_roi}"
              f"  tc={'on' if info.get('task_centered') else 'off'}"
              f"(rf={info.get('residual_frac', 1.0):.3f})")

    distinct = {tuple(s) for s in sels}
    jac = [_jaccard(a, b) for a, b in itertools.combinations(sels, 2)]
    first_k, last_k = list(range(k)), list(range(NUM_LAYERS - k, NUM_LAYERS))
    res = dict(backbone=name, k=k, selections=sels, distinct=len(distinct),
               mean_jaccard=float(np.mean(jac)) if jac else 1.0,
               interaction_mean=float(np.mean(inters)),
               task_center=str(getattr(args, "bioscore_task_center", "running")),
               residual_frac=rfracs,        # expected ~sqrt(tau) ~0.1; constant 1.000 means task_ref was always None
               min_layers=[int(min(s)) for s in sels],
               eq_first=sum(1 for s in sels if s == first_k),
               eq_last=sum(1 for s in sels if s == last_k),
               roi_names=list(calc.roi_names), roi_profiles=profiles,
               # Per-task D~ hashes + atlas fingerprint: lets other scripts check they used the same atlas / D.
               atlas_sig=atlas_signature(calc),
               d_sig=[__import__("hashlib").sha1(
                   np.ascontiguousarray(d, dtype=np.float32).tobytes()).hexdigest()[:12]
                   for d in d_hist])
    print(f"\n[G1:{name}] distinct={res['distinct']}/10  mean_Jaccard={res['mean_jaccard']:.3f}  "
          f"task_center={res['task_center']} median_residual_frac={float(np.median(rfracs)):.3f}  "
          f"mean_interaction_energy={res['interaction_mean']:.3f}  "
          f"=first-k {res['eq_first']}x / =last-k {res['eq_last']}x  "
          f"min(S)={res['min_layers']}")
    return res


def verdict(all_g1, k):
    print("\n" + "=" * 66)
    ok = True
    for r in all_g1.values():
        if r is None:
            continue
        a = r["distinct"] >= 5 and r["mean_jaccard"] < 0.7
        b = r["interaction_mean"] > 0.05
        c = r["eq_first"] == 0 and r["eq_last"] == 0
        print(f"[G1:{r['backbone']}] varies across tasks {'PASS' if a else 'FAIL'} | "
              f"interaction energy non-empty {'PASS' if b else 'FAIL'} | not a fixed rule {'PASS' if c else 'FAIL'}")
        ok &= a and b and c
    # Cross-backbone: compare all pairs and average over all tasks (a pairwise-only
    # `len(keys) == 2` check would silently skip this with three backbones).
    # The |S & S'| <= k/2 threshold has no empirical basis (two same-domain ViT-B/16 may well share
    # layers), so it is reported descriptively only and not folded into `ok`.
    keys = [x for x in all_g1 if all_g1[x]]
    cross = {}
    for a, b in itertools.combinations(keys, 2):
        A, B = all_g1[a]["selections"], all_g1[b]["selections"]
        n = min(len(A), len(B))
        inters = [len(set(A[t]) & set(B[t])) for t in range(n)]
        jac = float(np.mean([_jaccard(A[t], B[t]) for t in range(n)])) if n else 1.0
        cross[f"{a}|{b}"] = dict(mean_inter=float(np.mean(inters)) if n else 0.0,
                                 mean_jaccard=jac, per_task_inter=inters,
                                 anchors_a=_anchor_layers(A), anchors_b=_anchor_layers(B))
        print(f"[G1:cross-backbone] {a} vs {b}: mean per-task overlap={np.mean(inters):.2f}/{k} "
              f"mean_Jaccard={jac:.3f} | anchors {a}={_anchor_layers(A)} {b}={_anchor_layers(B)}"
              + ("  <- different anchors = backbone-specific" if _anchor_layers(A) != _anchor_layers(B) else ""))
    if not cross:
        print("[G1:cross-backbone] only one backbone has results; skipping cross-backbone comparison.")
    print("[G1:cross-backbone] ^ descriptive only, not part of the pass decision (the |S&S'|<=k/2 threshold has no empirical basis).")
    print("=" * 66)
    print(("PASS: selection varies with task/backbone -> proceed" if ok else
           "FAIL: do not proceed. If the interaction energy is ~0, the atlas ROI x layer interaction is empty; "
           "report as a negative result (diagnostic plot + u[l] curve + Delta R^2 oracle)."))
    return ok, cross


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    only = os.environ.get("ONLY", "").strip().lower()
    k = int(os.environ.get("K", 4))
    out = dict(k=k, gamma=float(os.environ.get("GAMMA", 0.0)), g0={}, g1={})
    for name, bk in BACKBONES.items():
        if only and only != name:
            continue
        try:
            out["g0"][name] = g0(name, _args(bk), device)
        except Exception:
            traceback.print_exc()
        try:
            out["g1"][name] = g1(name, bk, device, k)
        except Exception:
            traceback.print_exc()
            print(f"[G1:{name}] ✗ failed (most likely ckpt missing / fMRI data not present).")
    _ok, out["cross"] = verdict(out["g1"], k)
    # Cross-run consistency: the same backbone must use the same atlas in every script/run,
    # otherwise per-task selections cannot be cross-referenced. Print the fingerprints here.
    sigs = {n: r["atlas_sig"]["sha1"] for n, r in out["g1"].items() if r and r.get("atlas_sig")}
    if sigs:
        print("\n[atlas fingerprint] " + "  ".join(f"{n}={s}" for n, s in sigs.items()))
        print("  -> compare with the atlas_v2_diag.py dump; if they differ, per-task selections of the two must not be cross-referenced.")
    dump = os.environ.get("DUMP", f"./outputs/atlas_v2_gate_k{k}.json")
    os.makedirs(os.path.dirname(dump) or ".", exist_ok=True)
    with open(dump, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[dump] {dump}")


if __name__ == "__main__":
    main()
