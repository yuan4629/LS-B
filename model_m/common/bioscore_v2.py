# -*- coding: utf-8 -*-
"""BioScore v2: ROI x layer drive matrix + submodular coverage layer selection.

Why not patch v1: the v1 score (`selectors.BioScoreCalculator._compute_score_last`) is

    score_t[l] = sum_r v_t[r] * A[r,l]      v_t = relu(W_roi_ch @ Var_b(feat_last))

Layer information lives only in the task-independent A and task information only in the
layer-independent v_t (a rank-1 outer product). When the ROI layer profiles are roughly proportional
(visual-cortex encoding generally favors deep layers), score_t ~ (u . v_t) * a for every task, so the
argsort does not depend on t and top-k is fixed across tasks and backbones. The `perlayer` mode does not
fix this: ViT feature scale grows by ~6 orders of magnitude from layer 0 to layer 11, so the ranking is
dominated by a task-independent per-layer scale instead.

Principle: any task-independent positive per-layer factor silently decides the ranking, so the score
must be invariant to such factors, i.e. standardized against a reference distribution, not raw magnitude.

v2 recovers the discarded interaction term. The encoding model (model.py of the third-party brainnet
package) is exactly linear in layers:

    y_hat[b,n] = sum_l sel_layer[n,l] * c_l[b,n] + bias[n]
    c_l[b,n]   = (1/D) sum_d weight[n,d] * [ (1 - s_n) * local_l[b,n,d] + s_n * g_l[b,d] ]

c_l[b,n] is "the response voxel n would give if it read only layer l", a task x voxel x layer tensor.
The v1 atlas keeps only its two marginals (W_roi_ch, A_roi_layer); the discarded interaction is the only
component that lets the selection change with the task.

Four steps (forward passes only):
  1. s_t[n,l] = std of c_l[b,n] over the task's images; z-standardize against the sampling distribution
     of NSD reference sub-pools -> z_t[n,l]
  2. aggregate to ROIs, weighted by encoding quality q[n] = val R^2 and the encoder's own layer
     assignment sel_layer[n,l] -> D_t[r,l]
  3. two-way centering keeps the interaction -> D~_t (layer main effect is an optional term lambda_depth, default 0)
  4. facility-location submodular greedy picks k layers (max over S gives diminishing returns, so redundant
     adjacent layers are avoided), plus a cross-task interference penalty P_t (layers used by similar
     earlier tasks are not reused)
"""
from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model_m.common.selectors import BioScoreCalculator, _backbone_fingerprint

ATLAS_V2_VERSION = 2
EPS = 1e-8


# ---------------------------------------------------------------------------
# Submodular layer selection (pure numpy, testable in isolation)
# ---------------------------------------------------------------------------
def select_layers_submodular(D, k, roi_w=None, gamma=0.0, history=None, cur_w=None,
                             lambda_depth=0.0, layer_main=None):
    """Facility-location greedy: F(S) = sum_r w[r] * max_{l in S} H[r,l] - gamma * sum_{l in S} P[l]

    Why not top-k: top-k scores each layer independently and cannot express "these two layers do the
    same thing". max over S gives diminishing returns: a second layer serving the same ROI gains ~0.

    Args
      D            [R,L] two-way-centered ROI x layer drive matrix (column means are 0, hence relu, see below)
      k            number of layers to select
      roi_w        [R] ROI weights, None = uniform
      gamma        interference penalty weight
      history      [(S_prev: list[int], w_prev: np.ndarray|None), ...] earlier tasks' layers and ROI profiles
      cur_w        [R] current task's ROI profile (for similarity to earlier tasks)
      lambda_depth weight of the layer main effect, default 0 (any depth preference must be explicit)
      layer_main   [L] layer main effect (the term removed by centering)

    Returns (selected: list[int] ascending, gains: list[float] marginal gain of each layer when selected)
    """
    D = np.asarray(D, dtype=np.float64)
    R, L = D.shape
    k = int(max(1, min(k, L)))
    w = np.ones(R) / R if roi_w is None else np.asarray(roi_w, dtype=np.float64)
    w = w / (w.sum() + EPS)

    # Positive part, not a per-ROI min shift: D is two-way centered, so every column mean is 0 and the
    # first greedy gain w . H[:,l] would be identical for all layers under a shift (first pick decided by
    # float noise; caught by tests/test_bioscore_v2.py [5]). relu(D~)[r,l] is ROI r's preference for layer l
    # above its own average ("is this a dedicated layer for some ROI"), and stays >= 0 (monotone submodular).
    H = np.maximum(D, 0.0)
    if lambda_depth and layer_main is not None:
        lm = np.asarray(layer_main, dtype=np.float64)
        lm = lm - lm.min()
        lm = lm / (lm.max() + EPS)               # scale to [0,1] so lambda has a stable unit
        H = H + lambda_depth * lm[None, :]
    hmax = float(H.max())
    if hmax > EPS:
        H = H / hmax                             # global normalization: objective is invariant to scaling D
    hbar = float(H.mean())

    # Interference penalty: layers used by earlier tasks, weighted by ROI-profile similarity to the current task
    P = np.zeros(L)
    if gamma and history:
        for S_prev, w_prev in history:
            sim = 1.0
            if cur_w is not None and w_prev is not None:
                a, b = np.asarray(cur_w, float).ravel(), np.asarray(w_prev, float).ravel()
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                sim = float(a @ b / (na * nb)) if na > EPS and nb > EPS else 0.0
                sim = max(0.0, sim)
            for l in S_prev:
                if 0 <= int(l) < L:
                    P[int(l)] += sim
        P /= max(1, len(history))
        P *= hbar + EPS      # same scale as coverage gains, otherwise gamma drifts with data scale

    cov = np.zeros(R)
    base = float(w @ cov)
    sel, gains = [], [0.0] * L
    for _ in range(k):
        best_l, best_g = None, -np.inf
        for l in range(L):
            if l in sel:
                continue
            g = float(w @ np.maximum(cov, H[:, l])) - base - gamma * P[l]
            if g > best_g:
                best_l, best_g = l, g
        if best_l is None:
            break
        sel.append(best_l)
        gains[best_l] = best_g
        cov = np.maximum(cov, H[:, best_l])
        base = float(w @ cov)
    return sorted(sel), gains


def two_way_center(D):
    """Two-way centering keeps the interaction: D~[r,l] = D - mean_r - mean_l + grand.
    Returns (D~, layer_main); layer_main is the removed layer main effect (added back via lambda_depth)."""
    D = np.asarray(D, dtype=np.float64)
    grand = D.mean()
    row = D.mean(axis=1, keepdims=True)
    col = D.mean(axis=0, keepdims=True)
    return D - row - col + grand, (col.ravel() - grand)


def interaction_energy(D):
    """Non-additive energy fraction = ||two-way centered||^2 / ||grand-mean removed||^2.

    Measures how much of D exceeds ROI main effect + layer main effect, which is exactly what the
    coverage objective uses: if D is additive (D[r,l] = a_r + b_l) the centered matrix is 0, no ROI
    prefers any layer, and selection degenerates to arbitrary/fixed. Used as v2's stop diagnostic.

    Note: not a rank test. A pure outer product D = u v^T (v1's structure) centers to (u-u_bar)(v-v_bar)^T,
    which is nonzero and has high interaction energy. To test v1's fixed top-k use `reachable_topk_sets`."""
    D = np.asarray(D, dtype=np.float64)
    tot = float(((D - D.mean()) ** 2).sum())
    inter = float((two_way_center(D)[0] ** 2).sum())
    return inter / (tot + EPS)


def jaccard(a, b):
    a, b = set(int(x) for x in a), set(int(x) for x in b)
    return len(a & b) / max(1, len(a | b))


def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def binom_tail(hits, n, p):
    """P(X >= hits), X ~ B(n, p). Computed directly so scipy is not required."""
    from math import comb
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(int(hits), n + 1)))


def spearman_brown(r1, n):
    """Split-half reliability r1 -> reliability with n times the data (how many batches are enough)."""
    return n * r1 / (1 + (n - 1) * r1) if r1 > 0 else 0.0


def identify_tasks(dA, dB):
    """Task identification: identify which task each half-B residual belongs to using half-A residuals.
    Returns (hits, T, p, r_match, r_mismatch).

    Primary test for task information in the ROI x layer drive. Each task contributes R x L cells, so
    power is much higher than for selection reproducibility (only T binary outcomes). Under the null the
    hit rate is 1/T (exact binomial tail). Synthetic calibration (tests/test_bioscore_v2.py [7]): split-half
    correlation as low as 0.31 still gives 10/10 hits; pure noise stays at 1/T.
    """
    dA, dB = np.asarray(dA, dtype=np.float64), np.asarray(dB, dtype=np.float64)
    T = dA.shape[0]
    X, Y = dA.reshape(T, -1).copy(), dB.reshape(T, -1).copy()
    X -= X.mean(1, keepdims=True)
    Y -= Y.mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + EPS
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + EPS
    M = X @ Y.T
    hits = int(np.sum(np.argmax(M, axis=1) == np.arange(T)))
    off = M[~np.eye(T, dtype=bool)]
    return hits, T, binom_tail(hits, T, 1.0 / T), float(np.mean(np.diag(M))), float(off.mean())


def selection_reproducibility(selA, selB, seed=0, n_perm=5000):
    """Selection reproducibility: do layer sets selected on two independent halves match?
    Returns (match J, mismatch J, p, number of exact matches).

    Note: the null must not be uniform random k-subsets; if the method prefers some layers, a uniform
    baseline is too low and noise looks like signal. The null is a mismatch permutation (shuffle the
    task labels of side B), which keeps both sides' selection marginals.
    Note: exact equality is too strict; k-of-L greedy is discontinuous, and with a planted r=0.80 signal
    exact matches are still only 2/10. So only the Jaccard match-mismatch gap is tested.
    Note: with T=10 power is limited (strong signals reach only p~0.05); secondary test only,
    identify_tasks is the primary one.
    """
    T = len(selA)
    M = np.array([[jaccard(a, b) for b in selB] for a in selA])
    match = float(np.mean(np.diag(M)))
    mismatch = float(M[~np.eye(T, dtype=bool)].mean()) if T > 1 else match
    rng = np.random.RandomState(int(seed))
    null = [float(np.mean(M[np.arange(T), rng.permutation(T)])) for _ in range(int(n_perm))]
    p = (1 + sum(1 for v in null if v >= match)) / (1 + len(null))
    return match, mismatch, p, sum(1 for a, b in zip(selA, selB) if list(a) == list(b))


def diagnose_verdict(p_ident, p_repro, j_match, j_mismatch,
                     alpha_ident=0.05, alpha_repro=0.10):
    """Three-way diagnostic decision. Returns 'stop' | 'proceed' | 'more_data'.

    A single function so the logic exists once and can be tested on synthetic data
    (tests/test_bioscore_v2.py [7]).

    Three branches because synthetic calibration shows a real middle zone: task information clearly
    exists (identification 10/10) but SNR is too low for the k-of-L greedy to converge (reproducibility
    ~ random). There the right action is more data, neither stopping nor adding task centering.

    The primary test must be checked first: the secondary test uses alpha=0.10 (limited power at T=10),
    at the cost of ~10% false positives (pure noise can reach p=0.089). Only the primary test ruling out
    'stop' first keeps that false positive from turning noise into downstream runs. Do not reorder.
    """
    if p_ident >= alpha_ident:
        return "stop"                       # no task information -> do not add task centering
    if p_repro < alpha_repro and j_match > j_mismatch:
        return "proceed"                    # information present and stable enough to drive selection
    return "more_data"                      # present but unstable -> increase SNR, do not stop


def reachable_topk_sets(A, k, n_samples=20000, seed=0, min_layers=1):
    """Diagnostic: how many distinct top-k sets can the v1 score = A^T v (v >= 0) produce in principle?

    v1 keeps all task information in v and all layer information in A, so reachable scores form the
    nonnegative cone spanned by A's rows. Sample v uniformly on the simplex and count top-k sets: if few
    sets are reachable and one covers most of the cone, a fixed top-k across tasks/backbones is structural,
    not a coincidence.

    Returns (n_distinct, [(set_tuple, fraction), ...] sorted by fraction, descending).
    """
    A = np.asarray(A, dtype=np.float64)
    R, L = A.shape
    rng = np.random.RandomState(int(seed))
    V = rng.dirichlet(np.ones(R), size=int(n_samples))       # [S,R] uniform on the simplex
    S = V @ A                                                # [S,L]
    kk = int(max(min_layers, min(k, L)))
    order = np.argsort(-S, axis=1)[:, :kk]
    counts = {}
    for row in order:
        t = tuple(sorted(int(i) for i in row))
        counts[t] = counts.get(t, 0) + 1
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    return len(items), [(t, c / len(V)) for t, c in items]


# ---------------------------------------------------------------------------
# Rich atlas + online scoring
# ---------------------------------------------------------------------------
class BioScoreV2Calculator(BioScoreCalculator):
    """v2 calculator: reuses v1's fingerprint, cache dir and PLModel train/load; replaces atlas content and scoring.

    atlas_v2.pt stores the encoder's per-voxel parameters (sel_layer/sel_scale/sel_space/weight/bias, both
    bottlenecks), per-voxel encoding quality q = val R^2, and the NSD reference distribution (ref_mu, ref_sd).
    v1's atlas.pt is untouched; both coexist under different file names.
    """

    def __init__(self, args, device, base_backbone):
        self.atlas_v2_path = None      # used by _build_or_load_atlas inside super().__init__
        super().__init__(args, device, base_backbone)

    # ---------------- build / load
    def _build_or_load_atlas(self):
        self.atlas_v2_path = self.cache_dir / "atlas_v2.pt"
        force = bool(getattr(self.args, "brainnet_force_rebuild", False))
        if not force and self.atlas_v2_path.exists():
            try:
                self._load_atlas_v2()
                print(f"[BioScoreV2] loaded cached atlas from {self.atlas_v2_path}")
                return
            except Exception as e:
                warnings.warn(f"[BioScoreV2] cached atlas_v2 load failed: {e}")

        plm = None
        if not force and self.ckpt_path.exists():
            try:
                # skip_data=False: v2 needs NSD data for per-voxel R^2 and the reference distribution
                plm = self._load_plmodel_with_data()
                print(f"[BioScoreV2] loaded PLModel ckpt from {self.ckpt_path}")
            except Exception as e:
                warnings.warn(f"[BioScoreV2] ckpt load failed: {e}")
        if plm is None and self._fmri_data_available():
            print(f"[BioScoreV2] training PLModel once -> {self.ckpt_path}")
            plm = self._train_plmodel()
        if plm is None:
            raise RuntimeError(
                "BioScore v2 needs a real PLModel (ckpt or fMRI data); v2 is meaningless on a mock atlas."
                f" Expected {self.ckpt_path} or fMRI data in {self.args.fmri_data_dir}.")

        self._extract_atlas_v2(plm)
        self._save_atlas_v2()

    def _load_plmodel_with_data(self):
        from brainnet.config import get_cfg_defaults
        from brainnet.plmodel import PLModel

        cfg = get_cfg_defaults()
        cfg.DATASET.DATA_DIR = self.args.fmri_data_dir
        cfg.DATASET.RESOLUTION = (224, 224)
        plm = PLModel(cfg, self.base_backbone, draw=False, cached=False, skip_data=False)
        state = torch.load(str(self.ckpt_path), map_location="cpu")
        plm.load_state_dict(state.get("state_dict", state), strict=False)
        return plm

    # ---------------- extraction
    @torch.no_grad()
    def _extract_atlas_v2(self, plm):
        dev = self.device
        plm.eval().to(dev)
        m = plm.model

        sel_space, sel_layer, sel_scale = plm.get_selectors()      # [N,2] [N,L] [N,1]
        weight = m.weight.detach()                                  # [N,D]
        bias = m.bias.detach()                                      # [N]
        self.layers = list(m.layers)
        L = len(self.layers)

        q = self._voxel_r2(plm)                                     # [N] per-voxel val R^2
        roi_map, _ = self._resolve_roi_indices()
        self.roi_names = list(roi_map.keys())
        # v2 only makes sense on real anatomical ROIs (coverage relies on V1/V4/FFA/PPA preferring
        # different depths). _resolve_roi_indices silently falls back to 5 equal voxel segments when the
        # real ROIs fail to load; v2 results on that atlas are meaningless, so refuse here.
        if len(self.roi_names) < 8 and not getattr(self.args, "allow_mock_atlas", False):
            raise RuntimeError(
                f"[BioScoreV2] only {len(self.roi_names)} ROIs resolved ({self.roi_names}); "
                "looks like the equal-segment mock partition (the real anatomical partition has 12). v2 refuses to run; "
                "check that brainnet.roi is available, or pass --allow_mock_atlas for smoke tests only.")

        # Keep the topq most reliable voxels (by q) per ROI. Needed for memory (all voxels: [B,D,N] is
        # 2.5 GB per layer) and statistically cleaner: poorly encoded voxels should not vote on layers.
        topq = int(getattr(self.args, "bioscore_voxel_topq", 500) or 500)
        keep, roi_slices, cursor = [], {}, 0
        for name in self.roi_names:
            idx = np.asarray(roi_map[name], dtype=np.int64)
            order = np.argsort(-q[idx])
            take = idx[order[:min(topq, idx.size)]]
            keep.append(take)
            roi_slices[name] = (cursor, cursor + take.size)
            cursor += take.size
        keep = np.concatenate(keep) if keep else np.zeros(0, dtype=np.int64)
        if keep.size == 0:
            raise RuntimeError("[BioScoreV2] all ROI voxel sets are empty; cannot build the atlas.")

        ti = torch.from_numpy(keep).long().to(weight.device)
        self.v_sel_layer = sel_layer[ti].detach().to(dev).float()   # [N',L]
        self.v_sel_scale = sel_scale[ti].detach().to(dev).float().view(-1)   # [N']
        self.v_sel_space = sel_space[ti].detach().to(dev).float()   # [N',2]
        self.v_weight = weight[ti].to(dev).float()                  # [N',D]
        self.v_bias = bias[ti].to(dev).float()                      # [N']
        self.v_q = torch.from_numpy(np.clip(q[keep], 0.0, None)).to(dev).float()  # [N']
        self.voxel_index = keep
        self.roi_slices = roi_slices

        self.bottlenecks = nn.ModuleDict()       # global: same name/structure as v1 for comparison
        for lk, lin in m.global_token_bottleneck.items():
            new = nn.Linear(lin.in_features, lin.out_features, bias=False)
            new.load_state_dict(lin.state_dict())
            self.bottlenecks[lk] = new.to(dev)
        self.local_bottlenecks = nn.ModuleDict()
        for lk, conv in m.local_token_bottleneck.items():
            new = nn.Conv2d(conv.in_channels, conv.out_channels, 1, bias=False)
            new.load_state_dict(conv.state_dict())
            self.local_bottlenecks[lk] = new.to(dev)

        self.is_mock = False
        # Reference distribution: NSD training images split into M sub-pools the size of a task sample,
        # giving the estimator's own sampling distribution.
        self.ref_mu, self.ref_sd = self._reference_stats(plm)
        print(f"[BioScoreV2] atlas: R={len(self.roi_names)} L={L} "
              f"voxels={keep.size} (topq={topq}/ROI) meanR2={float(self.v_q.mean()):.3f}")

    @torch.no_grad()
    def _voxel_r2(self, plm):
        """Per-voxel val R^2. Poorly encoded voxels give noisy (ROI, layer) drives and must be
        down-weighted by R^2, otherwise D_t is dominated by noise."""
        loader = plm.val_dataloader()
        n = plm.n_vertices
        s = torch.zeros(n, dtype=torch.float64, device=self.device)
        s2 = torch.zeros(n, dtype=torch.float64, device=self.device)
        sse = torch.zeros(n, dtype=torch.float64, device=self.device)
        cnt = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device).double()
            yh = plm(x)[0].double()
            s += y.sum(0)
            s2 += (y ** 2).sum(0)
            sse += ((y - yh) ** 2).sum(0)
            cnt += y.shape[0]
        if cnt == 0:
            warnings.warn("[BioScoreV2] val set is empty; setting q = 1 everywhere (no R^2 weighting).")
            return np.ones(n, dtype=np.float32)
        sst = s2 - s ** 2 / cnt
        r2 = 1.0 - sse / (sst + EPS)
        return r2.float().cpu().numpy()

    @torch.no_grad()
    def _reference_stats(self, plm):
        """Split NSD training images into M sub-pools, compute s[n,l] = std_b c_l per pool -> mu_ref/sd_ref [N',L].
        Sub-pools match the task sample size (not the full set) so z is relative to the sampling
        distribution of the same estimator; otherwise sd_ref is underestimated and z blows up."""
        bs = int(getattr(self.args, "batch_size", 128) or 128)
        nb = int(getattr(self.args, "selector_batches", 10) or 10)
        pools = int(getattr(self.args, "bioscore_ref_pools", 10) or 10)
        loader = torch.utils.data.DataLoader(plm.train_dataset, batch_size=bs, shuffle=False,
                                             num_workers=0)
        it, samples = iter(loader), []
        for _ in range(pools):
            batches = []
            for _ in range(nb):
                try:
                    x, _y = next(it)
                except StopIteration:
                    it = iter(loader)
                    x, _y = next(it)
                batches.append(x)
            samples.append(self._pool_std(batches))          # [N',L]
        S = torch.stack(samples, dim=0)                      # [M,N',L]
        mu = S.mean(0)
        sd = S.std(0, unbiased=True) if S.shape[0] > 1 else torch.ones_like(mu)
        return mu, sd

    # ---------------- per-layer drive
    @torch.no_grad()
    def _drive(self, x):
        """c_l[b,n]: the response voxel n would give if it read only layer l, [B,N',L].
        Matches the forward pass of the third-party brainnet model.py (local tokens resized to 8x8 before the bottleneck)."""
        dev = self.device
        _local, _global = self.base_backbone.to(dev).eval().get_tokens(x.to(dev))
        N, D = self.v_weight.shape
        out = torch.empty(x.shape[0], N, len(self.layers), device=dev)
        grid = self.v_sel_space.view(1, N, 1, 2)
        chunk = int(getattr(self.args, "bioscore_voxel_chunk", 2048) or 2048)
        for li, lk in enumerate(self.layers):
            lt = torch.nn.functional.interpolate(_local[lk], size=(8, 8), mode="bilinear",
                                                 align_corners=False)
            lt = self.local_bottlenecks[lk](lt)                       # [B,D,8,8]
            g = self.bottlenecks[lk](_global[lk])                     # [B,D]
            glob = g @ self.v_weight.t()                              # [B,N']
            loc = torch.empty(x.shape[0], N, device=dev)
            for a in range(0, N, chunk):                              # chunked grid_sample to bound memory
                b = min(a + chunk, N)
                gs = torch.nn.functional.grid_sample(
                    lt, grid[:, a:b].expand(x.shape[0], -1, -1, -1),
                    align_corners=False, mode="bilinear", padding_mode="zeros")
                loc[:, a:b] = torch.einsum("nd,bdn->bn", self.v_weight[a:b], gs.squeeze(-1))
            sc = self.v_sel_scale.view(1, N)
            out[:, :, li] = ((1.0 - sc) * loc + sc * glob) / D
        return out

    @torch.no_grad()
    def pool_moments(self, batches):
        """Returns (n, sum_b c, sum_b c^2) as float64 accumulators.

        Moments instead of std so that full-sample moments are exactly the sum of the two halves: the
        split-half diagnostic gets half A, half B and the full estimate from one forward pass (std is not
        linear; averaging stds would be biased).
        float64 is required: the mean of c can be much larger than its std across images, and
        E[x^2] - E[x]^2 in float32 cancels catastrophically."""
        n, s1, s2 = 0, None, None
        for x in batches:
            c = self._drive(x).double()                               # [B,N',L]
            a, b = c.sum(0), (c ** 2).sum(0)
            s1, s2 = (a, b) if s1 is None else (s1 + a, s2 + b)
            n += c.shape[0]
        return n, s1, s2

    def std_from_moments(self, n, s1, s2):
        """(n, sum c, sum c^2) -> s[n,l] = std of c_l[b,n] over images, [N',L] float32."""
        if s1 is None or n < 2:
            return torch.zeros(self.v_weight.shape[0], len(self.layers), device=self.device)
        var = (s2 - s1 ** 2 / n) / (n - 1)
        return var.clamp_min(0).sqrt().float()

    @torch.no_grad()
    def _pool_std(self, batches):
        """s[n,l] over one pool of images, without holding the whole pool's c in memory."""
        return self.std_from_moments(*self.pool_moments(batches))

    # ---------------- online: ROI x layer drive matrix
    @torch.no_grad()
    def collect_batches(self, loader, max_batches=10):
        """Image tensors of the loader's first max_batches batches (labels dropped); selection is unsupervised.

        Uses its own deterministic seed and restores the global RNG state afterwards. This fixes two problems:

        1) Nondeterminism: the selector reads a mode="train" loader with RandomResizedCrop + HFlip, whose
           randomness comes from the global RNG (the sampler is seeded, the augmentation is not). Two runs
           with the same atlas, batch order and code then differ by 1.4-1.8% in D~, while the task-specific
           signal is only 5-10% of ||D~||, so per-task selections agreed on only 6-7 of 10 tasks.

        2) RNG leakage: consuming global random numbers in the selector makes the later training trajectory
           diverge (two runs with identical per-task selections still differed by ~0.4pp). Restoring the
           state makes the selector transparent to training, and different selectors no longer diverge
           because they consume different amounts of randomness.

        Using the same seed for every task is intentional: task images differ anyway, and a fixed
        augmentation sequence makes augmentation a task-invariant nuisance that the third centering axis
        (running task mean) removes, instead of cross-task noise.
        """
        seed = int(getattr(self.args, "seed", 0) or 0) * 1000003 + 7
        st_t, st_np, st_py = torch.get_rng_state(), np.random.get_state(), random.getstate()
        st_c = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 31 - 1))
            random.seed(seed)
            out = []
            for i, batch in enumerate(loader, start=1):
                if max_batches and i > max_batches:
                    break
                out.append(batch[0] if isinstance(batch, (tuple, list)) else batch)
        finally:
            torch.set_rng_state(st_t)
            np.random.set_state(st_np)
            random.setstate(st_py)
            if st_c is not None:
                torch.cuda.set_rng_state_all(st_c)
        if not out:
            raise ValueError("[BioScoreV2] selector loader yielded no batches.")
        return out

    @torch.no_grad()
    def aggregate(self, s_t):
        """s_t [N',L] -> (D_raw [R,L], roi_w [R]), both numpy.

        Exposed separately because the diagnostic aggregates two disjoint halves of the same task for
        split-half reliability.
        Note: z is relative to the NSD reference pools (see atlas_v2_diag.py). The domain shift between
        NSD and the CL dataset leaves an ROI x layer component in z shared by all tasks, one to two orders
        of magnitude larger than between-task differences.
        """
        z = (s_t - self.ref_mu) / (self.ref_sd + EPS)                  # removes the task-independent per-layer scale
        R, L = len(self.roi_names), len(self.layers)
        D = torch.zeros(R, L, device=self.device)
        w_t = torch.zeros(R, device=self.device)
        for ri, name in enumerate(self.roi_names):
            a, b = self.roi_slices[name]
            if b <= a:
                continue
            qq = self.v_q[a:b]
            qq = qq / (qq.sum() + EPS)                                 # normalize encoding quality within the ROI
            # sel_layer is the encoder's own layer assignment: voxels that do not read layer l do not vote for it
            D[ri] = (qq.unsqueeze(1) * self.v_sel_layer[a:b] * z[a:b]).sum(0)
            w_t[ri] = (qq * z[a:b].abs().mean(dim=1)).sum()
        return D.detach().cpu().numpy(), torch.relu(w_t).detach().cpu().numpy()

    @torch.no_grad()
    def drive_matrix(self, loader, max_batches=10):
        """Returns (D~ [R,L], info). info holds the uncentered D, layer main effect, ROI profile w_t and interaction energy."""
        batches = self.collect_batches(loader, max_batches)
        Dn, w = self.aggregate(self._pool_std(batches))
        Dt, layer_main = two_way_center(Dn)
        info = dict(D_raw=Dn, layer_main=layer_main, roi_w=w,
                    interaction=interaction_energy(Dn),
                    roi_names=list(self.roi_names),
                    # Number of batches/images actually read. collect_batches does not fail when the loader
                    # has fewer than max_batches batches (e.g. ImageNet-R T=20 task 1 has 1142 images = 9
                    # batches). Recorded only; the readout is unchanged.
                    n_batches=len(batches), n_images=int(sum(len(b) for b in batches)))
        return Dt, info

    @torch.no_grad()
    def select(self, loader, k, max_batches=10, history=None, task_ref=None):
        """Full layer selection for one task. Returns (selected: list[int], gains: list[float], info: dict).

        `task_ref` is the third centering axis: running mean [R,L] of the earlier tasks' D~, None at t=0.

        Why: z is referenced to NSD while task data come from the CL dataset, so the domain shift leaves an
        ROI x layer component in D~ shared by all tasks. The task component was only ~5-10% of the shared
        one in amplitude, and selection was dominated by the shared component (nearly all tasks picked the
        same set). Two-way centering removes only ROI and layer main effects, not the shift's own ROI x layer
        interaction, so the cross-task mean must be subtracted as well.

        Safety: first verify that the task residual is not sampling noise (identify_tasks,
        selection_reproducibility, diagnose_verdict). Applied to pure noise, this term would also make every
        task's selection distinct and look like a pass. Do not reorder.

        The running mean is held by the caller, not the calculator: calculators are memoized globally by
        fingerprint in `_V2_CALCS` and reused across CL sequences in one process, so storing it here would
        leak state between sequences.
        """
        Dt, info = self.drive_matrix(loader, max_batches)
        info["D_centered"] = Dt.copy()          # D~ before task centering, for the caller's running mean
        mode = str(getattr(self.args, "bioscore_task_center", "running") or "running")
        info["task_centered"] = False
        if mode != "none" and task_ref is not None:
            ref = np.asarray(task_ref, dtype=np.float64)
            if ref.shape == Dt.shape:
                n0 = float(np.linalg.norm(Dt))
                Dt = Dt - ref
                info["task_centered"] = True
                # Residual fraction: expected ~0.1 (sqrt of the task share); near 1 means the running mean had no effect
                info["residual_frac"] = float(np.linalg.norm(Dt) / (n0 + EPS))
        uniform = str(getattr(self.args, "bioscore_roi_weight", "uniform") or "uniform") == "uniform"
        roi_w = None if uniform else info["roi_w"]
        sel, gains = select_layers_submodular(
            Dt, k, roi_w=roi_w,
            gamma=float(getattr(self.args, "bioscore_gamma", 0.0) or 0.0),
            history=history, cur_w=info["roi_w"],
            lambda_depth=float(getattr(self.args, "bioscore_lambda_depth", 0.0) or 0.0),
            layer_main=info["layer_main"])
        return sel, gains, info

    # ---------------- cache I/O
    def _save_atlas_v2(self):
        state = dict(
            version=ATLAS_V2_VERSION, fingerprint=self.fingerprint,
            layers=self.layers, roi_names=self.roi_names, roi_slices=self.roi_slices,
            # Stored as a tensor, not numpy: torch.load in PyTorch >= 2.6 defaults to weights_only=True, and a
            # numpy array fails the whole load ("GLOBAL numpy._core.multiarray._reconstruct not allowed").
            # That failure only warns, so the cache never hits and the atlas is rebuilt every run; GPU reduction
            # nondeterminism then flips the top-topq voxel boundary, so atlases (and selections) differ across runs.
            voxel_index=torch.as_tensor(np.asarray(self.voxel_index), dtype=torch.long),
            sel_layer=self.v_sel_layer.cpu(), sel_scale=self.v_sel_scale.cpu(),
            sel_space=self.v_sel_space.cpu(), weight=self.v_weight.cpu(),
            bias=self.v_bias.cpu(), q=self.v_q.cpu(),
            ref_mu=self.ref_mu.cpu(), ref_sd=self.ref_sd.cpu(),
            global_bottleneck={k: v.state_dict() for k, v in self.bottlenecks.items()},
            local_bottleneck={k: v.state_dict() for k, v in self.local_bottlenecks.items()},
        )
        torch.save(state, str(self.atlas_v2_path))
        with open(self.cache_dir / "meta_v2.json", "w", encoding="utf-8") as f:
            json.dump(dict(version=ATLAS_V2_VERSION, fingerprint=self.fingerprint,
                           roi_names=self.roi_names, n_voxels=int(self.v_q.numel()),
                           mean_val_r2=float(self.v_q.mean()),
                           timm_weights=getattr(self.args, "timm_weights", None)),
                      f, ensure_ascii=False, indent=2)
        print(f"[BioScoreV2] atlas cached -> {self.atlas_v2_path}")

    def _load_atlas_v2(self):
        # weights_only=False: the file is written by this code (not external), and older files store
        # voxel_index as numpy, which fails under weights_only=True. Fallback for old torch without the kwarg.
        try:
            st = torch.load(str(self.atlas_v2_path), map_location=self.device, weights_only=False)
        except TypeError:
            st = torch.load(str(self.atlas_v2_path), map_location=self.device)
        if st.get("fingerprint") != self.fingerprint:
            raise ValueError(f"atlas_v2 fingerprint mismatch: {st.get('fingerprint')} != {self.fingerprint}")
        dev = self.device
        self.layers = list(st["layers"])
        self.roi_names = list(st["roi_names"])
        self.roi_slices = {k: tuple(v) for k, v in st["roi_slices"].items()}
        vi = st["voxel_index"]
        self.voxel_index = vi.cpu().numpy() if torch.is_tensor(vi) else np.asarray(vi)
        self.v_sel_layer = st["sel_layer"].to(dev)
        self.v_sel_scale = st["sel_scale"].to(dev)
        self.v_sel_space = st["sel_space"].to(dev)
        self.v_weight = st["weight"].to(dev)
        self.v_bias = st["bias"].to(dev)
        self.v_q = st["q"].to(dev)
        self.ref_mu = st["ref_mu"].to(dev)
        self.ref_sd = st["ref_sd"].to(dev)
        self.bottlenecks = nn.ModuleDict()
        for lk, sd in st["global_bottleneck"].items():
            o, i = sd["weight"].shape
            lin = nn.Linear(i, o, bias=False)
            lin.load_state_dict(sd)
            self.bottlenecks[lk] = lin.to(dev)
        self.local_bottlenecks = nn.ModuleDict()
        for lk, sd in st["local_bottleneck"].items():
            o, i = sd["weight"].shape[:2]
            conv = nn.Conv2d(i, o, 1, bias=False)
            conv.load_state_dict(sd)
            self.local_bottlenecks[lk] = conv.to(dev)
        self.is_mock = False
        # v1's two marginals are unused, but the attributes are kept so external code does not break on None
        self.W_roi_ch = None
        self.A_roi_layer = torch.zeros(len(self.roi_names), len(self.layers), device=dev)


_V2_CALCS = {}


def get_bioscore_v2(args, device, base_backbone):
    key = _backbone_fingerprint(args, base_backbone)
    if key not in _V2_CALCS:
        _V2_CALCS[key] = BioScoreV2Calculator(args, device, base_backbone)
    return _V2_CALCS[key]
