"""Continual-learning metric helpers."""
import numpy as np

from core.train import eval_model


def update_matrix_and_metrics(model, tid, test_loaders, matrix, device, max_eval_steps, task_aware=False):
    # Strict Class-IL (task_aware=False): argmax over seen classes only, as in BiLoRA.
    # task_aware=True (TIL) uses per-task heads without a mask.
    restrict = None if task_aware else True
    sample_counts = []
    for j in range(tid + 1):
        eval_task = j if task_aware else None
        matrix[tid, j], sample_count = eval_model(
            model,
            test_loaders[j],
            device,
            max_eval_steps,
            task_id=eval_task,
            restrict_to_seen=restrict,
            return_stats=True,
        )
        sample_counts.append(sample_count)
    # Class-IL ACC is computed over the pooled test set of all seen classes;
    # the matrix is kept only for BWT / forgetting.
    weights = np.asarray(sample_counts, dtype=np.float32)
    acc = float(np.nansum(matrix[tid, : tid + 1] * weights) / max(weights.sum(), 1.0))
    if tid == 0:
        return acc, 0.0
    diag = np.diag(matrix)[:tid]
    bwt = float(np.nanmean(matrix[tid, :tid] - diag))
    return acc, bwt


def matrix_to_jsonable(matrix):
    out = []
    for row in matrix:
        out.append([None if np.isnan(v) else float(v) for v in row.tolist()])
    return out


# ============================================================
# Relative-gap metrics (backbone-agnostic). Pure functions on aggregated ACC,
# so they can be applied at merge time using same-suite Joint/Seq anchors.
# ============================================================
def gap_closed_ratio(acc, seq, joint):
    """Gap-Closed Ratio = (Acc - Seq) / (Joint - Seq).

    Normalized relative gap: how much of the Seq(lower-bound)->Joint(upper-bound)
    gap a run closes, on the SAME suite/backbone.
    Guard: if Joint/Seq missing or Joint-Seq ~0 (degenerate denominator), return nan
    so a suite lacking anchors merges gracefully instead of dividing by zero.
    """
    if acc is None or seq is None or joint is None:
        return float("nan")
    denom = joint - seq
    # Near-zero denominator (bounds coincide, no dynamic range): GCR is undefined.
    if abs(denom) < 1e-12:
        return float("nan")
    return (acc - seq) / denom


def pct_of_joint(acc, joint):
    """%-of-Joint = Acc / Joint (fraction of the upper bound reached).

    Guard: if Joint missing or ~0, return nan (no division by zero; missing anchors degrade gracefully).
    """
    if acc is None or joint is None or abs(joint) < 1e-12:
        return float("nan")
    return acc / joint
