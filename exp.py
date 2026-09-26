# -*- coding: utf-8 -*-
"""Unified experiment entry (orchestration layer).

Writes metrics.json as {"config", "env", "task_classes", "results"}. Methods are
resolved through core.registry.REGISTRY -- adding a method = add a folder +
register one line.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid the duplicate-OpenMP-runtime abort

import json
import sys
from datetime import datetime
from pathlib import Path

from core.cli import build_argparser, apply_preset, set_seed, get_device
from core.data import build_split
from core.plotting import plot_results
from core.registry import get_method


# Resume: an existing metrics.json is continued only if all identity keys match.
# Every config that affects results must be listed here; otherwise a run with another
# backbone or hyperparameter would be mistaken for a finished unit and skipped.
_IDENTITY_KEYS = (
    "preset", "dataset", "seed", "num_tasks", "classes_per_task", "task_order", "backbone",
    "timm_model", "timm_weights", "timm_weights_format",  # backbone weight identity (AugReg vs iBOT)
    # selector_layers / min_selected_layers decide which layers are actually adapted:
    # in a per-layer sweep l=6 and l=7 share the same threshold and would otherwise collide.
    "route", "selector_mode", "selector_layers", "min_selected_layers",
    "bioscore_score_mode", "precision", "lr", "epochs", "init_epoch",
    "lr_schedule", "max_train_steps", "max_eval_steps",
    # Method hyperparameters that affect results: per-group lr, LoRA capacity, injection
    # targets, orthogonality strength/form (joint_epochs is legacy).
    "head_lr", "adapter_lr", "lora_rank", "lora_alpha", "lora_targets", "ortho_lambda", "ortho_mode", "joint_epochs",
    # BiLoRA backbone weights (AugReg .npz vs iBOT .pth must be distinguished).
    # bilora_skip_inactive is deliberately not keyed: it is an exact skip that changes
    # wall-clock time only, not results.
    "bilora_weights",
    # BioScore v2: these decide which layers are selected; unkeyed, a gamma/lambda sweep
    # would skip every point after the first. atlas_seed picks the atlas; voxel_topq changes
    # results and is keyed, voxel_chunk (pure chunking) and ref_pools (sampling precision) are not.
    "atlas_seed", "bioscore_gamma", "bioscore_lambda_depth", "bioscore_roi_weight",
    "bioscore_voxel_topq", "bioscore_task_center",
    # Epoch scaling of the proxy protocol: proxy and full runs must never be merged.
    "epoch_scale",
    # D2 shared adapter (bilora_d2): these four decide the shared/specific allocation and the
    # shared-slot training; unkeyed, oracle vs causal and different k / lr_scale arms collide.
    # d2_alloc_file is deliberately not keyed: the path is not stable across machines; the
    # allocation itself is logged on the [gate] line and can be recomputed independently.
    "bioscore_split_mode", "n_specific_layers", "shared_adapter_lr_scale", "split_freeze_task",
    # d2_alloc_seed decides which layers random_split draws and **must be keyed**: two draws
    # under the same training seed would otherwise be treated as one unit (and silently
    # overwrite each other under the same job name). Older metrics.json without it (None)
    # equals the argparse default (None), so existing resume decisions are unchanged.
    "d2_alloc_seed",
    # d2_pin_layers (layers committed at t=K by the pinned arm, canonical comma string such as
    # "8,9,10,11") **must be keyed**: deep and shallow pinned runs differ only in this key, and
    # it (with bioscore_split_mode) separates a pinned run from the best_approximation run with
    # the same k/K/seed. None means "not pinned", so it is not in _V2_KEY_DEFAULTS.
    "d2_pin_layers",
)

# Keys added later are absent (None) in older metrics.json files; normalize them to the
# argparse defaults so resume decisions for older runs are unchanged.
_V2_KEY_DEFAULTS = {
    "atlas_seed": 0, "bioscore_gamma": 0.0, "bioscore_lambda_depth": 0.0,
    "bioscore_roi_weight": "uniform", "bioscore_voxel_topq": 500, "epoch_scale": 1.0,
    "bioscore_task_center": "running",
    # D2 keys: missing (None) -> argparse default (checked in tests/test_d2.py [3]).
    "bioscore_split_mode": "none", "n_specific_layers": 4,
    "shared_adapter_lr_scale": 0.1, "split_freeze_task": 3,
}


def _identity(cfg):
    ident = {k: cfg.get(k) for k in _IDENTITY_KEYS}
    # Keys added over time are absent (None) in older metrics.json; map them to their
    # semantically equivalent defaults so older runs still resume.
    ident["ortho_mode"] = ident["ortho_mode"] or "raw"
    ident["bioscore_score_mode"] = ident["bioscore_score_mode"] or "last"
    ident["bilora_weights"] = ident["bilora_weights"] or ""  # "" = auto
    ident["selector_layers"] = ident["selector_layers"] or ""  # "" = no explicit layer list
    # min_selected_layers: fall back to the argparse default 2 if missing.
    ident["min_selected_layers"] = 2 if ident["min_selected_layers"] is None else ident["min_selected_layers"]
    for k, v in _V2_KEY_DEFAULTS.items():
        if ident.get(k) is None:
            ident[k] = v
    return ident


def _env_fingerprint():
    """Runtime environment fingerprint.

    Deliberately NOT in _IDENTITY_KEYS (a torch patch release would otherwise invalidate
    every resume), but always saved: runs from different environments mixed into one
    paired difference are otherwise undetectable. _load_resume uses it to block
    cross-environment resumes (see below)."""
    env = {"python": sys.version.split()[0]}
    # sklearn is included because the atlas encoding model uses it; a major-version
    # change can alter the allocation.
    for mod in ("torch", "numpy", "timm", "sklearn"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception as e:                      # record failure explicitly, never leave it empty
            env[mod] = f"unavailable({type(e).__name__})"
    try:
        import torch
        env["cuda"] = torch.version.cuda or "cpu"
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception as e:
        env["cuda"] = env["gpu"] = f"unavailable({type(e).__name__})"
    # Code version: any change to the training path during a batch is within-batch drift.
    # Recording code_sha per run makes it detectable afterwards. code_dirty=True means the
    # run used an uncommitted working tree, so the sha alone does not identify the code.
    try:
        import subprocess
        _here = str(Path(__file__).resolve().parent)
        _run = lambda a: subprocess.run(a, cwd=_here, capture_output=True, text=True, timeout=10)
        _p = _run(["git", "rev-parse", "HEAD"])
        env["code_sha"] = _p.stdout.strip()[:12] if _p.returncode == 0 else f"unavailable(rc={_p.returncode})"
        _q = _run(["git", "status", "--porcelain"])
        env["code_dirty"] = bool(_q.stdout.strip()) if _q.returncode == 0 else f"unavailable(rc={_q.returncode})"
    except Exception as e:
        env["code_sha"] = env["code_dirty"] = f"unavailable({type(e).__name__})"
    return env


_ENV = _env_fingerprint()


def _atomic_dump(path, payload):
    # Atomic write (.tmp then os.replace): a crash never leaves a truncated metrics.json.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_resume(metrics_path, cur_cfg):
    """Return (results, done). Identity keys match -> resume (preload old results and the
    done set); otherwise back up the old file and start from scratch.
    done is keyed by _run_name (requested name), not method (output label); the two can differ."""
    if not metrics_path.exists():
        return [], set()
    try:
        with open(metrics_path, encoding="utf-8") as f:
            prev = json.load(f)
    except Exception as e:
        print(f"[RESUME] failed to read old metrics.json ({e}); starting from scratch.")
        return [], set()
    if _identity(prev.get("config", {})) == _identity(cur_cfg):
        results = prev.get("results", [])
        done = {(r.get("_run_name", r.get("method")), r.get("threshold")) for r in results}
        # Cross-environment guard: matching identity keys do not make numbers comparable.
        # If the old units' torch/numpy/timm versions are unknown (no env field), mixing them
        # with new units makes per-seed paired differences cross-environment -- invisible to
        # any downstream check.
        old_env = prev.get("env")
        if done and old_env != _ENV:
            shown = old_env or "not recorded (run predates env logging)"
            if os.environ.get("ALLOW_ENV_DRIFT") != "1":
                raise RuntimeError(
                    f"[ENV] refusing to resume across environments: {metrics_path}\n"
                    f"  environment of the {len(done)} finished units = {shown}\n"
                    f"  current environment = {_ENV}\n"
                    "  Identity keys match but environments differ; resuming would mix two batches into one paired difference.\n"
                    "  Fix: rerun under a new JOB name (recommended), or set ALLOW_ENV_DRIFT=1 after confirming comparability.")
            print(f"[ENV] WARNING: ALLOW_ENV_DRIFT=1, explicitly allowing a cross-environment resume (old={shown} new={_ENV})")
        print(f"[RESUME] identity keys match; {len(done)} finished units found, they will be skipped.")
        return results, done
    bak = metrics_path.with_name(f"metrics.json.bak.{datetime.now():%Y%m%d_%H%M%S}")
    os.replace(metrics_path, bak)
    print(f"[RESUME] identity keys differ; old results backed up to {bak.name}, starting from scratch.")
    return [], set()


class _Tee:
    """Mirror a stream (stdout/stderr) to a log file; console output is unchanged."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._fh.flush()  # flush immediately so the last output survives a crash

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):  # forward other attributes (isatty, ...) to the original stream
        return getattr(self._stream, name)


def _open_run_log(out_dir):
    """Append this run's log to out_dir/run.log (with a separator if a log exists). Returns (fh, orig_stdout, orig_stderr)."""
    log_path = out_dir / "run.log"
    had_previous = log_path.exists() and log_path.stat().st_size > 0
    fh = open(log_path, "a", encoding="utf-8")  # append, do not overwrite earlier runs
    if had_previous:
        fh.write("\n\n" + "=" * 78 + "\n")
    fh.write(f"===== RUN {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    fh.write("CMD: " + " ".join(sys.argv) + "\n\n")
    fh.flush()
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_stdout, fh)
    sys.stderr = _Tee(orig_stderr, fh)
    print(f"Logging to: {log_path.resolve()}")
    return fh, orig_stdout, orig_stderr


def main():
    args = apply_preset(build_argparser().parse_args())
    set_seed(args.seed)
    device = get_device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_fh, orig_stdout, orig_stderr = _open_run_log(out_dir)
    try:
        _run(args, device, out_dir)
    finally:  # restore streams and close the log even on error (the traceback already went through the stderr tee)
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        log_fh.close()


def _run(args, device, out_dir):
    print("=== Continual Learning General Test (exp.py) ===")
    print(f"Device: {device}")
    print(f"Preset: {args.preset}")
    print(f"Task order: {args.task_order}")
    print(f"Validation split: {args.val_split}")
    print(f"Output: {out_dir.resolve()}")

    train_loaders, val_loaders, test_loaders, task_classes, ncls = build_split(args)
    print(f"Prepared {args.dataset} split: {args.num_tasks} tasks x {args.classes_per_task} classes")

    metrics_path = out_dir / "metrics.json"
    cur_cfg = vars(args)
    results, done = _load_resume(metrics_path, cur_cfg)

    def _persist():
        _atomic_dump(metrics_path, {"config": cur_cfg, "env": _ENV,
                                    "task_classes": task_classes, "results": results})

    for name in args.methods:
        module = get_method(name)
        spec = getattr(module, "METHOD_SPEC", {})
        thresholds = args.thresholds if spec.get("needs_threshold") else [None]
        for th in thresholds:
            if (name, th) in done:
                print(f"[SKIP] {name} threshold={th} already in metrics.json, skipping.")
                continue
            if th is None:
                res = module.run(train_loaders, val_loaders, test_loaders, ncls, args, device)
            else:
                res = module.run(train_loaders, val_loaders, test_loaders, ncls, args, device, threshold=th)
            # Final trainable tensors attached by the runner must be popped before JSON
            # serialization (tensors are not serializable; leaving them makes _persist fail
            # loudly, by design, rather than dropping them silently).
            _ft = res.pop("_final_trainable", None)
            if _ft is not None:
                import torch as _torch
                _ckpt_path = out_dir / "final_trainable.pt"
                _torch.save(_ft, _ckpt_path)
                print(f"[ckpt] final_trainable.pt: tensors={len(_ft)} "
                      f"bytes={_ckpt_path.stat().st_size}")
            res["_run_name"] = name  # resume skip key (requested name, distinct from the output label method)
            results.append(res)
            done.add((name, th))
            _persist()  # atomic save after each unit; a crash loses at most the unit in progress

    plot_results(results, out_dir)
    _persist()

    print("\n=== Summary ===")
    for result in results:
        name = result["method"] if result["threshold"] is None else f"{result['method']}(th={result['threshold']})"
        print(f"{name:24s} final_ACC={result['acc_curve'][-1]:.4f} final_BWT={result['bwt_curve'][-1]:.4f} runtime={result['runtime_sec']:.1f}s")
    print(f"\nSaved figures and metrics to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
