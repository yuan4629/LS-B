#!/usr/bin/env bash
# ============================================================
# exp_timm.bash -- runs one job through the unified entry exp.py: checks the Python
# environment, then assembles the exp.py command line from environment variables.
#
# Output layout:
#   $BASE_OUT/$JOB_NAME/metrics.json   <- exp.py writes one file with all results (+ run.log)
#   ./outputs/.latest_exp_suite        <- records the last $BASE_OUT
# exp.py loops over methods x thresholds internally, so one invocation yields one
# metrics.json holding every (method, threshold) record.
#
# Usage (run from the repository root; run_d2.bash sets all of this for the D2 experiments):
#   bash exp_timm.bash                                          # smoke run (2 tasks, 2 steps)
#   PRESET=timm_cil_imagenet_r DATASET=imagenet_r NUM_TASKS=10 \
#     METHODS=bilora_d2 BIOSCORE_SPLIT_MODE=ratio_causal SEED=0 \
#     MAX_TRAIN_STEPS=0 MAX_EVAL_STEPS=0 JOB_NAME=d2_ratio_causal_inr_s0 bash exp_timm.bash
# ============================================================
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ENTRY="exp.py"

# --- Environment check ---------------------------------------------------------
# The paper's results were produced with the versions below. A different interpreter (e.g. a
# system python that happens to import everything) runs without error, but its numbers are not
# comparable with the other runs. exp.py's own environment guard only catches resumes, not new
# jobs, so the check is done here. PYTHON_BIN selects the interpreter; ALLOW_ENV_DRIFT=1 runs
# anyway (with a warning).
"$PYTHON_BIN" - "${ALLOW_ENV_DRIFT:-0}" <<'PYGUARD'
import sys
WANT = {"python": "3.13.13", "torch": "2.12.0+cu130", "numpy": "2.4.6",
        "timm": "1.0.27", "sklearn": "1.9.0"}          # versions used for the paper
got = {"python": ".".join(map(str, sys.version_info[:3]))}
for m in ("torch", "numpy", "timm", "sklearn"):
    try:
        got[m] = __import__(m).__version__
    except Exception as e:
        got[m] = f"unavailable({type(e).__name__})"
bad = {k: (v, got[k]) for k, v in WANT.items() if got[k] != v}
if bad:
    lines = "\n".join(f"    {k}: expected={w}  found={g}" for k, (w, g) in sorted(bad.items()))
    msg = (f"\n[ENV] ✗ environment differs from the versions used for the paper (interpreter={sys.executable}):\n{lines}\n"
           "      Results from another environment are not comparable (paired per-seed differences with other runs).\n"
           "      Fix: point PYTHON_BIN at the matching interpreter, e.g. PYTHON_BIN=/path/to/env/bin/python;\n"
           "      set ALLOW_ENV_DRIFT=1 only after confirming the results are comparable.")
    if sys.argv[1] == "1":
        print(msg + "\n[ENV] WARNING: ALLOW_ENV_DRIFT=1, running anyway (explicitly allowed)", flush=True)
    else:
        raise SystemExit(msg)
else:
    print(f"[ENV] ✓ environment matches the versions used for the paper (python {got['python']} / torch {got['torch']} / "
          f"timm {got['timm']} / numpy {got['numpy']} / sklearn {got['sklearn']})", flush=True)
PYGUARD

DATA_ROOT="${DATA_ROOT:-./data}"
FMRI_DATA_DIR="${FMRI_DATA_DIR:-./subj01}"
WSN_DIR="${WSN_DIR:-./WSN}"
SEED="${SEED:-0}"
PRECISION="${PRECISION:-fp32}"          # aligned default; PRECISION=amp to override
NUM_WORKERS="${NUM_WORKERS:-4}"

PRESET="${PRESET:-timm_cil_cifar100}"   # BiLoRA-aligned strict Class-IL preset
METHODS="${METHODS:-bilora_d2}"
SELECTOR_MODE="${SELECTOR_MODE:-grad}"  # inherited option (part of the identity key)
THRESHOLDS="${THRESHOLDS:-1.0}"         # looped over by exp.py for methods that use a threshold
NUM_TASKS="${NUM_TASKS:-2}"             # smoke=2; full aligned run sets 10
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2}" # smoke cap; 0 = full epoch
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-2}"   # smoke cap; 0 = full eval

# --- Output layout -----------------------------------------------------------
# BASE_OUT/<job>/metrics.json + ./outputs/.latest_exp_suite
BASE_OUT="${BASE_OUT:-./outputs/exp_suite_$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-exp_timm}"
JOB_OUT="$BASE_OUT/$JOB_NAME"
mkdir -p "$JOB_OUT" ./outputs
echo "$BASE_OUT" > ./outputs/.latest_exp_suite

# Optional env -> CLI pass-through: unset variables are not passed, so the preset/argparse
# defaults apply. INIT_EPOCH/EPOCHS shorten smoke runs; TIMM_MODEL/TIMM_WEIGHTS select the timm
# backbone and its offline weights; BACKBONE/ROUTE/JOINT_EPOCHS are inherited options, unused by
# the D2 pipeline. `if` is used instead of `&&` so that set -e does not exit on an unset variable.
EXTRA_ARGS=()
# DATASET selects the dataset (cifar100/imagenet_r/cub); unset = preset/argparse default (cifar100).
# Use PRESET=timm_cil_imagenet_r DATASET=imagenet_r for ImageNet-R (see the usage above).
if [ -n "${DATASET:-}" ];      then EXTRA_ARGS+=(--dataset "$DATASET"); fi
if [ -n "${BACKBONE:-}" ];     then EXTRA_ARGS+=(--backbone "$BACKBONE"); fi
if [ -n "${ROUTE:-}" ];        then EXTRA_ARGS+=(--route "$ROUTE"); fi
if [ -n "${INIT_EPOCH:-}" ];   then EXTRA_ARGS+=(--init_epoch "$INIT_EPOCH"); fi
if [ -n "${EPOCHS:-}" ];       then EXTRA_ARGS+=(--epochs "$EPOCHS"); fi
if [ -n "${JOINT_EPOCHS:-}" ]; then EXTRA_ARGS+=(--joint_epochs "$JOINT_EPOCHS"); fi
if [ -n "${TIMM_MODEL:-}" ];   then EXTRA_ARGS+=(--timm_model "$TIMM_MODEL"); fi
if [ -n "${TIMM_WEIGHTS:-}" ]; then EXTRA_ARGS+=(--timm_weights "$TIMM_WEIGHTS"); fi
# TIMM_WEIGHTS_FORMAT=ibot_pth|dino_pth selects self-supervised backbone weights.
# ORTHO_LAMBDA / ORTHO_MODE / BIOSCORE_SCORE_MODE / LORA_TARGETS / LORA_RANK / LORA_ALPHA /
# HEAD_LR / ADAPTER_LR: inherited options, unused by the D2 pipeline.
if [ -n "${TIMM_WEIGHTS_FORMAT:-}" ]; then EXTRA_ARGS+=(--timm_weights_format "$TIMM_WEIGHTS_FORMAT"); fi
if [ -n "${ORTHO_LAMBDA:-}" ];        then EXTRA_ARGS+=(--ortho_lambda "$ORTHO_LAMBDA"); fi
if [ -n "${ORTHO_MODE:-}" ];          then EXTRA_ARGS+=(--ortho_mode "$ORTHO_MODE"); fi
if [ -n "${BIOSCORE_SCORE_MODE:-}" ]; then EXTRA_ARGS+=(--bioscore_score_mode "$BIOSCORE_SCORE_MODE"); fi
if [ -n "${LORA_TARGETS:-}" ];        then EXTRA_ARGS+=(--lora_targets "$LORA_TARGETS"); fi
if [ -n "${LORA_RANK:-}" ];           then EXTRA_ARGS+=(--lora_rank "$LORA_RANK"); fi
if [ -n "${LORA_ALPHA:-}" ];          then EXTRA_ARGS+=(--lora_alpha "$LORA_ALPHA"); fi
if [ -n "${HEAD_LR:-}" ];             then EXTRA_ARGS+=(--head_lr "$HEAD_LR"); fi
if [ -n "${ADAPTER_LR:-}" ];          then EXTRA_ARGS+=(--adapter_lr "$ADAPTER_LR"); fi
# ALLOW_MOCK_ATLAS=1 allows a mock (random fallback) atlas, for smoke tests only (results are meaningless).
if [ "${ALLOW_MOCK_ATLAS:-0}" = "1" ]; then EXTRA_ARGS+=(--allow_mock_atlas); fi
# BRAINNET_FORCE_REBUILD=1 forces an atlas rebuild (ignores the disk cache, retrains the PLModel once
# and overwrites the cache). Normally unset: the atlas fingerprint triggers rebuilds when needed.
if [ "${BRAINNET_FORCE_REBUILD:-0}" = "1" ]; then EXTRA_ARGS+=(--brainnet_force_rebuild); fi
# BiLoRA: BILORA_WEIGHTS selects the backbone weights (empty = augreg npz, then the HF hub);
# BILORA_SKIP_INACTIVE=0 disables the zero-dW forward skip, to check that results are bit-identical
# with and without it (skipping should only affect wall-clock time).
if [ -n "${BILORA_WEIGHTS:-}" ];        then EXTRA_ARGS+=(--bilora_weights "$BILORA_WEIGHTS"); fi
if [ "${BILORA_SKIP_INACTIVE:-1}" = "0" ]; then EXTRA_ARGS+=(--no_bilora_skip_inactive); fi
# SELECTOR_LAYERS / MIN_SELECTED_LAYERS: layer-selection options of legacy methods (inherited options,
# unused by the D2 pipeline).
if [ -n "${SELECTOR_LAYERS:-}" ];       then EXTRA_ARGS+=(--selector_layers "$SELECTOR_LAYERS"); fi
if [ -n "${MIN_SELECTED_LAYERS:-}" ];   then EXTRA_ARGS+=(--min_selected_layers "$MIN_SELECTED_LAYERS"); fi
# BioScore v2 (brain readout): ATLAS_SEED / GAMMA / LAMBDA_DEPTH / ROI_WEIGHT / TASK_CENTER / VOXEL_TOPQ
# are in the exp.py identity key; VOXEL_CHUNK / REF_POOLS only affect memory / sampling precision.
# ATLAS_SEED default 0 = all CL seeds share one atlas (a brain prior should not change with the CL seed).
if [ -n "${ATLAS_SEED:-}" ];            then EXTRA_ARGS+=(--atlas_seed "$ATLAS_SEED"); fi
if [ -n "${BIOSCORE_GAMMA:-}" ];        then EXTRA_ARGS+=(--bioscore_gamma "$BIOSCORE_GAMMA"); fi
if [ -n "${BIOSCORE_LAMBDA_DEPTH:-}" ]; then EXTRA_ARGS+=(--bioscore_lambda_depth "$BIOSCORE_LAMBDA_DEPTH"); fi
if [ -n "${BIOSCORE_ROI_WEIGHT:-}" ];   then EXTRA_ARGS+=(--bioscore_roi_weight "$BIOSCORE_ROI_WEIGHT"); fi
if [ -n "${BIOSCORE_TASK_CENTER:-}" ];  then EXTRA_ARGS+=(--bioscore_task_center "$BIOSCORE_TASK_CENTER"); fi
if [ -n "${BIOSCORE_VOXEL_TOPQ:-}" ];   then EXTRA_ARGS+=(--bioscore_voxel_topq "$BIOSCORE_VOXEL_TOPQ"); fi
if [ -n "${BIOSCORE_VOXEL_CHUNK:-}" ];  then EXTRA_ARGS+=(--bioscore_voxel_chunk "$BIOSCORE_VOXEL_CHUNK"); fi
if [ -n "${BIOSCORE_REF_POOLS:-}" ];    then EXTRA_ARGS+=(--bioscore_ref_pools "$BIOSCORE_REF_POOLS"); fi
# EPOCH_SCALE (e.g. 0.25) = cheap proxy protocol (scales BiLoRA's init_epoch/epochs). It is in the
# identity key, so proxy runs never collide with full runs.
if [ -n "${EPOCH_SCALE:-}" ];           then EXTRA_ARGS+=(--epoch_scale "$EPOCH_SCALE"); fi
# D2 shared adapter (METHODS=bilora_d2, see run_d2.bash): the first four are in the exp.py identity
# key (oracle/causal arms and different k / lr_scale never share a done set); D2_ALLOC_FILE is read
# only by ratio_oracle (diag npz; the host computes the allocation with analyze_d2_split.allocate).
if [ -n "${BIOSCORE_SPLIT_MODE:-}" ];   then EXTRA_ARGS+=(--bioscore_split_mode "$BIOSCORE_SPLIT_MODE"); fi
if [ -n "${N_SPECIFIC_LAYERS:-}" ];     then EXTRA_ARGS+=(--n_specific_layers "$N_SPECIFIC_LAYERS"); fi
if [ -n "${SHARED_ADAPTER_LR_SCALE:-}" ]; then EXTRA_ARGS+=(--shared_adapter_lr_scale "$SHARED_ADAPTER_LR_SCALE"); fi
if [ -n "${SPLIT_FREEZE_TASK:-}" ];     then EXTRA_ARGS+=(--split_freeze_task "$SPLIT_FREEZE_TASK"); fi
if [ -n "${D2_ALLOC_FILE:-}" ];         then EXTRA_ARGS+=(--d2_alloc_file "$D2_ALLOC_FILE"); fi
# D2_ALLOC_SEED: allocation-draw seed of random_split, decoupled from the training SEED.
# Unset -> flag not passed -> argparse default None -> drawn with the training seed (reproduces
# earlier runs). Must allow 0: `[ -n "$X" ]` is true for "0" (non-empty), so this test is safe;
# something like `[ "$X" -gt 0 ]` would silently drop alloc_seed=0 (the [2,5,8,11] allocation).
if [ -n "${D2_ALLOC_SEED:-}" ];         then EXTRA_ARGS+=(--d2_alloc_seed "$D2_ALLOC_SEED"); fi
# D2_PIN_LAYERS: pinned layers of the pinned arm, canonical comma string such as "8,9,10,11".
# Unset -> not passed -> argparse default None -> command lines and identity keys of the other arms
# are unchanged. Only the pinned arm accepts it (other arms raise).
if [ -n "${D2_PIN_LAYERS:-}" ];         then EXTRA_ARGS+=(--d2_pin_layers "$D2_PIN_LAYERS"); fi
# CLASSES_PER_TASK: classes per task (set to 10 by run_d2.bash for ImageNet-R with T=20).
# Unset -> not passed -> taken from the preset (ImageNet-R/CUB=20, CIFAR=10), so the command lines
# and identity keys of the other runs are unchanged. If set, it is passed right after --num_tasks.
# core/cli never lets the preset override explicit values, and on ImageNet-R/CUB
# num_tasks x classes_per_task != 200 raises (no silent T=10 or half-dataset run).
# --num_tasks itself is always passed (NUM_TASKS defaults to 2 = smoke run).
TASK_ARGS=()
if [ -n "${CLASSES_PER_TASK:-}" ];      then TASK_ARGS+=(--classes_per_task "$CLASSES_PER_TASK"); fi

"$PYTHON_BIN" "$ENTRY" \
  --preset "$PRESET" \
  --data_root "$DATA_ROOT" \
  --fmri_data_dir "$FMRI_DATA_DIR" \
  --wsn_dir "$WSN_DIR" \
  --seed "$SEED" \
  --precision "$PRECISION" \
  --num_workers "$NUM_WORKERS" \
  --methods $METHODS \
  --selector_mode "$SELECTOR_MODE" \
  --thresholds $THRESHOLDS \
  --num_tasks "$NUM_TASKS" \
  ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --max_eval_steps "$MAX_EVAL_STEPS" \
  --output_dir "$JOB_OUT" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo
echo "=== Finished ==="
echo "Output : $JOB_OUT/metrics.json"
echo "Suite  : $BASE_OUT"
echo "Log    : $JOB_OUT/run.log"
