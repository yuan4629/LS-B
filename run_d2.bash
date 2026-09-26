#!/usr/bin/env bash
# ============================================================
# run_d2.bash -- batch driver for the D2 shared adapter (shared/specific layer allocation).
#
# Runs method bilora_d2 through exp_timm.bash for one backbone (BK) and one dataset (DATASET),
# looping over seeds and allocation arms. Each run writes $BASE_OUT/<job>/metrics.json and
# run.log; job names have the form
#   d2_<arm>_<inr|c100|cub>[_t20][_ibot|_dino]_s<seed>[_a<alloc_seed>][_fast]
#
# STAGE selects the arms:
#   ENDPOINT    all_shared / all_specific / random_split (endpoint sensitivity)
#   MAIN        ratio_causal (brain-derived allocation) / ratio_oracle / random_split
#   GRID        fixed_depth_l<l>: the shallow l layers are specific, l in GRID_LS ("2 4 6 8 10")
#   CALIB       random_split with two fixed allocations (alloc seeds 0 and 2), varying training seed
#   RAND2       random_split with a second allocation draw per seed (alloc seed = seed + 100)
#   WINDOW      fixed_window_w<W>: k=4 sliding window, W in WINDOWS ("1 2 3 4 5 6 7")
#   DEEP        fixed_depth_deep: the deep DEEPN layers are specific (DEEPN=4 or 10, required)
#   BESTAPPROX  best_approximation_k<k>f<K>: causal allocation for k in BA_KS ("4"), K in BA_KFS
#               ("3 1 2"); k=4, K=3 must reproduce ratio_causal
#   PINNED      pinned_k4f3_<set>: causal warm-up, then fixed layers; PIN_SETS (required) is
#               "deep" (8,9,10,11) and/or "shallow" (0,1,2,3)
#
# Environment variables (defaults in brackets):
#   STAGE [ENDPOINT]   BK=augreg|ibot|dino [augreg]   DATASET=inr|c100|cub [inr]   SEEDS ["0 1 2"]
#   NTASKS=10|20 [10]  20 = ImageNet-R with 20 tasks x 10 classes (DATASET=inr, BK=augreg, and only
#                      all_shared, all_specific, fixed_depth_l{2,4,6}, best_approximation_k{2,4}f3)
#   N_SPECIFIC [4] (k)   FREEZE_T [3] (K, warm-up tasks)   LR_SCALE [0.1] (shared-slot lr factor)
#   D2_ALLOC_FILE [./outputs/atlas_v2_diag_k4_b10.npz] diag npz for ratio_oracle (CIFAR-100 by
#                 default; pass the file of the dataset in use)
#   BASE_OUT [./outputs/exp_d2, or ./outputs/exp_d2_t20 for NTASKS=20]
#   AUGREG_NPZ / IBOT_PTH / DINO_PTH   backbone weights (under ./pretrained/)
#   ONLY="arm ..."  run only these arm names      FAST=1  2-step smoke run (suffix _fast)
# The ratio_* arms need the diag npz of the dataset, e.g. for ImageNet-R:
#   DATASET=inr DUMP=./outputs/atlas_v2_diag_inr_k4_b10.json python baselines/bilora_adapter/atlas_v2_diag.py
#
# Examples (run from the repository root):
#   STAGE=ENDPOINT BK=augreg DATASET=inr bash run_d2.bash
#   STAGE=MAIN BK=augreg DATASET=inr D2_ALLOC_FILE=./outputs/atlas_v2_diag_inr_k4_b10.npz bash run_d2.bash
#   STAGE=PINNED BK=ibot DATASET=inr PIN_SETS="deep shallow" SEEDS="0 1 2" bash run_d2.bash
# ============================================================
set -euo pipefail

# NTASKS: number of tasks T. Unset = 10; its validity is checked after the dataset branch.
# T=20 uses a separate default output root (and _t20 in the job name), so the two never mix.
NTASKS="${NTASKS:-10}"
if [ "$NTASKS" = "20" ]; then
  BASE_OUT="${BASE_OUT:-./outputs/exp_d2_t20}"
else
  BASE_OUT="${BASE_OUT:-./outputs/exp_d2}"
fi
STAGE="${STAGE:-ENDPOINT}"
BK="${BK:-augreg}"
DATASET="${DATASET:-inr}"
AUGREG_NPZ="${AUGREG_NPZ:-./pretrained/vit_b16_augreg_in21k.npz}"
IBOT_PTH="${IBOT_PTH:-./pretrained/ibot_vitb16.pth}"
DINO_PTH="${DINO_PTH:-./pretrained/dino_vitbase16_pretrain.pth}"   # third backbone
SEEDS="${SEEDS-0 1 2}"                       # seeds (paired across arms)
N_SPECIFIC="${N_SPECIFIC:-4}"                 # k_specific (main setting 4)
FREEZE_T="${FREEZE_T:-3}"                     # K = number of warm-up tasks of the causal arms
LR_SCALE="${LR_SCALE:-0.1}"                   # shared-slot lr = task lr x 0.1
# Allocation table of ratio_oracle (diag npz). The default is the CIFAR-100 file; pass the
# matching file for other datasets.
D2_ALLOC_FILE="${D2_ALLOC_FILE:-./outputs/atlas_v2_diag_k4_b10.npz}"

# Backbone identity: the scoring backbone (TIMM_*) and the training backbone (BILORA_WEIGHTS) must
# use the same weights, otherwise layer indices do not transfer.
# Note: job names carry a backbone suffix; without it augreg and iBOT runs would collide
# (identity mismatch -> backup and rerun).
case "$BK" in
  augreg) BK_ENV=(TIMM_MODEL=vit_base_patch16_224.augreg_in21k
                  TIMM_WEIGHTS="$AUGREG_NPZ" TIMM_WEIGHTS_FORMAT=augreg_npz
                  BILORA_WEIGHTS="$AUGREG_NPZ"); BK_SUF="" ;;
  ibot)   BK_ENV=(TIMM_MODEL=vit_base_patch16_224
                  TIMM_WEIGHTS="$IBOT_PTH" TIMM_WEIGHTS_FORMAT=ibot_pth
                  BILORA_WEIGHTS="$IBOT_PTH"); BK_SUF="_ibot" ;;
  dino)   # third backbone. Same timm base model as iBOT (vit_base_patch16_224, no suffix),
          # loaded through the dino_pth key remapping (timm_backbone.build_timm_vit).
          BK_ENV=(TIMM_MODEL=vit_base_patch16_224
                  TIMM_WEIGHTS="$DINO_PTH" TIMM_WEIGHTS_FORMAT=dino_pth
                  BILORA_WEIGHTS="$DINO_PTH"); BK_SUF="_dino" ;;
  *) echo "unknown BK=$BK (valid: augreg/ibot/dino)"; exit 1 ;;
esac

# Dataset: preset, dataset name and job suffix are bound together; never change only one of them.
# Every branch sets DATASET explicitly (the CLI only accepts cifar100/imagenet_r/cub).
# cub = CUB-200 (APER split).
case "$DATASET" in
  inr)  DS_ENV=(PRESET=timm_cil_imagenet_r DATASET=imagenet_r); DS_SUF="_inr" ;;
  c100) DS_ENV=(PRESET=timm_cil_cifar100 DATASET=cifar100);     DS_SUF="_c100" ;;
  cub)  DS_ENV=(PRESET=timm_cil_cub DATASET=cub);               DS_SUF="_cub" ;;
  *) echo "unknown DATASET=$DATASET (valid: inr/c100/cub)"; exit 1 ;;
esac

# Number of tasks T: only 10 (default) and 20 on ImageNet-R are supported.
# The classes per task follow from T (ImageNet-R has 200 classes: T=10 -> 20 from the preset,
# T=20 -> 10 passed explicitly) and may not be set by the caller; otherwise a job name could say
# t20 while the run uses T=10 (core/cli also raises if T x classes != 200).
if [ -n "${CLASSES_PER_TASK:-}" ]; then
  echo "CLASSES_PER_TASK=$CLASSES_PER_TASK: classes per task are set by this script from NTASKS (T=10 -> 20 from the preset, T=20 -> 10); callers may not set it -> refusing"
  exit 1
fi
case "$NTASKS" in
  10) T_SUF=""; T_ENV=() ;;
  20)
    if [ "$DATASET" != "inr" ]; then
      echo "NTASKS=20 is only supported for DATASET=inr, got DATASET=$DATASET -> refusing"
      exit 1
    fi
    T_SUF="_t20"; T_ENV=(CLASSES_PER_TASK=10) ;;
  *) echo "NTASKS=$NTASKS is not supported: only 10 (default) or 20 on DATASET=inr -> refusing"; exit 1 ;;
esac
# The output root must differ per T: an explicit BASE_OUT pointing T=20 into the T=10 root (or
# vice versa) is refused. The _t20 suffix prevents name collisions, the separate root prevents
# mixing batches. Only these two roots are guarded; other custom roots are allowed.
if [ "$NTASKS" = "20" ] && [ "$(realpath -m -- "$BASE_OUT")" = "$(realpath -m -- ./outputs/exp_d2)" ]; then
  echo "NTASKS=20 with BASE_OUT=$BASE_OUT points at the T=10 output root -> refusing (T=20 default: ./outputs/exp_d2_t20)"
  exit 1
fi
if [ "$NTASKS" = "10" ] && [ "$(realpath -m -- "$BASE_OUT")" = "$(realpath -m -- ./outputs/exp_d2_t20)" ]; then
  echo "T=10 with BASE_OUT=$BASE_OUT points at the T=20 output root -> refusing"
  exit 1
fi
# The 7 configurations supported for T=20, as job tokens. With NTASKS=20 a dry pass first collects
# every token of this invocation; if any is not listed, the whole batch is refused before a single
# run starts (see the two stage_main calls at the end of the file).
T20_REGISTERED="all_shared all_specific fixed_depth_l2 fixed_depth_l4 fixed_depth_l6 best_approximation_k2f3 best_approximation_k4f3"
_PLAN_PASS=0          # internal (not read from the environment): 1 = dry pass, run() only collects tokens
PLAN_TOKS=()

# SELECTOR_MODE is irrelevant for bilora_d2 but part of the identity key, so it is pinned to grad
# (a drifting default would make the same job look like a different identity and trigger a rerun).
# NUM_TASKS comes from NTASKS; T_ENV is CLASSES_PER_TASK=10 for T=20 and empty for T=10.
COMMON_ENV=(
  NUM_TASKS="$NTASKS"
  ${T_ENV[@]+"${T_ENV[@]}"}
  "${DS_ENV[@]}"
  "${BK_ENV[@]}"
  METHODS=bilora_d2 SELECTOR_MODE=grad THRESHOLDS=1.0
  N_SPECIFIC_LAYERS="$N_SPECIFIC" SPLIT_FREEZE_TASK="$FREEZE_T"
  SHARED_ADAPTER_LR_SCALE="$LR_SCALE"
  ATLAS_SEED="${ATLAS_SEED:-0}"
  BASE_OUT="$BASE_OUT"
)
if [ "${FAST:-0}" = "1" ]; then
  STEPS_ENV=(MAX_TRAIN_STEPS=2 MAX_EVAL_STEPS=2); SUF="_fast"
else
  STEPS_ENV=(MAX_TRAIN_STEPS=0 MAX_EVAL_STEPS=0); SUF=""
fi

run() {   # run <arm> <seed> [extra env...]
  local arm="$1" sd="$2"; shift 2
  # The fixed_depth_l job name must contain l (different grid points are different jobs).
  # l is set by the GRID branch in GRID_L (not via a `VAR=x func` prefix, whose semantics for
  # functions are fragile in bash).
  local tok="$arm"
  [ "$arm" = "fixed_depth_l" ] && tok="fixed_depth_l${GRID_L:?fixed_depth_l needs GRID_L}"
  # Same for fixed_depth_deep (n=4 -> [8,9,10,11], n=10 -> [2..11]): without l both n would share
  # one job name, and the second would be judged an identity mismatch and overwrite the first.
  [ "$arm" = "fixed_depth_deep" ] && tok="fixed_depth_deep_l${GRID_L:?fixed_depth_deep needs GRID_L}"
  # Same for best_approximation: k and K must both be in the name (k4f3 and k4f1 differ by
  # 2x(12-4)=16 persistent parameter groups). The separator is f (freeze), not K, because
  # Windows file systems are case-insensitive.
  [ "$arm" = "best_approximation" ] && tok="best_approximation_k${BA_K:?best_approximation needs BA_K}f${BA_KF:?best_approximation needs BA_KF}"
  # pinned: k, K and the pin-set name all go into the name; deep and shallow runs differ only in
  # the pinned layers and would otherwise overwrite each other.
  [ "$arm" = "pinned" ] && tok="pinned_k${PIN_K:?pinned needs PIN_K}f${PIN_KF:?pinned needs PIN_KF}_${PIN_SET:?pinned needs PIN_SET}"
  # If D2_ALLOC_SEED is set it must be in the job name: two draws under the same training seed
  # would otherwise share a directory, and since d2_alloc_seed is in the exp.py identity key the
  # second would restart from scratch and overwrite the first.
  local asuf=""
  [ -n "${ALLOC_SEED_TAG:-}" ] && asuf="_a${ALLOC_SEED_TAG}"
  # T_SUF: _t20 for T=20, after the dataset suffix and before the backbone suffix; empty for T=10.
  local tag="d2_${tok}${DS_SUF}${T_SUF}${BK_SUF}_s${sd}${asuf}"
  if [ -n "${ONLY:-}" ]; then          # ONLY matches arm names exactly
    local hit=0
    for w in $ONLY; do case "$arm" in "$w") hit=1 ;; esac; done
    [ "$hit" = "1" ] || { echo "--- skip $tag (ONLY=$ONLY)"; return 0; }
  fi
  # Dry pass (only during the NTASKS=20 batch pre-check, see the end of the file): collect the
  # token and start nothing.
  if [ "$_PLAN_PASS" = "1" ]; then PLAN_TOKS+=("$tok"); return 0; fi
  echo "=== ${tag}${SUF} (GPU=${CUDA_VISIBLE_DEVICES:-?}) ==="
  env "${COMMON_ENV[@]}" "${STEPS_ENV[@]}" "$@" \
    JOB_NAME="${tag}${SUF}" BIOSCORE_SPLIT_MODE="$arm" SEED="$sd" \
    bash exp_timm.bash
}

# The arm tables of all stages are wrapped in a function: with NTASKS=20 it is first called in a
# dry pass (_PLAN_PASS=1) to pre-check the whole batch, then for real; with T=10 it is called once.
stage_main() {
case "$STAGE" in
  ENDPOINT)
    # Endpoint arms; they need no atlas/npz.
    for sd in $SEEDS; do
      run all_shared    "$sd"
      run all_specific  "$sd"
      run random_split  "$sd"
    done
    ;;
  MAIN)
    # Main matrix. The ratio arms need the diag npz of the dataset.
    for sd in $SEEDS; do
      run ratio_causal  "$sd"
      run ratio_oracle  "$sd" D2_ALLOC_FILE="$D2_ALLOC_FILE"
      run random_split  "$sd"
      # fixed_depth (k=4, specific=[0,1,2,3]) is not run here: it is the same experiment as
      # fixed_depth_l4 from STAGE=GRID, which is reused.
    done
    ;;
  GRID)
    # Depth grid: the shallow l layers are specific, l in {2,4,6,8,10} (l=0/12 are the
    # all_shared/all_specific endpoints; do not rerun them).
    # N_SPECIFIC_LAYERS is appended after COMMON_ENV via "$@"; env keeps the last value, so l applies.
    for sd in $SEEDS; do
      for L in ${GRID_LS:-2 4 6 8 10}; do   # GRID_LS narrows the l points (e.g. "4 6 8")
        GRID_L="$L"
        run fixed_depth_l "$sd" N_SPECIFIC_LAYERS="$L"
      done
    done
    ;;
  CALIB)
    # Calibration: fix the allocation and vary only the training seed. This separates the run
    # noise of a fixed allocation from the allocation effect (the two are confounded when every
    # seed draws its own allocation).
    #
    # 2 allocations x 3 training seeds per backbone; the cells where alloc_seed equals the
    # training seed already exist in the ENDPOINT batch:
    #   alloc_seed=0 -> [2,5,8,11], ENDPOINT has training seed 0 -> add seeds 1,2
    #   alloc_seed=2 -> [1,6,9,10], ENDPOINT has training seed 2 -> add seeds 0,1
    # Note: these runs use two chosen allocations, not random draws, so they must not be pooled
    # with the random_split control arm.
    for A in 0 2; do
      for sd in $SEEDS; do
        # Skip the cell already run by ENDPOINT (alloc_seed == training seed).
        if [ "$A" = "$sd" ]; then
          echo "--- skip alloc=$A seed=$sd (same as the existing ENDPOINT run d2_random_split${DS_SUF}${BK_SUF}_s${sd})"
          continue
        fi
        ALLOC_SEED_TAG="$A" run random_split "$sd" D2_ALLOC_SEED="$A"
      done
    done
    ;;
  RAND2)
    # Second allocation draw for random_split (m=2). Each training seed gets a different
    # allocation; averaging the two draws halves the variance term (sigma^2_run + sigma^2_alloc)
    # of the brain-vs-random difference. Estimator, thresholds and alpha are unchanged; this only
    # reduces variance.
    #
    # alloc_seed = training seed + 100: disjoint from the first draw (alloc_seed=None uses the
    # training seed), and the `_a<N>` job suffix keeps the names apart.
    #
    # Note: schedule this batch last; once any of its runs has started it must not be withdrawn
    # (stopping half-way would be optional stopping).
    for sd in $SEEDS; do
      A=$(( sd + 100 ))
      ALLOC_SEED_TAG="$A" run random_split "$sd" D2_ALLOC_SEED="$A"
    done
    ;;
  WINDOW)
    # Sliding-window profile with fixed k=4 (w1..w7; w0/w8 reuse fixed_depth_l4 and
    # fixed_depth_deep(n=4)). WINDOWS narrows the windows (e.g. to fill gaps), like SEEDS.
    for sd in $SEEDS; do
      for W in ${WINDOWS:-1 2 3 4 5 6 7}; do
        run "fixed_window_w${W}" "$sd"
      done
    done
    ;;
  DEEP)
    # fixed_depth_deep. DEEPN=4: the w8 window endpoint and the deep-side heuristic (also the
    # alpha/ESD top-4 on AugReg/iBOT); DEEPN=10: k ablation.
    : "${DEEPN:?STAGE=DEEP needs DEEPN (4 or 10)}"
    for sd in $SEEDS; do
      GRID_L="$DEEPN"
      run fixed_depth_deep "$sd" N_SPECIFIC_LAYERS="$DEEPN"
    done
    ;;
  BESTAPPROX)
    # Capacity-accuracy Pareto ablation (exploratory). Same procedure as ratio_causal (K warm-up
    # tasks -> allocation from the first K tasks' D_t -> frozen), sweeping k and K.
    #
    # Persistent parameter groups P(k,K) = 12(K+1) + k(T-K-1); with T=10, K=3: P = 6k+48
    #   k=1->54  k=2->60  k=3->66  k=4->72  k=5->78  k=6->84  k=7->90  k=8->96   (all_specific = 120)
    # The grid (fixed_depth_l) has P = 9l+12; the two lines have exactly equal capacity only at
    # 66 (k=3 <-> l=6) and 84 (k=6 <-> l=8); other points are compared against the grid's
    # capacity-interpolation line.
    #
    # K axis (BA_KS="4" BA_KFS="1 2"): k=4 fixed, K varies; P(k=4,K) = 8K+48 -> K=1: 56, K=2: 64,
    #   K=3: 72. The warm-up cost is K(L-k), so K is its multiplier.
    # k axis (BA_KFS="3" BA_KS="1 2 3 5 6 7 8"): K=3 fixed, k varies, P = 6k+48.
    #
    # Note: k=4, K=3 is an acceptance point, not a data point: it must reproduce the existing
    # d2_ratio_causal_<ds>[_bk]_s<seed> runs exactly (same seed, same K, same allocation
    # algorithm). It is in the default grid on purpose; if it does not match, the code change
    # introduced a behaviour drift and the batch is invalid.
    : "${BA_KS:=4}"
    : "${BA_KFS:=3 1 2}"
    for sd in $SEEDS; do
      for KV in $BA_KS; do
        for KFV in $BA_KFS; do
          BA_K="$KV"; BA_KF="$KFV"
          run best_approximation "$sd" N_SPECIFIC_LAYERS="$KV" SPLIT_FREEZE_TASK="$KFV"
        done
      done
    done
    ;;
  PINNED)
    # Pinned allocation + observation period (exploratory). Same procedure as the k4f3 point of
    # BESTAPPROX (K=3 warm-up; at t=K rho is still computed and logged in layer_scores), except
    # that the 4 committed layers are fixed constants instead of the rho top-4. P=72 (as for
    # best_approximation_k4f3).
    #
    # k=4 and K=3 are fixed in this branch and do not read N_SPECIFIC/FREEZE_T, so that the k4f3
    # in the directory name and the k/K passed to the host come from the same variables.
    #
    # Set names -> layer indices are fixed constants (deep=[8,9,10,11], shallow=[0,1,2,3]).
    : "${PIN_SETS:?STAGE=PINNED needs PIN_SETS (deep and/or shallow)}"
    pin_layers_of() {
      case "$1" in
        deep)    echo "8,9,10,11" ;;
        shallow) echo "0,1,2,3" ;;
        *) echo "unknown PIN_SETS member '$1' (valid: deep/shallow)" >&2; return 1 ;;
      esac
    }
    # Validate the whole list before the first run: a misspelt set name must not fail half-way,
    # after earlier runs have already used GPU time.
    for PS in $PIN_SETS; do pin_layers_of "$PS" > /dev/null || exit 1; done
    PIN_K=4; PIN_KF=3
    for sd in $SEEDS; do
      for PS in $PIN_SETS; do
        PIN_SET="$PS"; PIN_L="$(pin_layers_of "$PS")"
        run pinned "$sd" N_SPECIFIC_LAYERS="$PIN_K" SPLIT_FREEZE_TASK="$PIN_KF" D2_PIN_LAYERS="$PIN_L"
      done
    done
    ;;
  *) echo "unknown STAGE=$STAGE (valid: ENDPOINT/MAIN/GRID/CALIB/RAND2/WINDOW/DEEP/BESTAPPROX/PINNED)"; exit 1 ;;
esac
}

# Batch pre-check for NTASKS=20 (only AugReg x 7 configurations are supported): a dry pass
# collects every job token this invocation would start; if any is not supported, the whole batch
# is refused, instead of failing on an unsupported cell after the supported ones already ran or
# silently writing it into the T=20 root. T=10 skips this block and calls stage_main once.
if [ "$NTASKS" = "20" ]; then
  case "$STAGE" in
    ENDPOINT|GRID|BESTAPPROX) ;;
    *) echo "NTASKS=20 only supports the 7 configurations reachable from STAGE=ENDPOINT/GRID/BESTAPPROX, got STAGE=$STAGE -> refusing to run"; exit 1 ;;
  esac
  if [ "$BK" != "augreg" ]; then
    echo "NTASKS=20 only supports BK=augreg, got BK=$BK -> refusing to run"
    exit 1
  fi
  _PLAN_PASS=1; PLAN_TOKS=()
  stage_main > /dev/null
  _PLAN_PASS=0
  for _tk in ${PLAN_TOKS[@]+"${PLAN_TOKS[@]}"}; do
    case " $T20_REGISTERED " in
      *" $_tk "*) ;;
      *) echo "NTASKS=20: configuration $_tk is not one of the 7 supported configurations ($T20_REGISTERED) -> refusing the whole batch, no run started"
         exit 1 ;;
    esac
  done
fi
stage_main

echo
echo "=== STAGE=$STAGE BK=$BK DATASET=$DATASET finished ==="
echo "Outputs: $BASE_OUT/<job>/metrics.json and run.log (seeds \"${SEEDS// /,}\")"
echo "      (job = d2_<token>_s<seed>, e.g. token all_shared${DS_SUF}${T_SUF}${BK_SUF})"
if [ "$NTASKS" = "20" ]; then
  echo "      T=20: an offline recomputation of the causal allocation (best_approximation) needs a diag npz built on the T=20 split, not the T=10 one"
fi
echo "Each metrics.json holds the per-task accuracy matrix and the d2_* fields of every run."
echo "The allocation used at each task is in the [gate] lines of run.log."
