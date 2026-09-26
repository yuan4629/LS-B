# Reproduction

All commands run from the repository root. GPU steps assume one CUDA device per command
(`CUDA_VISIBLE_DEVICES=<id>`). The GPU steps below were not re-run for this code release;
the CPU steps (tests, split generation) were.

## 1. Environment and third-party code

See the Installation section of `README.md`:

```bash
pip install -c constraints.txt torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -c constraints.txt -e ".[brain,dev]"
bash scripts/fetch_third_party.sh
pytest
```

## 2. Data and weights

Put the datasets, the fMRI data (`subj01/`) and the three backbones (`pretrained/`) in place
as described in `docs/DATASETS.md`, then generate the ImageNet-R split:

```bash
python make_inr_split.py --inr_root data/imagenet-r --out data/imagenet-r_split
sha256sum data/imagenet-r_split/train.txt data/imagenet-r_split/test.txt   # compare with docs/DATASETS.md
```

## 3. Brain encoder and diagnostic file (GPU, forward passes only after training)

The brain encoder is trained once per backbone on `subj01/` and cached under
`outputs/bioscore_atlas/<fingerprint>/` (`plmodel.ckpt`, `atlas_v2.pt`). Any brain-scored
run trains it if the cache is missing. The `ratio_oracle` arm and `analyze_d2_split.py` also
need a diagnostic file with per-task drive matrices:

```bash
DATASET=inr DUMP=./outputs/atlas_v2_diag_inr_k4_b10.json python baselines/bilora_adapter/atlas_v2_diag.py
python analyze_d2_split.py --npz outputs/atlas_v2_diag_inr_k4_b10.npz      # allocations per backbone
```

`ONLY=augreg|ibot|dino` restricts a script to one backbone.

## 4. Continual-learning runs (GPU)

`run_d2.bash` runs one stage for one backbone (`BK=augreg|ibot|dino`) and dataset
(`DATASET=inr|cub|c100`). Seeds default to `SEEDS="0 1 2"`, results go to
`outputs/exp_d2/<job>/metrics.json` (T=10 tasks) and job names have the form
`d2_<arm>[<arm parameters>]_<inr|cub|c100>[_t20][_ibot|_dino]_s<seed>`.

| STAGE | Arms |
|---|---|
| `ENDPOINT` | `all_shared`, `all_specific`, `random_split` |
| `MAIN` | `ratio_causal` (brain readout after K=3 warm-up tasks), `ratio_oracle` (reads `D2_ALLOC_FILE`), `random_split` |
| `GRID` | `fixed_depth_l` with l in `GRID_LS` (default `2 4 6 8 10`) shallowest blocks specific |
| `WINDOW` | `fixed_window_w1` ... `w7` (contiguous four-block windows; `WINDOWS` selects a subset) |
| `DEEP` | `fixed_depth_deep` with `DEEPN=4` or `10` deepest blocks specific |
| `CALIB`, `RAND2` | extra `random_split` draws with separate allocation seeds |
| `BESTAPPROX` | `best_approximation` over `BA_KS` (number of specific blocks k) x `BA_KFS` (warm-up tasks K) |
| `PINNED` | `pinned` with `PIN_SETS="deep"` (blocks 8-11) and/or `"shallow"` (blocks 0-3) after the same warm-up |

For `MAIN`, always pass `D2_ALLOC_FILE` for the dataset in use: its default is the CIFAR-100
file `./outputs/atlas_v2_diag_k4_b10.npz`.

Examples:

```bash
STAGE=ENDPOINT BK=augreg DATASET=inr bash run_d2.bash
STAGE=MAIN     BK=ibot   DATASET=inr D2_ALLOC_FILE=./outputs/atlas_v2_diag_inr_k4_b10.npz bash run_d2.bash
STAGE=WINDOW   BK=augreg DATASET=inr bash run_d2.bash
STAGE=BESTAPPROX BK=augreg DATASET=inr BA_KS="1 2 3 4 5 6 7 8" BA_KFS="3" bash run_d2.bash
STAGE=PINNED   BK=augreg DATASET=inr PIN_SETS="deep shallow" bash run_d2.bash
```

`NTASKS=20` (ImageNet-R with 10 classes per task, outputs in `outputs/exp_d2_t20/`) is
accepted only for `BK=augreg`, `DATASET=inr` and the stages `ENDPOINT`, `GRID` and
`BESTAPPROX`.

Each `metrics.json` has the keys `config`, `env`, `task_classes` and `results`; `results`
holds the task-by-task accuracy matrix and curves and the per-task allocation
(`selected_layers`, `d2_shared_layers_curve`).
`d2_split.persistent_param_groups` gives the storage count used to compare arms.

## 5. Tests

```bash
pytest -rs          # CPU; prints the reason for every skipped check
```

With the diagnostic file at `outputs/atlas_v2_diag_k4_b10.npz` (CIFAR-100, from step 3 with
`DATASET=c100`), the allocation checks in `tests/test_d2.py` also run.
