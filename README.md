# LS-B

Code for experiments on where to place task-specific adapters in a frozen ViT-B/16 during
class-incremental learning. The learner is BiLoRA. Each of the 12 blocks gets either a
task-specific adapter or one adapter shared by all tasks. The block allocation comes either
from fixed rules or from a brain readout. For the brain readout, the first tasks are observed
through a frozen fMRI encoding model of human visual cortex, and the allocation is computed
once from those observations and then frozen.

## Repository layout

| Path | Contents |
|---|---|
| `run_d2.bash` | experiment driver: `STAGE` selects a set of allocation arms, `BK` the backbone, `DATASET` the benchmark |
| `exp_timm.bash` | environment check and command-line assembly for `exp.py` |
| `exp.py` | runs one job (method x seed), handles resume and writes `metrics.json` |
| `core/` | CLI, datasets and task splits, metrics, training utilities |
| `baselines/bilora_adapter/` | BiLoRA adapter, allocation host (`bilora_d2.py`), allocation arms (`d2_split.py`), dataset wiring, brain-readout scripts |
| `model_m/common/` | brain-alignment scoring on top of the brain encoder, frozen timm ViT wrapper |
| `analyze_d2_split.py` | allocation from per-layer statistics and summary readouts |
| `make_inr_split.py` | generates the ImageNet-R train/test split |
| `scripts/` | `fetch_third_party.sh` and notes on the scripts |
| `third_party/` | patch for the brain-encoder code and its license note |
| `tests/` | CPU tests (no data needed; checks that need data are skipped) |
| `docs/` | `ARCHITECTURE.md`, `DATASETS.md`, `REPRODUCTION.md` |

## Installation

Python 3.12 or 3.13. The versions used for the paper are pinned in `constraints.txt`
(Python 3.13.13, torch 2.12.0 with CUDA 13.0).

```bash
python -m venv .venv && source .venv/bin/activate
# CUDA build (as used for the paper); use https://download.pytorch.org/whl/cpu for CPU only
pip install -c constraints.txt torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -c constraints.txt -e ".[brain,dev]"
bash scripts/fetch_third_party.sh
```

The package has three dependency groups: the core dependencies (continual-learning runs and
tests), `brain` (brain encoder and brain-derived allocations) and `dev` (pytest, ruff).
`scripts/fetch_third_party.sh` clones BiLoRA and the brain-encoder code at pinned commits;
neither is redistributed here (see `docs/DATASETS.md` for sources and licenses).

## Quick check (CPU, no data)

```bash
pytest
```

All tests either pass or are skipped, and each skip names the missing input.

## Reproduction

Datasets, fMRI data and pretrained weights are not included. `docs/DATASETS.md` lists
sources, terms and checksums. `docs/REPRODUCTION.md` gives the full command sequence. The
main runs have the form

```bash
STAGE=MAIN BK=augreg DATASET=inr D2_ALLOC_FILE=./outputs/atlas_v2_diag_inr_k4_b10.npz bash run_d2.bash
```

Results are written as `outputs/exp_d2/<job>/metrics.json`.

`exp_timm.bash` stops if the environment differs from the one used for the paper
(Python 3.13.13, torch 2.12.0+cu130, numpy 2.4.6, timm 1.0.27, scikit-learn 1.9.0).
Set `ALLOW_ENV_DRIFT=1` to run anyway; a warning is printed.

## License

MIT (see `LICENSE`). The code that `scripts/fetch_third_party.sh` downloads
(`baselines/BiLoRA/` and `brainnet/`) is not part of this repository and keeps its own license.
`third_party/brainnet_plmodel.patch` modifies CC BY-NC licensed code and is distributed under
the terms of that code, not under the MIT license; see `third_party/README.md`.
