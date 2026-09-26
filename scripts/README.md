# Scripts

| Script | Purpose |
|---|---|
| `scripts/fetch_third_party.sh` | clones BiLoRA and the brain-encoder code (Brain Decodes Deep Nets) at pinned commits into `baselines/BiLoRA/` and `brainnet/`, and applies `third_party/brainnet_plmodel.patch` |
| `run_d2.bash` (repository root) | experiment driver; `STAGE` selects the allocation arms (see the header of the script and `docs/REPRODUCTION.md`) |
| `exp_timm.bash` (repository root) | environment check and command-line assembly for `exp.py`; called by `run_d2.bash` |
| `make_inr_split.py` (repository root) | generates the ImageNet-R train/test split (`docs/DATASETS.md`) |
| `analyze_d2_split.py` (repository root) | allocation and summary readouts from a diagnostic `.npz` file |
| `baselines/bilora_adapter/atlas_v2_export.py` | exports the brain-encoder atlas and runs structural checks (GPU, forward passes only) |
| `baselines/bilora_adapter/atlas_v2_diag.py` | writes the diagnostic `.npz` file with per-task drive matrices (GPU, forward passes only) |

The entry scripts stay at the repository root because they import each other by relative
path and are run from the root.
