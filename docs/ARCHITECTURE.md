# Architecture

## Call chain of a continual-learning run

```
run_d2.bash                 one STAGE = a list of (arm, seed) jobs for one backbone and dataset
  -> exp_timm.bash          environment check, assembles the exp.py command line
    -> exp.py               run identity / resume, logging, writes <job>/metrics.json
      -> core/registry.py   method name -> module ("bilora", "bilora_d2")
        -> baselines/bilora_adapter/bilora_d2.py
             allocation host: decides which ViT blocks get task-specific adapters and which
             share one persistent adapter, then trains through
        -> baselines/bilora_adapter/bilora.py (run_bilora)
             adapts the upstream BiLoRA learner (baselines/BiLoRA, fetched separately)
             to timm 1.x via timm_compat.py and to the datasets in core/data.py,
             inr_data.py (ImageNet-R) and cub_data.py (CUB-200)
```

## Allocation arms

`--bioscore_split_mode` selects how the 12 blocks of the ViT-B/16 are split into
task-specific and shared adapters. The vocabulary and the static allocations live in
`baselines/bilora_adapter/d2_split.py`:

- fixed rules: `all_shared`, `all_specific`, `random_split`, `fixed_depth` / `fixed_depth_l`
  (shallowest blocks), `fixed_depth_deep` (deepest blocks), `fixed_window_w1..w7`
  (contiguous four-block windows);
- brain-derived: `ratio_causal` (the first K tasks are trained with task-specific adapters in
  every block; after task K the allocation is computed once from those tasks' brain-readout
  statistics and frozen), `best_approximation` (the same procedure for other numbers of
  specific blocks k), `pinned` (the same warm-up, then a fixed, pre-specified block set),
  `ratio_oracle` (allocation computed from a precomputed diagnostic file).

`analyze_d2_split.py` turns per-layer statistics into an allocation (`allocate`,
`allocate_from_scores`) and prints summary readouts for a diagnostic file.
`d2_split.persistent_param_groups` counts the persistent parameter groups of an arm, which is
the storage measure used to compare arms.

## Brain readout

- `model_m/common/selectors.py` (`BioScoreCalculator`): trains the third-party brain encoder
  (`brainnet` PLModel, fetched separately) on the fMRI data once per backbone and caches it under
  `outputs/bioscore_atlas/<fingerprint>/`.
- `model_m/common/bioscore_v2.py`: extracts the per-voxel encoder parameters (`atlas_v2.pt`) and
  computes the ROI x layer drive matrix of a task from forward passes only.
- `baselines/bilora_adapter/bioscore_gate.py`: the frozen scoring backbone (same weights as the
  learner's backbone) and a loader adapter, used by `bilora_d2.py` at the allocation step.
- `baselines/bilora_adapter/atlas_v2_export.py` and `atlas_v2_diag.py`: offline scripts that
  export the atlas, run structural checks, and write the diagnostic `.npz` file with per-task
  drive matrices (input of `ratio_oracle` and `analyze_d2_split.py`).
- `model_m/common/timm_backbone.py` and `lora.py`: frozen timm ViT wrapper (AugReg `.npz`,
  iBOT / DINO `.pth` weights) used for scoring.

## Shared modules

| Module | Role |
|---|---|
| `core/cli.py` | argument parser and dataset presets (some options are inherited and unused by this pipeline) |
| `core/data.py` | datasets and class-incremental task splits |
| `core/metrics.py` | accuracy matrix, average accuracy, backward transfer |
| `core/train.py` | parameter groups and small training utilities |
| `core/plotting.py` | accuracy curves |

## Tests

`tests/test_d2.py` and `tests/test_bioscore_v2.py` run on CPU without data. Checks that need
something not shipped (diagnostic `.npz`, run outputs, datasets, the upstream BiLoRA checkout)
are skipped with the reason printed.
