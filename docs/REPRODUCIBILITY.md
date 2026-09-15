# Reproducibility notes

This document covers environment setup, the full experiment grid, and known
deterministic/stochastic behavior. For provenance of specific numbers and
files, see [`docs/PROVENANCE.md`](PROVENANCE.md).

## Environment

- **Python:** >= 3.10 declared in `pyproject.toml`; the original study ran
  on Python 3.12 (confirmed from `hpc/run_deep_learning_models.sh`'s
  `module load python/3.12-conda`).
- **GPU:** the original study ran on an NVIDIA RTX 3080
  (`hpc/run_deep_learning_models.sh`'s `--gres=gpu:rtx3080:1`); training and
  inference also run on CPU (slower) or other CUDA GPUs.
- **PyTorch / CUDA build:** not pinned anywhere for the actual study runs
  (see `docs/PROVENANCE.md` §7). `pyproject.toml` declares `torch>=2.1`,
  which is a compatible range, not a claim about the exact paper build.
- **Dependencies:** `pip install -e .` installs everything needed from
  PyPI -- no private package index, no `psp_helper`, no FAU-internal
  infrastructure (see `docs/PROVENANCE.md` §3).

## Installing

```bash
git clone https://github.com/julianoelhaf/dl-fault-analysis
cd dl-fault-analysis
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -e .
```

## Dataset

Download PROTECT-90 v1.0.0 from Zenodo
(`10.5281/zenodo.21109169`) and set
`dataset_directory` in `config/dataset/hv_double_line_90kv.yaml` (or pass it
as a CLI override, see below) to the directory containing the windowed
`.raw` / `.raw.meta.json` / `.parquet` triplets produced for each window
length (see `src/dl_fault_analysis/data/window_io.py` for the exact expected
layout and filenames).

Known sanity values for the windowed dataset (from the manuscript):

| Window length | Total windows | Fault windows |
| ------------: | ------------: | ------------: |
|         10 ms |       279,682 |         9,022 |
|         20 ms |       261,638 |        27,066 |
|         30 ms |       243,594 |        45,110 |
|         40 ms |       225,550 |        63,154 |
|         50 ms |       207,506 |        81,198 |

9,022 episodes total, 6.4 kHz sampling. There is no bundled validation
command for these counts in this repository (it would require the multi-GB
dataset itself to test); use them as a manual sanity check against
`y_<topology>_W<...>.parquet`'s row count and `y_fault_present.sum()`.

## Reproducing one representative experiment

```bash
python src/dl_fault_analysis/scripts/run_dl_experiment.py \
  dataset=hv_double_line_90kv \
  dataset.dataset_directory=/path/to/PROTECT-90/windows \
  model.model_name=gru_classifier \
  window_extraction.window_length=0.05 \
  training.target_label=y_fault_present
```

This runs the full episode-grouped 5-fold CV protocol for one
model/task/window-length combination (Fault Detection, GRU, 50 ms). Output
(checkpoints, fold predictions, aggregate metrics) is written under
`outputs/run_<id>/`. Swap `training.target_label` for `y_fault_class` (FC),
`y_fault_line` (FLI), or `y_fault_location` (FL, use a `*_regressor` model
instead), and `window_extraction.window_length` for any of
{0.010, 0.020, 0.030, 0.040, 0.050}.

## Reproducing the paper's full grid

`hpc/run_deep_learning_models.sh` demonstrates the full parameter grid (5
window lengths x 4 targets x up to 9 models per target = up to 180 runs) as
a SLURM template. Its original per-job wall-clock budget and job-array
structure could not be recovered from project history (see the script's own
header comment) -- size a job array to your cluster's actual per-run
walltime rather than running the script as one job.

## Reproducing the reviewer-added analyses

Each has its own runner and frozen config:

```bash
# Reduced-observability sensitivity (Table 8)
python src/dl_fault_analysis/scripts/run_revision_observability.py \
  --revision-config config/paper/observability/reduced_observability_fl_50ms.yaml \
  --windows-local-dir /path/to/PROTECT-90/windows

# Communication/synchronization perturbation (Table 9)
python src/dl_fault_analysis/scripts/run_perturbation.py \
  --config config/paper/communication/minimal_communication_perturbation_fl_50ms.yaml \
  --windows-local-dir /path/to/PROTECT-90/windows

# Impedance-based localization baseline (Table 10) -- analytic, no training
python src/dl_fault_analysis/scripts/run_impedance_baseline.py \
  --base-dir /path/to/PROTECT-90/windows \
  --csv /path/to/PROTECT-90/hv_double_line_90kv_labels.csv
```

The impedance baseline is deterministic (no training, no folds); re-running
it reproduces the archived `results/paper/impedance/` values exactly. The
observability and perturbation runners train models, so re-running them is
subject to the stochastic-training caveats below.

## Deterministic vs. stochastic behavior

- **Deterministic:** the CV split algorithm for a fixed seed and scikit-learn
  version (see `config/paper/splits/SPLIT_LIMITATION.md`); the impedance
  baseline (a closed-form calculation).
- **Stochastic (not expected to reproduce bit-identical numbers):** any
  trained-model result (main benchmark, observability, perturbation).
  `training.seeds` is a single fixed seed (`[42]`) per
  `config/training/default.yaml`, but GPU non-determinism (cuDNN algorithm
  selection, atomic-add ordering in backward passes), PyTorch version
  differences, and hardware differences all mean a fresh training run should
  be expected to differ from the archived numbers by a small margin, not
  match them exactly. This is a property of GPU-trained deep learning
  pipelines in general, not specific to this repository.
- **`config.training.seeds` lists only `[42]`:** the training loop
  (`run_dl_experiment.py`) only ever consumes `seeds[0]`; a list with
  multiple entries was cleaned up in this release since the paper reports a
  single fixed seed and the code never used more than one (see
  `docs/PROVENANCE.md`).

## Runtime benchmark (Table 11)

Archived as published; not independently reproducible on different hardware
by design (a benchmark on a different GPU is a new measurement, not a
reproduction). See `config/paper/runtime/README.md` for the recovered
protocol and two discrepancies against the manuscript's stated methodology.

## Verifying without retraining

Every table's raw data is archived locally and inspectable without W&B, GPU,
or the dataset:

- `results/paper/{impedance,observability,communication}/` -- the actual
  output of the runs behind Tables 8-10, copied byte-for-byte from the
  source commits.
- `results/paper/main/raw/` -- the author's Weights & Biases export behind
  Tables 4-7 (`configs.jsonl`, `summaries.jsonl`, `runs.csv`, `history/`).
  Regenerate the derived tables with:

  ```bash
  WANDB_EXPORT_DIR=results/paper/main/raw OUT_DIR=/tmp/regenerated \
    python src/dl_fault_analysis/evaluation/make_tables_wandb.py
  ```

  See `results/paper/main/PROVENANCE.md` for coverage (all 180
  window/target/model combinations, zero gaps), the retry-deduplication
  rule applied, and two known gaps (Table 3's parameter counts and the FL
  appendix's R^2/P90 columns are not present in this export).
