# Deep Learning Architectures for Fault Analysis in Power System Protection

[![CI](https://github.com/julianoelhaf/dl-fault-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/julianoelhaf/dl-fault-analysis/actions/workflows/ci.yml)
[![Paper DOI](https://img.shields.io/badge/paper-10.1016%2Fj.epsr.2026.114169-blue)](https://doi.org/10.1016/j.epsr.2026.114169)
[![Dataset DOI](https://img.shields.io/badge/dataset-10.5281%2Fzenodo.21109169-blue)](https://doi.org/10.5281/zenodo.21109169)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

Code accompanying:

**Deep Learning Architectures for Fault Analysis in Power System Protection: A Reproducible Evaluation on Public Waveform Data**
Julian Oelhaf, Georg Kordowich, Christian Bergler, Andreas Maier, Johann Jäger, Siming Bayer
*Electric Power Systems Research*, Vol. 265 (2027), Art. 114169.
DOI: [10.1016/j.epsr.2026.114169](https://doi.org/10.1016/j.epsr.2026.114169)

Dataset: **PROTECT-90 v1.0.0**, DOI: [10.5281/zenodo.21109169](https://doi.org/10.5281/zenodo.21109169)

This repository contains the deep-learning benchmark code (four protection
tasks, nine architectures) plus three reviewer-added evaluations
(reduced-observability sensitivity, communication-perturbation robustness,
and a classical impedance-based localization baseline). It installs and runs
without any FAU-internal or otherwise private dependency. See
[Reproducibility notes](#reproducibility-notes) for exactly what is and is
not reproducible from this repository alone.

---

## What this repository contains

Four protection tasks, evaluated under identical preprocessing, windowing,
and episode-grouped 5-fold cross-validation:

- **Fault Detection (FD)** -- binary, fault / no fault
- **Fault Classification (FC)** -- 11-class, fault type
- **Fault Line Identification (FLI)** -- 4-class, which of the 4 line circuits
- **Fault Localization (FL)** -- regression, position along the faulted line

Nine architectures per applicable task (CNN, Dilated CNN, RNN, LSTM, GRU,
CNN-LSTM, TCN, InceptionTime, TFT), across five window lengths (10-50 ms).

Plus three EPSR reviewer-response analyses (Sections 3.5-3.7): reduced
sensing (Table 8), input perturbation robustness (Table 9), and a classical
physics-based fault-localization baseline (Table 10) -- see
[Paper result mapping](#paper-result-mapping).

## Quick start

```bash
git clone https://github.com/julianoelhaf/dl-fault-analysis
cd dl-fault-analysis
pip install -e .

python src/dl_fault_analysis/scripts/run_dl_experiment.py \
  dataset=hv_double_line_90kv \
  dataset.dataset_directory=/path/to/PROTECT-90/windows \
  model.model_name=gru_classifier \
  window_extraction.window_length=0.05 \
  training.target_label=y_fault_present
```

This runs one representative experiment (Fault Detection, GRU, 50 ms) through
the full 5-fold CV protocol and writes checkpoints, fold predictions, and
aggregate metrics under `outputs/run_<id>/`. No private dependency, no W&B
account, and no GPU are required to install or run this (GPU recommended for
realistic training time).

## Data

Download PROTECT-90 v1.0.0 from
[Zenodo](https://doi.org/10.5281/zenodo.21109169) and point
`dataset.dataset_directory` (CLI override, as above, or edit
`config/dataset/hv_double_line_90kv.yaml`) at the directory containing the
windowed `.raw` / `.raw.meta.json` / `.parquet` triplets for the window
length(s) you want to run. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the exact expected
file layout and known sample/window sanity counts (Table 2).

## Reproduce the paper

`hpc/run_deep_learning_models.sh` documents the full parameter grid (5 window
lengths x 4 targets x up to 9 models). It is a SLURM **template**: the
original per-job wall-clock/array structure could not be recovered from
project history, so size a job array to your own cluster rather than running
it as a single job. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the reviewer-added
analyses' runner commands and for reproducibility caveats (the CV split, GPU
non-determinism, and the Table 11 runtime benchmark).

## Paper result mapping

| Paper item | Command / config | Archived result |
| --- | --- | --- |
| Table 2 (sample/window counts) | -- | sanity counts in `docs/REPRODUCIBILITY.md` |
| Table 3 (parameter counts) | `create_model_from_name` (`models/model_utils.py`) | see `docs/PROVENANCE.md` §4 -- the published FC counts reflect an earlier 12-output configuration |
| Table 4 (FD) | `scripts/run_dl_experiment.py training.target_label=y_fault_present` | `results/paper/main/table_y_fault_present_{main,appendix}.csv` |
| Table 5 (FC) | `... training.target_label=y_fault_class` | `results/paper/main/table_y_fault_class_{main,appendix}.csv` |
| Table 6 (FLI) | `... training.target_label=y_fault_line` | `results/paper/main/table_y_fault_line_{main,appendix}.csv` |
| Table 7 (FL) | `... training.target_label=y_fault_location` (`*_regressor` model) | `results/paper/main/table_y_fault_location_{main,appendix}.csv` |
| Table 8 (reduced observability) | `scripts/run_revision_observability.py` + `config/paper/observability/` | `results/paper/observability/` |
| Table 9 (communication perturbation) | `scripts/run_perturbation.py` + `config/paper/communication/` | `results/paper/communication/` |
| Table 10 (impedance baseline) | `scripts/run_impedance_baseline.py` | `results/paper/impedance/` |
| Table 11 (runtime) | not re-executable as published; see `config/paper/runtime/README.md` | archived in the manuscript only |
| Figure 4 (window length vs. performance) | `visualization/plot_window_length_vs_performance_grid.py` (pulls from the live W&B API; not adapted for the offline export) | underlying data archived in `results/paper/main/`, the plot itself is not regenerable offline |

## Published result artifacts

`results/paper/{main,impedance,observability,communication}/` contain the
actual archived output behind Tables 4-10 -- inspectable directly, no W&B
account needed. Each has its own `PROVENANCE.md` recording exact source
(commit, or the W&B export date/coverage for Tables 4-7) and whether values
are raw experiment output or derived. Two known gaps are documented rather
than papered over: parameter counts were never logged by the campaign behind
Tables 4-7, and the FL appendix's R^2/P90 columns are absent from the
archived W&B summaries.

One provenance note on Table 3, stated plainly: the fault-classification
task has 11 classes and the models behind the published FC results used an
11-output head, but **Table 3's published parameter counts correspond to an
earlier 12-output FC configuration**. This is a historical inconsistency in
the parameter counts only; it does not affect the published FC performance
results, which are reproduced exactly from `results/paper/main/raw/`. Full
evidence: [`docs/PROVENANCE.md`](docs/PROVENANCE.md) §4.

## Reproducibility notes

This repository provides the code and frozen configurations for every
experiment family in the paper, and archived result files for the three
reviewer-added analyses. It does **not** claim that every number is
reproducible byte-for-byte:

- The exact published cross-validation fold assignment is not recoverable
  (only the algorithm and seed are frozen) -- see
  `config/paper/splits/SPLIT_LIMITATION.md`.
- Trained-model results (the main benchmark, observability, perturbation)
  are subject to ordinary GPU/PyTorch non-determinism; re-running them
  should be expected to differ from the archived numbers by a small margin.
- Table 11 is archived as a **publication result, not a reproducible
  benchmark**: the recovered timing code and run metadata show a timed batch
  up to 256 samples (not single-sample) and multiple GPU models across the
  source runs (V100, RTX 2080 Ti, RTX 3080) rather than the manuscript's
  stated fixed-RTX-3080 single-sample protocol. See
  `docs/PROVENANCE.md` §6 for the precise finding.

Full detail: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) (how to
run things) and [`docs/PROVENANCE.md`](docs/PROVENANCE.md) (where everything
came from, including the FC 11-vs-12-class question and the private-dependency
removal).

## Computational environment

Python >= 3.10; PyTorch, scikit-learn, Hydra, Weights & Biases (optional,
disabled by default), matplotlib/seaborn. `pip install -e .` installs
everything from PyPI -- no private package index. The main benchmark's
actual training environment (Python 3.12.11, PyTorch 2.5.1+cu121, NumPy
2.3.4, CUDA 12.1) is recorded per-run in `results/paper/main/raw/`; see
`docs/PROVENANCE.md` §7 for how that compares to the (RTX 3080-equipped)
hardware `hpc/run_deep_learning_models.sh` targets.

## Repository structure

```text
config/         Hydra configs -- general defaults, plus config/paper/ (frozen
                per-experiment-family configs and reproducibility notes)
src/            dl_fault_analysis package: data pipeline, models, training
                scripts, the three reviewer-added analyses, evaluation
results/paper/  Archived publication result artifacts
docs/           REPRODUCIBILITY.md, PROVENANCE.md
tests/          Unit + integration tests (CPU-only)
hpc/            SLURM template for the full experiment grid
```

## Citation

```bibtex
@article{oelhaf2027dl,
  title={Deep Learning Architectures for Fault Analysis in Power System Protection: A Reproducible Evaluation on Public Waveform Data},
  author={Oelhaf, Julian and Kordowich, Georg and Bergler, Christian and Maier, Andreas and J\"{a}ger, Johann and Bayer, Siming},
  journal={Electric Power Systems Research},
  volume={265},
  pages={114169},
  year={2027},
  doi={10.1016/j.epsr.2026.114169}
}
```

Software citation metadata: [`CITATION.cff`](CITATION.cff). Please also cite
the dataset (PROTECT-90 v1.0.0, DOI above) if you use it.

## License

[MIT](LICENSE)
