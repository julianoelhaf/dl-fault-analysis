# Provenance: communication/synchronization perturbation (Table 9)

- **Source repository:** private development repository
  `dl-fault-classification-power-grid` (not public).
- **Source commits:** merge `f643342`; underlying commits `065fd29`,
  `0d900ce`, `51c7ddc`; result commit `b465869`.
- **Source paths:** `src/dl_psp/data/perturbations.py`,
  `src/dl_psp/scripts/run_perturbation.py`,
  `config/revision/minimal_communication_perturbation_fl_50ms.yaml`,
  `results/revision/minimal_communication_perturbation_fl_50ms/table_perturbation.{csv,tex}`.
- **Ported to:** `src/dl_fault_analysis/data/perturbations.py`,
  `src/dl_fault_analysis/scripts/run_perturbation.py`,
  `config/paper/communication/minimal_communication_perturbation_fl_50ms.yaml`,
  `results/paper/communication/*` (this directory).
- **Classification:** the published table is `DERIVED_FROM_RAW` -- script
  output, not manual transcription, but the *last* step of a multi-stage
  aggregation rather than direct run output. The generating script
  (`run_perturbation.py`) produces, in order:

  | Stage | File | Content | Classification | Archived here? |
  |---|---|---|---|---|
  | 1 | `predictions.parquet` / `.csv.gz` | per-window predictions | (raw) | **no** -- never committed to either repository |
  | 2 | `perturbation_metrics.csv` | per-fold metrics computed from stage 1 | `DERIVED_FROM_RAW` | yes |
  | 3 | `aggregate_metrics.csv` | mean/std across folds, plus `delta_mae_abs` / `delta_mae_rel` | `DERIVED_FROM_RAW` | yes |
  | 4 | `table_perturbation.{csv,tex}` | `"mean ± std"` formatted strings from stage 3 | `DERIVED_FROM_RAW` | yes |
  | -- | `summary.md` | prose + table rendering of stages 3-4 | `DERIVED_FROM_RAW` | yes |
  | -- | `run_config.yaml`, `dataset_hash.txt` | run / input metadata | run metadata | yes |

  The published table's cells are formatted text (`8.34 ± 0.55`), not numeric
  values, and carry absolute MAE only. **The *relative* degradation figures
  discussed in Section 3.6 (~+0.5% for a 2-sample time shift, ~+118.6% for
  10 ms block loss) live at stage 3 as `delta_mae_rel`**, which is why
  porting `aggregate_metrics.csv` matters: without it those published figures
  had no archived source in this repository.
  **Stages 2-4 are all archived, so the published table and the quoted
  degradations can be checked against the surviving intermediate metrics** --
  and are, by `tests/unit/test_reviewer_analysis_archive.py`, which
  re-derives stage 3 from stage 2 with the repository's own `_aggregate()`
  and stage 4 from stage 3. **They cannot be recomputed from per-window
  predictions, because stage 1 was never committed to either repository and
  does not survive.** Re-running the ported script trains a new model and is
  therefore subject to the same stochastic-training caveats as the main
  benchmark (see docs/REPRODUCIBILITY.md), not expected to reproduce
  bit-identical numbers.
- **Completeness:** all 7 files archived at source commit `b465869` are
  present here -- `table_perturbation.{csv,tex}`, `perturbation_metrics.csv`,
  `aggregate_metrics.csv`, `dataset_hash.txt`, `run_config.yaml` and
  `summary.md` -- each extracted commit-addressed via
  `git show b465869:<path>` (never from the private working tree) and
  verified byte-for-byte, with the one redaction noted below. This experiment
  has no `fold_metrics.csv` or `audits_summary.json`; its per-fold file is
  `perturbation_metrics.csv`.
- **Path redaction:** `dataset_hash.txt` recorded the compute-node scratch
  directory the input manifest was read from
  (`/scratch/<job-id>/windows_tmp/...`). The committed copy replaces that
  directory with `<redacted-internal-scratch-path>`; the manifest SHA-256 and
  the window-tensor filename are unchanged, and no other line was touched.
  The original file (as extracted from commit `b465869`) has SHA-256
  `219a9af1ce64f899a6f90d0bce531ed2dd0a0bc611833454f6122da246a319d1`.
  The manifest hash itself (`1ab9e1a8...`) is identical to the
  observability experiment's, confirming both ran against the same window
  tensor.
- **Execution environment** (from the ported `run_config.yaml`, recorded by
  the run itself): Python 3.12.11, PyTorch 2.8.0+cu128, NumPy 2.3.3,
  scikit-learn 1.7.2, CUDA 12.8; run at private-repo commit `ce594c61` with
  `git_status=dirty`, i.e. the recorded commit is a lower bound on the code
  state. Note this differs from the main benchmark's environment
  (`docs/PROVENANCE.md` §7) -- these reviewer analyses were run later, on
  different machines.
- **Manuscript mapping:** Section 3.6, Table 9.
