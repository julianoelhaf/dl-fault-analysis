# Provenance: reduced-observability sensitivity (Table 8)

- **Source repository:** private development repository
  `dl-fault-classification-power-grid` (not public).
- **Source commits:** merge `f643342` (branch `epsr-revision-reduced-observability`
  into `main`); underlying commits `9a3cf6d`, `97c6512`, `97cf425`, `d83eaa7`,
  `cfba6b5`, `1525b53`; final result commit `2484dd8`.
- **Source paths:** `src/dl_psp/data/observability.py`,
  `src/dl_psp/scripts/run_revision_observability.py`,
  `config/revision/reduced_observability_fl_50ms.yaml`,
  `results/revision/reduced_observability_fl_50ms/table_reduced_observability.{csv,tex}`.
- **Ported to:** `src/dl_fault_analysis/data/observability.py`,
  `src/dl_fault_analysis/data/line_conditioned.py`,
  `src/dl_fault_analysis/scripts/run_revision_observability.py`,
  `config/paper/observability/reduced_observability_fl_50ms.yaml`,
  `results/paper/observability/*` (this directory).
- **Classification:** the published table is `DERIVED_FROM_RAW` -- script
  output, not manual transcription, but the *last* step of a multi-stage
  aggregation rather than direct run output. The generating script
  (`run_revision_observability.py`) produces, in order:

  | Stage | File | Content | Classification | Archived here? |
  |---|---|---|---|---|
  | 1 | `predictions.parquet` / `.csv.gz` | per-window predictions | (raw) | **no** -- never committed to either repository |
  | 2 | `fold_metrics.csv` | per-fold metrics computed from stage 1 | `DERIVED_FROM_RAW` | yes |
  | 3 | `aggregate_metrics.csv` | mean/std across folds, plus `delta_mae_abs` / `delta_mae_rel` | `DERIVED_FROM_RAW` | yes |
  | 4 | `table_reduced_observability.{csv,tex}` | `"mean ± std"` formatted strings from stage 3 | `DERIVED_FROM_RAW` | yes |
  | -- | `summary.md` | prose + table rendering of stages 3-4 | `DERIVED_FROM_RAW` | yes |
  | -- | `run_config.yaml`, `dataset_hash.txt` | run / input metadata | run metadata | yes |
  | -- | `audits_summary.json` | per-model boolean roll-up of the split and normalization audit results | validation metadata (derived; the per-model `split_audit.json` / `normalization_audit.json` it summarizes were gitignored and did not survive) | yes |

  The published table's cells are formatted text (`10.70 ± 2.42`), not
  numeric values. **Stages 2-4 are all archived, so the published table can
  be checked against the surviving intermediate metrics** -- and is, by
  `tests/unit/test_reviewer_analysis_archive.py`, which re-derives stage 3
  from stage 2 with the repository's own `_aggregate()` and stage 4 from
  stage 3. **It cannot be recomputed from per-window predictions, because
  stage 1 was never committed to either repository and does not survive.**
  Re-running the ported script trains new models and is therefore subject to
  the same stochastic-training caveats as the main benchmark (see
  docs/REPRODUCIBILITY.md), not expected to reproduce bit-identical numbers.
- **Completeness:** all 8 files archived at source commit `2484dd8` are
  present here -- `table_reduced_observability.{csv,tex}`,
  `fold_metrics.csv`, `aggregate_metrics.csv`, `audits_summary.json`,
  `dataset_hash.txt`, `run_config.yaml` and `summary.md` -- each extracted
  commit-addressed via `git show 2484dd8:<path>` (never from the private
  working tree) and verified byte-for-byte, with the one redaction noted
  below.
- **Path redaction:** `dataset_hash.txt` recorded the compute-node scratch
  directory the input manifest was read from
  (`/scratch/<job-id>/windows_tmp/...`). The committed copy replaces that
  directory with `<redacted-internal-scratch-path>`; the manifest SHA-256 and
  the window-tensor filename are unchanged, and no other line was touched.
  The original file (as extracted from commit `2484dd8`) has SHA-256
  `0de9a899c10894d3096cc3cd3ac7660e0b80f10058790564f7e6fc70e2eecaf5`.
- **Execution environment** (from the ported `run_config.yaml`, recorded by
  the run itself): Python 3.12.11, PyTorch 2.8.0+cu128, NumPy 2.3.3,
  scikit-learn 1.7.2, CUDA 12.8; run at private-repo commit `03004a76` with
  `git_status=dirty`, i.e. the recorded commit is a lower bound on the code
  state. Note this differs from the main benchmark's environment
  (`docs/PROVENANCE.md` §7) -- these reviewer analyses were run later, on
  different machines.
- **Manuscript mapping:** Section 3.5, Table 8.
