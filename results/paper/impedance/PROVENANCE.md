# Provenance: impedance-based fault-localization baseline (Table 10)

- **Source repository:** private development repository
  `dl-fault-classification-power-grid` (not public).
- **Source commits:** `58ee289` (implementation), `48a4283` (unrelated docs
  remap, no changes to these files).
- **Source paths:** `src/dl_psp/models/impedance_locator.py`,
  `src/dl_psp/data/line_parameters.py`,
  `src/dl_psp/scripts/run_impedance_baseline.py`,
  `results/revision/impedance_baseline_fl_50ms/*`.
- **Ported to:** `src/dl_fault_analysis/models/impedance_locator.py`,
  `src/dl_fault_analysis/data/line_parameters.py`,
  `src/dl_fault_analysis/scripts/run_impedance_baseline.py`,
  `results/paper/impedance/*` (this directory).
- **Classification:** mixed; see per file. All are the output of
  `run_impedance_baseline.py` against the full PROTECT-90 50 ms FL window
  set -- not manually transcribed, and not re-derived from another source --
  and all are copied byte-for-byte from the source commit (except the path
  redaction noted below). But the result files are aggregations, not direct
  per-window output:

  | File | Classification | Why |
  |---|---|---|
  | `aggregate_metrics.csv` | `DERIVED_FROM_RAW` | `_agg()` summaries (mae/median/rmse/p90/p99/hit-rates) over 81,030 windows / 9,022 episodes |
  | `per_line.csv` | `DERIVED_FROM_RAW` | same `_agg()`, grouped by line |
  | `sensitivity.csv` | `DERIVED_FROM_RAW` | same `_agg()`, grouped into Rf / distance / faulted-phase buckets |
  | `summary.md` | `DERIVED_FROM_RAW` | prose + table rendering of the three files above |
  | `dataset_hash.txt` | run metadata | SHA-256 of the input window tensor; direct run output, no aggregation |
  | `run_config.yaml` | run metadata | the run's own resolved config (path-redacted, see below) |

  The underlying per-window estimates and errors are written by the script
  as `predictions.parquet`, but that file was **never committed to either
  repository** (it is absent from source commit `58ee289`). The aggregations
  above therefore **cannot be re-checked against their inputs** from this
  repository. Re-running the ported script against the same dataset does
  regenerate both the predictions and these aggregates identically, since
  the method is a deterministic analytic calculation rather than a trained
  model.
- **Completeness:** all 6 files archived at source commit `58ee289` were
  ported here; nothing committed upstream was dropped.
- **Path redaction:** `run_config.yaml` is the one exception to
  byte-for-byte preservation. The original archived file recorded a
  personal HPC cluster path (`/cluster/<username>/Datasets/PROTECT-90/...`)
  in its `base_dir`/`csv_path` fields. The committed copy replaces those two
  path values with `<redacted-personal-cluster-path>`; every other field
  (`stem`, `out_dir`, `f0_hz`, `samples_per_cycle`) is unchanged, including
  the original file's own duplicated `f0_hz` key (a cosmetic artifact of the
  source script's writer loop, not altered here). The original,
  unredacted file (as extracted from commit `58ee289`) has SHA-256
  `668c394fd72df708cf5c2a8587282826e47f71b79517523b6fedc9f83a977972`.
- **Manuscript mapping:** Section 3.7, Table 10.
