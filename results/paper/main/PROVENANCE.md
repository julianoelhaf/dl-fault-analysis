# Provenance: main benchmark (Tables 4-7)

- **Source:** live Weights & Biases project `julian_oelhaf/dl_comparison_hv_double_line_90kv`,
  exported by the author via the W&B API on 2026-09-15 into `raw/`.
- **Raw files** (`raw/`): `configs.jsonl` (per-run resolved config), `summaries.jsonl`
  (per-run final `cv5_mean/*` / `cv5_std/*` metrics plus `protocol/*`, `env/*`,
  `data/*` metadata), `runs.csv` (run id, name, state, timestamps), `history/`
  (per-run logged scalar history -- for classification runs this includes
  per-fold, per-class precision/recall/f1-score; confusion matrices are
  referenced as W&B table-file artifacts, not included as raw values in this
  export), `export_failures.json` (empty -- the export completed with no
  failures).
- **Classification:** `RAW_EXPERIMENT_OUTPUT` for everything under `raw/`,
  in the precise sense that it is the verbatim export of what the training
  runs themselves logged -- this repository applied no transformation,
  selection, or recomputation to it. Note for exactness: the metric fields
  in `summaries.jsonl` (`cv5_mean/*`, `cv5_std/*`) are fold-aggregates that
  the *training run* computed and logged as its own output, not per-window
  predictions; genuinely per-fold, per-class precision/recall/f1 scalars are
  in `raw/history/*.csv` for classification runs. Per-window predictions
  were never logged to the tracking project and are not archived anywhere.
  `table_y_<target>_{main,appendix}.{csv,tex}` are `DERIVED_FROM_RAW`,
  produced deterministically from `raw/` by
  `src/dl_fault_analysis/evaluation/make_tables_wandb.py` (see below) --
  not manually transcribed.

## Coverage

```
total archived runs:                          265
  finished:                                    253
  crashed:                                     12
  failed / running / other:                    0
unique (window, target, model) combinations:   180  (full grid, zero gaps)
combinations with >1 run (any state):          63
maximum attempts for one combination:          5
selected runs after dedupe_latest:             180
```

Every combination of the full grid (5 window lengths x 4 targets x 9 models
per target) has at least one `finished` run; every `crashed` run has a
later `finished` retry of the same combination, so `dedupe_latest()`
(keep the latest `created_at` per combination -- the exact same
deduplication rule this repository's own table-generation script already
used against the live API before this release) never selects an incomplete
run. This is now directly validated, not just structurally plausible: the
tables regenerated from this archive with this rule reproduce the
*published manuscript's* FC tables cell-for-cell (see
`docs/PROVENANCE.md` §4 and §8, `tests/unit/test_main_results_archive.py`).

## Regenerating the tables

```bash
WANDB_EXPORT_DIR=results/paper/main/raw OUT_DIR=/tmp/regenerated \
  python src/dl_fault_analysis/evaluation/make_tables_wandb.py
```

This reads only the local `raw/` files (no W&B account, no network) and
reproduces `table_y_*_{main,appendix}.{csv,tex}` byte-for-byte from the
archived export.

## Known gaps in this archive

- **Table 3 (parameter counts)** cannot be derived from this export:
  `model/num_params` was never logged to W&B for these runs (`params.num_params`
  is `None`/absent in every row). This is not just a missing-data gap: a
  separate, earlier private-repository experiment (MLflow, ~Oct 2025) did
  log parameter counts, and they match published Table 3 exactly for all 9
  architectures -- but that experiment used a 12-unit FC output head, while
  the runs archived in this directory (which reproduce the published FC
  *results* exactly) use an 11-unit head. See `docs/PROVENANCE.md` §4 for
  the full investigation; this is a confirmed historical discrepancy between
  Table 3 and the rest of the FC material, not something fillable from this
  export.
- **R^2 and P90 (FL appendix)** are not present anywhere in this export --
  confirmed absent from `summaries.jsonl` (`cv5_mean/r2`, `cv5_mean/p90_ae`)
  and, on a second targeted check, from `raw/history/*.csv` for FL runs too
  (those files log only `cv5/fold_metrics`, a table-artifact pointer, no raw
  R^2/P90 scalars). The appendix FL table has empty cells for these two
  columns rather than fabricated values. If these are ever archived from the
  manuscript's printed values rather than raw experiment output, they must
  be classified as `PUBLISHED_TABLE_VALUES`, not `RAW_EXPERIMENT_OUTPUT` --
  not done in this release.
- **Fold-level (not just mean/std) overall metrics** exist on W&B as a table
  artifact per run (`cv5/fold_metrics`, 5 rows x 9 cols) but were not
  downloaded as part of this export (`run_files_inventory.csv`, not
  included here, lists them as existing server-side). Per-fold, per-class
  precision/recall/f1 *are* present as raw scalars in `raw/history/*.csv`
  for classification runs.

## Environment (recovered from these runs' actual config/summary)

Unlike the reviewer-added analyses, this archive lets us report the *actual*
training environment directly from the runs that produced Tables 4-7,
rather than an incidental unrelated snapshot:

- Python 3.12.11, PyTorch 2.5.1+cu121, NumPy 2.3.4, CUDA 12.1 --
  **identical across all 180 selected runs**, checked explicitly by
  grouping on each version field (not assumed). The only per-run variation
  is the OS kernel string across 19 HPC node hostnames (two RHEL patch
  levels, two Ubuntu-based kernels) -- expected for a multi-node cluster,
  irrelevant to the ML stack above.
- `split_seed=42`, `n_splits=5`, `training_seeds=[42]` (`summaries.jsonl`
  `protocol/*`) -- confirms the single-fixed-seed protocol independent of
  the code-level finding already documented in `docs/PROVENANCE.md`.
- `git_commit` recorded per run (private development repository): 3 distinct
  commits among the 180 selected runs (147 / 26 / 7), all small sequential
  commits from February 2026, confirmed to exist directly in that
  repository's history. `git_status=dirty` for **every** selected run --
  the recorded commit is a lower bound on the code state, not a guarantee
  of an exact revision; this is disclosed, not resolved.

This supersedes the earlier, more tentative environment note in
`docs/PROVENANCE.md` §7, which only had an unrelated experiment's
incidental snapshot to go on.

## Manuscript mapping

Table 4 = `y_fault_present`, Table 5 = `y_fault_class`, Table 6 =
`y_fault_line`, Table 7 = `y_fault_location`. Main-text tables report
F1 + PR-AUC (classification) or MAE + RMSE (regression); appendix tables
add Accuracy + ROC-AUC (classification) or R^2 + P90 (regression, largely
empty per the gap noted above).
