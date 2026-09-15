# Main benchmark results (Tables 4-7)

Archived: exported from the live Weights & Biases project by the author and
committed here. See [`PROVENANCE.md`](PROVENANCE.md) for exactly what is
raw vs. derived, the deduplication rule applied, coverage, and known gaps
(Table 3's parameter counts and the FL appendix's R^2/P90 columns are not
recoverable from this export).

- `raw/` -- the archived W&B export (`configs.jsonl`, `summaries.jsonl`,
  `runs.csv`, `history/`), one entry per training run.
- `table_y_<target>_main.{csv,tex}` / `table_y_<target>_appendix.{csv,tex}`
  -- derived tables (mean +/- std over 5 folds), regenerable from `raw/`
  with `src/dl_fault_analysis/evaluation/make_tables_wandb.py`
  (`WANDB_EXPORT_DIR=results/paper/main/raw`, no W&B account needed).

`y_fault_present` = Table 4 (FD), `y_fault_class` = Table 5 (FC),
`y_fault_line` = Table 6 (FLI), `y_fault_location` = Table 7 (FL).
