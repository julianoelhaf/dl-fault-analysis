# Frozen configuration: main FD/FC/FLI/FL benchmark (Tables 4-7)

Unlike the three reviewer-added analyses, the main benchmark was never given
its own per-task/per-window frozen config file -- it is controlled entirely
by Hydra CLI overrides from `hpc/run_deep_learning_models.sh`, layered on the
shared defaults below. There is nothing to relocate here; this file records
exactly which pieces make up "the main benchmark configuration" so a reader
does not have to reconstruct that from the shell script alone.

**Shared defaults** (`config/{dataset,model,tracking,training,window_extraction}/default.yaml`):
`n_splits=5`, `split_seed=42`, `seeds=[42]` (single seed; see
docs/PROVENANCE.md), `batch_size=256`, `learning_rate=1e-4`,
`weight_decay=1e-4`, `epochs=500`, `patience=15`,
`tune_lr_wd=true` with `lr_grid`/`wd_grid = [0.01, 0.001, 0.0001]` on a single
calibration fold (`calib_fold=0`).

**Grid dimensions**, iterated by `hpc/run_deep_learning_models.sh`
(a template -- see the script's own header comment):
- 5 window lengths: `window_extraction.window_length` in
  {0.010, 0.020, 0.030, 0.040, 0.050} seconds.
- 4 targets: `training.target_label` in {`y_fault_present` (FD),
  `y_fault_class` (FC), `y_fault_line` (FLI), `y_fault_location` (FL)}.
- 9 models per target: `model.model_name` in
  {cnn, cnn_lstm, dilated_cnn, gru, inceptiontime, lstm, rnn, tcn, tft}
  {classifier,regressor} (classifier for FD/FC/FLI, regressor for FL).

**CV split:** episode-grouped 5-fold (`StratifiedGroupKFold` for FC/FLI,
`GroupKFold` for FD/FL); see `config/paper/splits/SPLIT_LIMITATION.md` for
what is and is not recoverable about the exact published split.

**Result archive:** see `results/paper/main/README.md` -- the archived
Weights & Biases export behind Tables 4-7, from which those tables are
regenerable offline.
