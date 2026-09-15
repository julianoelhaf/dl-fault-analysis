# Reduced-Observability Fault Localization Summary

## Goal

Assess how much 50 ms fault localization depends on full centralized eight-relay waveform access.

## Scope

Task: FL  
Window: 50 ms  
Models: cnn_lstm_regressor, gru_regressor, inceptiontime_regressor  
Observability: full, full-line-conditioned, terminal-pair, single-ended

## Optimizer / baseline caveat

All cells use a **fixed optimizer (lr = wd = 1e-4, no hyperparameter tuning)** so the observability conditions are directly comparable. The `full` row is therefore an **internal fixed-optimizer reference**, not a replacement for the tuned main-paper 50 ms FL table (which uses LR/WD search and reports slightly lower MAE). `full_line_conditioned` (per-line model, all 8 relays) is the control that isolates the per-line-modelling effect from the relay-count effect; compare it to `terminal_pair`/`single_ended` to attribute degradation to reduced sensing alone.

## Leakage audit

Episode leakage / window grouping / train-only normalization: pass (see split_audit.json, normalization_audit.json)

## Main results (MAE, % of line length)

| observability | relays | cnn_lstm_regressor_mae | gru_regressor_mae | inceptiontime_regressor_mae |
| --- | --- | --- | --- | --- |
| full | 8 | 10.70 ± 2.42 | 7.94 ± 0.52 | 9.89 ± 0.24 |
| full_line_conditioned | 8 | 10.69 ± 0.78 | 9.51 ± 1.59 | 8.91 ± 0.78 |
| terminal_pair | 2 | 10.46 ± 1.32 | 9.14 ± 4.48 | 8.26 ± 0.80 |
| single_ended | 1 | 18.67 ± 0.70 | 18.64 ± 1.33 | 16.93 ± 0.69 |

## Interpretation

Higher MAE under reduced observability indicates greater dependence on centralized sensing for accurate localization.

## Important limitation

Terminal-pair and single-ended settings are **line-conditioned**: they assume the faulted line is known (from a prior FLI stage or line-specific deployment) and use it to select terminal measurements. They are **not** a complete line-agnostic local-relay deployment setting (deployment_interpretation: conditional_on_fault_line_identification).

## Files

run_config.yaml, git_commit.txt, environment.txt, split_audit.json, normalization_audit.json, fold_metrics.csv, aggregate_metrics.csv, table_reduced_observability.{csv,tex}, predictions.csv (if enabled).
