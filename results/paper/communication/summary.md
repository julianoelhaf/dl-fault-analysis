# Minimal Communication-Perturbation Summary (FL, 50 ms)

## Goal

Probe how a trained 50 ms fault-localization model (GRU, full 8-relay observability) degrades under small **test-time** input perturbations — **no retraining**.

## Scope

Task: FL  
Window: 50 ms  
Model: gru_regressor  
Observability: full  
Perturbations: none, time_shift_2_samples, block_loss_10ms (test inputs only)

## Leakage audit

Episode disjointness / window grouping / train-only normalization: pass (see split_audit.json, normalization_audit.json)

## Main results (MAE, % of line length)

| perturbation | gru_regressor_mae |
| --- | --- |
| none | 8.34 ± 0.55 |
| time_shift_2_samples | 8.39 ± 0.55 |
| block_loss_10ms | 18.24 ± 0.93 |

## Interpretation

Compare each perturbation's MAE to `none` (the unperturbed full-observability baseline): a 2-sample time shift probes synchronisation sensitivity; a 10 ms block loss probes a brief communication dropout.

## Important limitation

This is a **compact, test-time-only** sensitivity probe (no retraining, full observability). It is **not** the broad DPSP robustness grid and covers only these three perturbations by design.

## Files

perturbation_metrics.csv, aggregate_metrics.csv, table_perturbation.{csv,tex}, split_audit.json, normalization_audit.json, run_config.yaml, dataset_hash.txt, predictions.parquet (if enabled).
