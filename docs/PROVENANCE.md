# Provenance

This document records where every non-obvious piece of this release came
from, and is honest about what could and could not be recovered or verified.
It complements [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md), which
focuses on *how* to run things.

## 1. Repository relationship

This repository (`dl-fault-analysis`) is the canonical public/citation
repository for the paper. It was exported as a single clean commit from a
private, multi-author development repository (`dl-fault-classification-power-grid`,
not public) that covers the same core study plus later work answering EPSR
reviewer comments. That reviewer-response work (Sections 3.5-3.7 of the
manuscript) was never ported to this repository until this release. Only the
three reviewer-added analyses and a small amount of shared supporting
infrastructure they need were curated out of that private repository's
history; its full commit history, unrelated development churn, and dirty
working-tree state were **not** merged or copied.

## 2. Reviewer-added analyses (Sections 3.5-3.7)

Each has its own `PROVENANCE.md` under `results/paper/<name>/` with exact
source commits and file-level mapping:

- [`results/paper/impedance/PROVENANCE.md`](../results/paper/impedance/PROVENANCE.md) -- Section 3.7, Table 10
- [`results/paper/observability/PROVENANCE.md`](../results/paper/observability/PROVENANCE.md) -- Section 3.5, Table 8
- [`results/paper/communication/PROVENANCE.md`](../results/paper/communication/PROVENANCE.md) -- Section 3.6, Table 9

**Classification note:** the archived result files for all three are
`DERIVED_FROM_RAW`, not raw run output. Each generating script emits a
layered set of artifacts (per-window predictions → per-fold metrics →
fold-aggregated metrics → the publication-facing table), and what is
archived here is at or near the end of that chain. `run_config.yaml` and
`dataset_hash.txt` are run/input metadata; `audits_summary.json` is
validation metadata. Each family's `PROVENANCE.md` gives the exact
stage-by-stage breakdown and per-file classification.

Every file that the three source commits archive is present here: the
publication-facing `table_*.{csv,tex}` and the surviving intermediate
metrics alike (`fold_metrics.csv`, `perturbation_metrics.csv`,
`aggregate_metrics.csv`, `audits_summary.json`, `run_config.yaml`,
`dataset_hash.txt`, `summary.md`), each extracted commit-addressed from
`2484dd8` / `b465869` / `58ee289` and verified byte-for-byte against the
source commit.

**The practical consequence:** the published Tables 8, 9 and 10 **can be
checked against the surviving intermediate metrics** -- `tests/unit/test_reviewer_analysis_archive.py`
re-derives the fold-aggregated metrics from the per-fold metrics using this
repository's own aggregation code, and the published table cells from those
aggregates, for Tables 8 and 9. They **cannot be recomputed from per-window
predictions**: those were written by the scripts at run time but were never
committed to either repository, for any of the three analyses, and do not
survive. That limit is real and is not closed by this archive.

All three archived result files were copied byte-for-byte from their source
commits and were not regenerated or edited, with one exception: the
impedance baseline's archived `run_config.yaml` had its personal HPC cluster
path redacted in the committed copy (checksum of the original preserved in
`results/paper/impedance/PROVENANCE.md`) -- no result value was touched.
Integration work was otherwise limited to: adapting the `dl_psp` namespace
to `dl_fault_analysis`; removing the same hardcoded private-cluster path
from the impedance baseline *script's* defaults in favor of CLI args /
environment variables; and porting a small amount of shared infrastructure
the ported runner scripts needed but that the original port instructions did
not enumerate (a `validation` audit package, `data/line_conditioned.py`, and
`utils/ckpt.py` -- none of these existed in the public repository before,
and none duplicate core FD/FC/FLI/FL functionality already present here).

## 3. `psp_helper` dependency removal

The repository as originally exported imported a private, FAU-internal
package (`psp_helper`, living in two identical-HEAD GitLab repositories,
neither public) in 9 files, without ever declaring it in `pyproject.toml` --
`pip install -e .` could not produce a runnable checkout.

Rather than publish `psp_helper` itself (a second codebase, with its own
release/hygiene burden, most of which is unrelated to this paper), the
minimal actually-used subset was vendored directly into `dl_fault_analysis`:

| Vendored into | Replaces |
|---|---|
| `utils/logging.get_logger` | `psp_helper.utils.logging.get_logger` (unchanged) |
| `config.MainConfig` | `psp_helper.config.MainConfig` (trimmed to fields this repo's config tree uses; see the module docstring for why this doesn't change runtime behavior) |
| `constants.{FAULT_LABEL_TO_ID,FAULT_ID_TO_LABEL,FAULT_TARGET_TO_OUTPUT_DIM}` | `psp_helper.constants` (trimmed -- see Section 4 below) |
| `data/window_io.{generate_paths,open_raw_memmap,load_feature_metadata,GlobalWindowLoader}` | `psp_helper.windows_helper` / `psp_helper.windows.global_loader` / `psp_helper.io.feature_meta` (trimmed to reading an already-windowed dataset; the original's window-*building* pipeline, which this repository never calls, was not ported) |

No scientific behavior changed; this is strictly a dependency-removal
refactor, verified by a clean-room install and full test suite (see
`docs/REPRODUCIBILITY.md`).

## 4. FC (fault classification): class count, output head, and Table 3

Three distinct things are easy to conflate here, so they are stated
separately up front. All three are established by direct evidence, set out
below.

| | |
|---|---|
| **11 semantic FC classes** | The task has 11 classes (`y_FC ∈ {c0,...,c10}`), as the manuscript states. |
| **11-output final FC campaign** | The training campaign that produced the published FC *results* (Table 5 and its appendix) used an **11-unit** output head. |
| **12-output earlier configuration** | The parameter counts reported in **Table 3** correspond to an **earlier, separate** experiment that used a **12-unit** FC output head. |

In short -- **historical parameter-count inconsistency: the parameter counts
reported in Table 3 correspond to an earlier 12-output FC configuration,
whereas the final FC campaign used an 11-output head.** This concerns the
parameter counts only. It does not affect the published FC performance
results, whose source campaign is identified and archived here (§8).

### 4.1 The 11 classes and the 11-output head

PROTECT-90 realizes exactly 11 distinct FC label values (`no_fault` plus
the 10 short-circuit types
`sc_type ∈ {0,1,2,3}` combined with phase selection produce), matching the
manuscript's `y_FC ∈ {c0,...,c10}` (Section 3.4). The current codebase (and,
per direct commit inspection below, the codebase actually used for the
final training sweep) derives the classification head's output dimension
**dynamically** as `len(class_to_idx)` from whatever label values are
actually present in the data (`scripts/run_dl_experiment.py`), which
evaluates to 11 for this data. `constants.FAULT_TARGET_TO_OUTPUT_DIM`
deliberately has no `y_fault_class` entry -- there is no fixed fallback for
this target in the current training path.

The archived export of the final campaign (§8) corroborates this directly
and independently of any code reading: every FC run's summary logs per-class
metrics for exactly `class_0` through `class_10` -- 11 classes, with no
twelfth ever populated.

### 4.2 Why Table 3's counts differ

The open question was whether the *specific trained models* behind the
*published* Table 3 (parameter counts) and Table 5 (FC results) are the same
models. They are not. Two separate training campaigns exist in the private
development repository's history:

1. **An earlier campaign (~Oct 2025, commit `eca94ba6`, script
   `src/dl_psp/models/run_model.py`, since deleted)** logged
   `num_trainable_params` to MLflow (experiment `915141650397191911`) for
   all 9 architectures at FC/50ms. Recomputing each architecture's parameter
   count with this repository's current model code at 11 vs. 12 output
   units gives:

   | Model | MLflow/Table 3 count | 11-output count | 12-output count | Match |
   | --- | ---: | ---: | ---: | :---: |
   | CNN | 44,684 | 44,555 | 44,684 | **12** |
   | RNN | 57,356 | 57,227 | 57,356 | **12** |
   | Dilated CNN | 69,388 | 69,259 | 69,388 | **12** |
   | TCN | 118,924 | 118,795 | 118,924 | **12** |
   | GRU | 168,972 | 168,843 | 168,972 | **12** |
   | LSTM | 224,780 | 224,651 | 224,780 | **12** |
   | CNN-LSTM | 308,876 | 308,747 | 308,876 | **12** |
   | InceptionTime | 1,004,044 | 1,003,531 | 1,004,044 | **12** |
   | TFT | 1,234,828 | 1,234,699 | 1,234,828 | **12** |

   All 9 match the 12-output count exactly and none match the 11-output
   count. This campaign's `run_model.py` called
   `create_model_from_name(config)` with no explicit `out_dim`, so `out_dim`
   came from `psp_helper.constants.FAULT_TARGET_TO_OUTPUT_DIM.get("fault_class")`
   (note: target name `fault_class`, without the `y_` prefix used later) --
   i.e. a **hardcoded** map lookup, not dynamic derivation. That map, at the
   external `psp_helper` package's state at the time, must have resolved
   `fault_class` to 12; the currently-available `psp_helper` snapshot no
   longer has an entry for this target at all (§3), so this hardcoded value
   was evidently changed or removed at some point after Oct 2025 and is not
   independently recoverable beyond this inference.

   **These 9 parameter counts match Table 3 exactly, digit for digit, for
   every architecture.** Table 3 was therefore generated from this earlier,
   12-output-head campaign.

2. **The final campaign (Feb 2026, commits `2763ee31`/`05e3a677`/`2ad63086`,
   archived in full at `results/paper/main/raw/`)** used the current
   `run_dl_experiment.py` logic, confirmed by direct inspection of
   `src/dl_psp/models/run_dl_experiment.py` at commit `2763ee31` (the commit
   behind 147 of the 180 selected runs): `out_dim = len(class_to_idx)`,
   computed dynamically from `canonicalize_multiclass_encoding(y_all, ...)`
   -- i.e. genuinely data-driven, evaluating to 11 for FC. This campaign
   never logged parameter counts to W&B (`model/num_params` is absent from
   every one of the 265 exported summaries -- see §8), so it could not be
   the source of Table 3.

   **This campaign's FC results are the source of published Table 5**: the
   regenerated `table_y_fault_class_main.csv` and `table_y_fault_class_appendix.csv`
   (`results/paper/main/`, built purely from this archive) match the
   manuscript's `tab_fc_main_review.tex` and its Accuracy-appendix table
   **exactly, cell-for-cell, to the published precision, across all 9
   models and 5 window lengths** (macro-F1 and Accuracy; see
   `tests/unit/test_main_results_archive.py::test_fc_main_and_accuracy_tables_match_the_published_manuscript`).
   This is not a plausible-sounding coincidence: 90 independently-varying
   numbers matching exactly is conclusive that this is the actual source of
   the published FC results, and that those results came from 11-output
   models.

### 4.3 Conclusion and disposition

The FC task was **trained with an 11-unit head** for the campaign whose
results are published in Table 5 (and, by the same code path, Tables 4/6/7
and the reviewer-added analyses) -- consistent with the manuscript's 11
semantic classes. **Table 3's published parameter counts, for all 9
architectures, instead reflect the earlier, separate campaign that used a
12-unit FC head** via a since-changed external dependency, and were never
regenerated against the campaign that actually produced the published
results.

This is a historical provenance inconsistency confined to Table 3's
parameter counts. It is not an ambiguity in the data, not a defect in this
repository's current code, and it does not call the published FC
performance results into question -- those are reproduced exactly from the
archived raw runs (§8).

**Disposition:** no model definition, constant, target logic, archived
result, or test was changed as a result of this finding. The current
dynamic 11-output behavior is exactly what produced the published FC
results and is preserved as-is. Table 3's parameter counts are neither
"corrected" to 11-output values (that would misrepresent what was
reported) nor re-derived from the final campaign (which never logged them).
Both the discrepancy and the correct current behavior are locked in by
`tests/unit/test_main_results_archive.py`.

## 5. Cross-validation split

See [`config/paper/splits/SPLIT_LIMITATION.md`](../config/paper/splits/SPLIT_LIMITATION.md).
In short: the splitting *algorithm and seed* (episode-grouped 5-fold,
`StratifiedGroupKFold` for multiclass / `GroupKFold` otherwise, seed 42) are
frozen and tested (`tests/unit/test_cv_utils.py`). The exact realized
episode-to-fold assignment used for the published Tables 4-10 is **not**
recoverable: no split manifest was ever committed to either repository, and
the matching MLflow run artifact directories in the private repository (354
runs, checked) contain no saved fold-split files or checkpoints. Re-running
the frozen algorithm with the frozen seed produces *a* valid split
satisfying the same invariants, not provably the *same* split byte-for-byte
across scikit-learn versions.

## 6. Runtime benchmark (Table 11)

See [`config/paper/runtime/README.md`](../config/paper/runtime/README.md)
for the recovered implementation details. Stated precisely, so this does not
read as a generic reproducibility caveat: **the archived Table 11 values are
preserved as the values reported in the publication. The surviving runtime
implementation and run metadata do not fully support the manuscript's
stated single-sample, fixed-RTX-3080 protocol** -- the recovered timing code
(`measure_runtime`, private-repo git history only, deleted before the
commit this release is based on) times a batch of
`min(config.batch_size, len(val_set))` (up to 256 samples, not a single
sample), and a raw per-model overview CSV from the same private repository
records `inference_hardware` varying across models (Tesla V100, RTX 2080 Ti,
RTX 3080), not one fixed GPU. Table 11's own caption ("aggregated over the
latest completed runs") is consistent with this being a rolling,
mixed-hardware measurement practice rather than a single controlled sweep.
**Table 11 is therefore treated as an archived publication result, not a
fully reproducible hardware benchmark.** No code was changed, no
re-measurement was performed, and no corrected numbers were computed for
this release.

## 7. Environment

Confirmed from the 180 selected (post-`dedupe_latest`) runs behind Tables
4-7 (the author's Weights & Biases export archived at
`results/paper/main/raw/`): **Python 3.12.11, PyTorch 2.5.1+cu121, NumPy
2.3.4, CUDA 12.1 -- identical across all 180 selected runs**, i.e. across
every architecture, target, and window length, with zero variation. This
was checked explicitly (not assumed): grouping the 180 selected runs by
`torch_version`/`numpy_version`/`python_version`/`env/cuda_version` yields
exactly one value each. This metadata is captured programmatically by
`utils/run_utils.get_env_info()` (`torch.__version__`, `np.__version__`,
`sys.version`, `platform.node()`) at the time each run executed, not
manually entered into a config file. The only metadata that *does* vary
across the 180 selected runs is the OS kernel string (`env/platform`: two
RHEL 8.10 patch levels and two Ubuntu-based kernels, split across 19 unique
HPC node hostnames) -- an expected, benign consequence of distributing runs
across many cluster nodes, and irrelevant to the Python/PyTorch/CUDA stack
above.

This is a materially better answer than what was recoverable before this
export existed: `hpc/run_deep_learning_models.sh` independently confirms
Python 3.12 and an NVIDIA RTX 3080 (`--gres=gpu:rtx3080:1`) for the intended
hardware, and no pinned environment file (conda export, `pip freeze`,
lockfile) exists in either repository as a redundant cross-check, but the
W&B export's per-run config is direct evidence from the runs that produced
the published numbers, not an incidental unrelated snapshot. (An earlier,
weaker data point -- an unrelated smoke-test run's MLflow environment
snapshot, `torch==2.8.0+cu128` -- is superseded by this and no longer the
best available evidence.)

**Caveat:** every one of the 180 selected runs has `git_status=dirty` --
i.e. each run's exact code state includes uncommitted local changes beyond
the recorded commit SHA (§8), which are not captured or recoverable from
this export. The *environment* (interpreter/library versions) is fully
pinned and verified; the *exact code diff* at run time is not.

## 8. Main-table (4-7) result archive

Now archived locally: the author exported the live Weights & Biases project
(`julian_oelhaf/dl_comparison_hv_double_line_90kv`) via the API into
`results/paper/main/raw/` (`configs.jsonl`, `summaries.jsonl`, `runs.csv`,
`history/`).

**Run breakdown:**

```
total archived runs:                265
  finished:                          253
  crashed:                           12
  failed / running / other:          0
unique (window, target, model) combinations:   180  (full grid, zero gaps)
combinations with more than one run (any state): 63
maximum attempts for one combination:            5  (all 5 "finished";
                                                      the 4 combinations
                                                      that reach 5 attempts
                                                      are all FL cnn/rnn
                                                      regressors at 10/20ms)
selected runs after dedupe_latest (keep latest per combination): 180
```

(An earlier draft of this section mis-stated 85 combinations with retries;
the correct figure, recomputed directly from `raw/`, is 63.)

**Selection rule provenance:** `dedupe_latest()` (keep the run with the
latest `created_at` per combination) is not merely "this script's current
rule" adopted for convenience -- it is the same rule
`make_tables_wandb.py` already used against the live API before this
release, and it is now directly validated: regenerating every FC cell from
`raw/` with this rule reproduces the *published manuscript's* Table 5
(macro-F1) and its Accuracy-appendix table exactly, cell-for-cell, across
all 9 models and 5 window lengths (§4,
`tests/unit/test_main_results_archive.py`). No crashed run is ever the
latest attempt for its combination, so the rule never silently selects an
incomplete run.

**Git commit provenance:** `git_commit` and `git_status` (of the private
development repository) are recorded per run; no `branch` or source-repository
URL field is present in the export. Across the 265 raw runs, 4 distinct
commits appear (`2763ee31`: 177, `05e3a677`: 47, `2ad63086`: 40, `4d7cab4e`: 1).
Restricted to the 180 *selected* runs, only 3 remain (`4d7cab4e`'s single
run was itself superseded by a later retry): `2763ee31` (147), `05e3a677`
(26), `2ad63086` (7) -- all three are small, sequential same-day/adjacent-day
commits in February 2026 (confirmed to exist directly in the private
repository's history, not inferred from timestamps). **Every one of the 180
selected runs has `git_status=dirty`**: the recorded commit is a lower
bound on the code state, not a guarantee of an exact, uncommitted-change-free
revision. This is disclosed rather than treated as resolved.

`src/dl_fault_analysis/evaluation/make_tables_wandb.py` was extended with an
offline mode (`WANDB_EXPORT_DIR`) that regenerates `table_y_*_{main,appendix}.{csv,tex}`
from this archive with no W&B account or network access needed. The offline
loader (`wandb_runs_to_df_from_export`) and the live-API loader
(`wandb_runs_to_df`) both call the same `_row_from_config_summary()` for
per-run field extraction and the same `dedupe_latest`/`prepare_wide`/
`render_transposed`/`wide_to_csv` for selection, aggregation, and
formatting -- there is one implementation of this logic, not two
independently-maintained copies, so behavioral parity between the two entry
points is structural, not merely tested. (A live-API-vs-offline-export
comparison for the same run IDs was not additionally performed here: doing
so would require live W&B credentials, which this release does not use.)
See [`results/paper/main/PROVENANCE.md`](../results/paper/main/PROVENANCE.md)
for exact coverage and two remaining gaps: Table 3's parameter counts (see
§4) and the FL appendix's R^2/P90 columns are not present in this export and
are not fabricated to fill the gap.

## 9. Metadata corrections made in this release

- Paper title, DOI, journal volume, and article number updated to the final
  published record (`10.1016/j.epsr.2026.114169`, *Electric Power Systems
  Research*, Vol. 265 (2027), Art. 114169) in `README.md` and `CITATION.cff`.
- Dataset DOI updated from an earlier Zenodo record to the final PROTECT-90
  v1.0.0 release (`10.5281/zenodo.21109169`).
- `pyproject.toml` / `CITATION.cff` repository URL corrected from
  `dl_fault_analysis` (underscore, never a real GitHub URL) to
  `dl-fault-analysis` (the actual remote).
- Stale "PSCC protocol" docstring/comment in `run_dl_experiment.py` reworded
  to describe the actual EPSR episode-grouped 5-fold protocol (leftover from
  an earlier PSCC-conference submission of the same code).
- A hardcoded FAU/RRZE internal proxy hostname removed from
  `hpc/run_deep_learning_models.sh`.
- Hardcoded personal W&B entity/project defaults in three files replaced
  with the same placeholder convention already used in
  `config/tracking/default.yaml`.
