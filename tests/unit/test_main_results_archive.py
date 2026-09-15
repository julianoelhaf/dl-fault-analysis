"""Verification for the archived main-benchmark results (Tables 4-7).

Regenerates the derived tables from the committed raw W&B export
(results/paper/main/raw/) and checks them against the committed CSVs and
against known invariants (full grid coverage, cross-references to values
documented elsewhere) -- this is stronger than testing that a CSV merely
exists, and requires no network access, W&B account, or GPU.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dl_fault_analysis.evaluation.make_tables_wandb import (
    FAULT_TARGETS,
    WINDOW_ORDER_MS,
    dedupe_latest,
    prepare_wide,
    specs_appendix,
    specs_main,
    wandb_runs_to_df_from_export,
    wide_to_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "results" / "paper" / "main" / "raw"
ARCHIVE_DIR = REPO_ROOT / "results" / "paper" / "main"

MODELS_PER_TARGET = 9
N_WINDOWS = len(WINDOW_ORDER_MS)


@pytest.fixture(scope="module")
def deduped_df() -> pd.DataFrame:
    df = wandb_runs_to_df_from_export(str(RAW_DIR))
    return dedupe_latest(df)


def test_raw_export_covers_the_full_grid_with_no_gaps(deduped_df: pd.DataFrame):
    combos = deduped_df[
        ["params.fault_target", "params.window_ms", "params.model"]
    ].drop_duplicates()
    assert len(combos) == len(FAULT_TARGETS) * N_WINDOWS * MODELS_PER_TARGET

    for target in FAULT_TARGETS:
        sub = combos[combos["params.fault_target"] == target]
        assert set(sub["params.window_ms"]) == set(WINDOW_ORDER_MS)
        assert sub["params.model"].nunique() == MODELS_PER_TARGET


def test_dedupe_keeps_exactly_one_row_per_combination(deduped_df: pd.DataFrame):
    key = ["params.fault_target", "params.window_ms", "params.model"]
    counts = deduped_df.groupby(key).size()
    assert (counts == 1).all()


@pytest.mark.parametrize("target", ["y_fault_present", "y_fault_class", "y_fault_line", "y_fault_location"])
def test_regenerated_main_table_matches_committed_csv(deduped_df: pd.DataFrame, target: str):
    specs = specs_main(target)
    mean_wide, std_wide = prepare_wide(deduped_df, target, specs)
    regenerated = wide_to_csv(mean_wide, std_wide, specs)

    committed_path = ARCHIVE_DIR / f"table_{target}_main.csv"
    committed = pd.read_csv(committed_path)

    pd.testing.assert_frame_equal(
        regenerated.reset_index(drop=True), committed.reset_index(drop=True)
    )


def test_fl_gru_50ms_mae_matches_the_value_cited_in_the_impedance_baseline_doc(
    deduped_df: pd.DataFrame,
):
    # docs/revision_impedance_baseline.md cites "DL GRU (50 ms, full obs) --
    # paper tuned MAE: 7.57" as an independent cross-reference for the main
    # benchmark's tuned result; this is an unrelated document (ported from
    # the private repo's reviewer-response work) computed independently of
    # this W&B export, so agreement is a genuine cross-check.
    row = deduped_df[
        (deduped_df["params.fault_target"] == "y_fault_location")
        & (deduped_df["params.window_ms"] == 50)
        & (deduped_df["params.model"] == "gru_regressor")
    ]
    assert len(row) == 1
    mae_pct = float(row["metrics.mae_mean"].iloc[0])
    assert mae_pct == pytest.approx(7.57, abs=0.01)


def test_table_3_params_and_fl_appendix_r2_p90_are_honestly_absent(deduped_df: pd.DataFrame):
    # These are documented gaps (results/paper/main/PROVENANCE.md), not
    # silently-missing data -- assert they are actually absent rather than
    # accidentally present-but-unused, so the documented limitation stays
    # accurate if the export is ever refreshed.
    assert deduped_df["params.num_params"].isna().all()

    specs = specs_appendix("y_fault_location")
    mean_wide, _ = prepare_wide(deduped_df, "y_fault_location", specs)
    r2_cols = [c for c in mean_wide.columns if c[1] == "$R^2$"]
    assert mean_wide[r2_cols].isna().all().all()


# Verbatim from the published manuscript's tab_fc_main_review.tex (macro-F1,
# mean±std) and tab_fc_accuracy_app_review (Accuracy) -- see
# docs/PROVENANCE.md §4 for why this exact-match is the decisive evidence
# that the archived W&B export is the source of the published FC results
# (and therefore that those results used an 11-output head, contra Table 3).
_PUBLISHED_FC_MACRO_F1 = {
    "InceptionTime": ["0.998 ± 0.002", "0.999 ± 0.001", "0.999 ± 0.000", "0.999 ± 0.001", "0.999 ± 0.000"],
    "GRU": ["0.995 ± 0.003", "0.997 ± 0.002", "0.998 ± 0.000", "0.998 ± 0.001", "0.999 ± 0.001"],
    "CNN--LSTM": ["0.996 ± 0.002", "0.997 ± 0.001", "0.998 ± 0.001", "0.998 ± 0.001", "0.998 ± 0.000"],
    "LSTM": ["0.995 ± 0.001", "0.998 ± 0.001", "0.998 ± 0.001", "0.998 ± 0.001", "0.998 ± 0.001"],
    "TFT": ["0.996 ± 0.003", "0.996 ± 0.001", "0.995 ± 0.003", "0.996 ± 0.001", "0.996 ± 0.001"],
    "Dilated CNN": ["0.988 ± 0.004", "0.991 ± 0.002", "0.992 ± 0.002", "0.991 ± 0.002", "0.993 ± 0.001"],
    "CNN": ["0.966 ± 0.005", "0.966 ± 0.006", "0.966 ± 0.005", "0.969 ± 0.008", "0.977 ± 0.001"],
    "TCN": ["0.781 ± 0.014", "0.766 ± 0.003", "0.738 ± 0.027", "0.721 ± 0.017", "0.803 ± 0.006"],
    "RNN": ["0.446 ± 0.489", "0.326 ± 0.223", "0.335 ± 0.056", "0.312 ± 0.317", "0.730 ± 0.057"],
}
_PUBLISHED_FC_ACCURACY = {
    "InceptionTime": ["0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.000"],
    "GRU": ["0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.000"],
    "LSTM": ["0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.000"],
    "CNN--LSTM": ["0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.000"],
    "TFT": ["0.999 ± 0.000", "0.999 ± 0.000", "0.999 ± 0.001", "0.998 ± 0.000", "0.998 ± 0.001"],
    "Dilated CNN": ["0.999 ± 0.000", "0.998 ± 0.000", "0.997 ± 0.001", "0.996 ± 0.001", "0.996 ± 0.001"],
    "CNN": ["0.998 ± 0.000", "0.994 ± 0.001", "0.989 ± 0.002", "0.985 ± 0.004", "0.984 ± 0.001"],
    "TCN": ["0.987 ± 0.001", "0.961 ± 0.001", "0.925 ± 0.010", "0.858 ± 0.016", "0.844 ± 0.004"],
    "RNN": ["0.980 ± 0.017", "0.916 ± 0.022", "0.835 ± 0.005", "0.770 ± 0.068", "0.795 ± 0.033"],
}


def test_fc_main_and_accuracy_tables_match_the_published_manuscript(deduped_df: pd.DataFrame):
    main_specs = specs_main("y_fault_class")
    mean_wide, std_wide = prepare_wide(deduped_df, "y_fault_class", main_specs)
    f1_csv = wide_to_csv(mean_wide, std_wide, main_specs).set_index("model")
    f1_cols = [f"{w}ms_F1" for w in WINDOW_ORDER_MS]
    for model, expected in _PUBLISHED_FC_MACRO_F1.items():
        assert list(f1_csv.loc[model, f1_cols]) == expected, model

    app_specs = specs_appendix("y_fault_class")
    mean_wide, std_wide = prepare_wide(deduped_df, "y_fault_class", app_specs)
    acc_csv = wide_to_csv(mean_wide, std_wide, app_specs).set_index("model")
    acc_cols = [f"{w}ms_Accuracy" for w in WINDOW_ORDER_MS]
    for model, expected in _PUBLISHED_FC_ACCURACY.items():
        assert list(acc_csv.loc[model, acc_cols]) == expected, model


def test_fc_parameter_counts_match_a_12_output_head_not_11():
    # Locks in the docs/PROVENANCE.md §4 finding: the parameter counts that
    # match published Table 3 come from a 12-unit FC head, not the 11-unit
    # head this repository's dynamic out_dim derivation actually produces
    # for FC. This does not mean the current code is wrong -- see §4 for why
    # the current 11-output behavior is what produced the published FC
    # *results* (Table 5) -- it documents the Table 3 discrepancy as a
    # locked-in fact so it cannot silently drift out of sync with this test.
    from types import SimpleNamespace

    from dl_fault_analysis.models.model_utils import create_model_from_name

    def cfg(model_name: str) -> SimpleNamespace:
        return SimpleNamespace(
            model=SimpleNamespace(
                model_name=model_name, hidden_size=128, num_layers=2, bidirectional=False, dropout=0.1
            ),
            dataset=SimpleNamespace(sampling_frequency=6400),
            window_extraction=SimpleNamespace(window_length=0.05),
            training=SimpleNamespace(target_label="y_fault_class"),
        )

    def count_params(model) -> int:
        return sum(p.numel() for p in model.parameters())

    # model -> (published Table 3 / MLflow count, expected 11-out, expected 12-out)
    published_table3 = {
        "cnn_classifier": 44684,
        "rnn_classifier": 57356,
        "dilated_cnn_classifier": 69388,
        "tcn_classifier": 118924,
        "gru_classifier": 168972,
        "lstm_classifier": 224780,
        "cnn_lstm_classifier": 308876,
        "inceptiontime_classifier": 1004044,
        "tft_classifier": 1234828,
    }
    for model_name, published in published_table3.items():
        p11 = count_params(create_model_from_name(cfg(model_name), n_features=48, out_dim=11))
        p12 = count_params(create_model_from_name(cfg(model_name), n_features=48, out_dim=12))
        assert p12 == published, f"{model_name}: 12-output count {p12} != published {published}"
        assert p11 != published, (
            f"{model_name}: 11-output count {p11} unexpectedly matches published {published} "
            "-- the docs/PROVENANCE.md §4 finding may need re-checking"
        )


# ---------------------------------------------------------------------------
# dedupe_latest: retry selection must not be decided by missing metadata
# ---------------------------------------------------------------------------


def _run_row(run_id: str, created_at, model: str = "gru_classifier") -> dict:
    return {
        "params.dataset": "hv_double_line_90kv",
        "params.fault_target": "y_fault_present",
        "params.window_ms": 50,
        "params.model": model,
        "meta.run_id": run_id,
        "meta.created_at": created_at,
    }


def test_dedupe_latest_picks_the_newest_run():
    df = pd.DataFrame(
        [
            _run_row("older", "2026-02-01T10:00:00Z"),
            _run_row("newest", "2026-02-10T10:00:00Z"),
            _run_row("middle", "2026-02-05T10:00:00Z"),
        ]
    )
    assert dedupe_latest(df)["meta.run_id"].tolist() == ["newest"]


def test_dedupe_latest_never_prefers_a_run_with_no_timestamp():
    """A run we cannot order must lose to one we can.

    NaT sorts last by default, so the obvious sort/drop_duplicates(keep="last")
    would select exactly the run with the least metadata -- which for this
    archive could mean a crashed attempt beating its successful retry.
    """
    df = pd.DataFrame(
        [
            _run_row("has_timestamp_older", "2026-02-01T10:00:00Z"),
            _run_row("has_timestamp_newest", "2026-02-10T10:00:00Z"),
            _run_row("no_timestamp", None),
        ]
    )
    assert dedupe_latest(df)["meta.run_id"].tolist() == ["has_timestamp_newest"]

    # Unparseable strings coerce to NaT and must be treated the same way.
    df_bad = pd.DataFrame(
        [
            _run_row("has_timestamp", "2026-02-01T10:00:00Z"),
            _run_row("unparseable", "not-a-timestamp"),
        ]
    )
    assert dedupe_latest(df_bad)["meta.run_id"].tolist() == ["has_timestamp"]


def test_dedupe_latest_is_deterministic_when_no_timestamps_exist():
    """With runs.csv absent every timestamp is None; selection must still be
    reproducible rather than depending on export/enumeration order."""
    rows = [_run_row("bbb", None), _run_row("aaa", None), _run_row("ccc", None)]
    forward = dedupe_latest(pd.DataFrame(rows))["meta.run_id"].tolist()
    reversed_ = dedupe_latest(pd.DataFrame(rows[::-1]))["meta.run_id"].tolist()
    assert forward == reversed_, "selection changed with input order"


def test_dedupe_latest_keeps_combinations_independent():
    df = pd.DataFrame(
        [
            _run_row("gru_new", "2026-02-10T10:00:00Z", model="gru_classifier"),
            _run_row("gru_old", "2026-02-01T10:00:00Z", model="gru_classifier"),
            _run_row("lstm_only", None, model="lstm_classifier"),
        ]
    )
    selected = set(dedupe_latest(df)["meta.run_id"])
    # lstm has no timestamped alternative, so it is still kept for its own combination.
    assert selected == {"gru_new", "lstm_only"}
