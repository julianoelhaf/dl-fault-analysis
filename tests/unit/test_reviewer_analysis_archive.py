"""Verify the archived reviewer-added analyses (Tables 8 and 9) hang together.

These experiments train models, so their numbers are not reproducible by
re-running them (see docs/REPRODUCIBILITY.md). What *is* checkable, now that
the intermediate artifacts have been ported alongside the publication tables,
is the derivation chain that produced those tables:

    per-window predictions   -- NOT ARCHIVED, never committed anywhere
        |
        v
    fold_metrics.csv / perturbation_metrics.csv   (stage 2, per fold)
        |  _aggregate()
        v
    aggregate_metrics.csv                          (stage 3, mean/std)
        |  _fmt()
        v
    table_*.{csv,tex}                              (stage 4, published)

Stages 2->3->4 are all archived, so these tests re-derive each step with the
repository's own shipped code and assert it lands on the archived values.
That also pins the port itself: the publication tables and the intermediate
metrics were extracted independently from the source commits, so a drift on
either side fails here.

Stage 1 is gone and cannot be reconstructed -- see
test_per_window_predictions_are_documented_as_unavailable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from dl_fault_analysis.scripts.run_perturbation import _aggregate as perturbation_aggregate
from dl_fault_analysis.scripts.run_revision_observability import _aggregate as observability_aggregate

REPO_ROOT = Path(__file__).resolve().parents[2]
OBS_DIR = REPO_ROOT / "results" / "paper" / "observability"
COMM_DIR = REPO_ROOT / "results" / "paper" / "communication"

# Tolerance is float-repr noise only: these are the same arithmetic on the
# same inputs, so anything above ~1e-12 means a value actually changed.
TOL = 1e-12

_AGG_COLUMNS = ["mae_mean", "mae_std", "rmse_mean", "rmse_std", "delta_mae_abs", "delta_mae_rel"]


def _cell(mean: float, std: float) -> str:
    """The published cell format, matching run_revision_observability._fmt."""
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


# ---------------------------------------------------------------------------
# Stage 2 -> stage 3
# ---------------------------------------------------------------------------


def test_observability_aggregate_is_reproducible_from_fold_metrics():
    fold = pd.read_csv(OBS_DIR / "fold_metrics.csv")
    archived = pd.read_csv(OBS_DIR / "aggregate_metrics.csv")

    recomputed = observability_aggregate(fold.to_dict("records"))
    merged = archived.merge(recomputed, on=["observability", "model"], suffixes=("_a", "_r"))

    assert len(merged) == len(archived) == 12
    for col in _AGG_COLUMNS:
        delta = (merged[f"{col}_a"].astype(float) - merged[f"{col}_r"].astype(float)).abs().max()
        assert delta < TOL, f"{col} drifted by {delta:g} between fold and aggregate metrics"


def test_perturbation_aggregate_is_reproducible_from_fold_metrics():
    fold = pd.read_csv(COMM_DIR / "perturbation_metrics.csv")
    archived = pd.read_csv(COMM_DIR / "aggregate_metrics.csv")

    recomputed = perturbation_aggregate(fold.to_dict("records"))
    merged = archived.merge(recomputed, on=["perturbation", "model"], suffixes=("_a", "_r"))

    assert len(merged) == len(archived) == 3
    for col in _AGG_COLUMNS:
        delta = (merged[f"{col}_a"].astype(float) - merged[f"{col}_r"].astype(float)).abs().max()
        assert delta < TOL, f"{col} drifted by {delta:g} between fold and aggregate metrics"


# ---------------------------------------------------------------------------
# Stage 3 -> stage 4 (the published tables)
# ---------------------------------------------------------------------------


def test_observability_published_table_matches_aggregate_metrics():
    agg = pd.read_csv(OBS_DIR / "aggregate_metrics.csv")
    table = pd.read_csv(OBS_DIR / "table_reduced_observability.csv")

    model_columns = [c for c in table.columns if c.endswith("_mae")]
    assert model_columns, "no model columns found in the published table"

    checked = 0
    for _, row in table.iterrows():
        for col in model_columns:
            model = col[: -len("_mae")]
            sel = agg[(agg["observability"] == row["observability"]) & (agg["model"] == model)]
            assert len(sel) == 1, f"no unique aggregate row for {row['observability']}/{model}"
            expected = _cell(float(sel["mae_mean"].iloc[0]), float(sel["mae_std"].iloc[0]))
            assert row[col] == expected, (
                f"published cell {row['observability']}/{model} is {row[col]!r}, "
                f"but aggregate_metrics.csv implies {expected!r}"
            )
            checked += 1
    assert checked == 12


def test_perturbation_published_table_matches_aggregate_metrics():
    agg = pd.read_csv(COMM_DIR / "aggregate_metrics.csv")
    table = pd.read_csv(COMM_DIR / "table_perturbation.csv")

    (col,) = [c for c in table.columns if c.endswith("_mae")]
    model = col[: -len("_mae")]

    for _, row in table.iterrows():
        sel = agg[(agg["perturbation"] == row["perturbation"]) & (agg["model"] == model)]
        assert len(sel) == 1
        expected = _cell(float(sel["mae_mean"].iloc[0]), float(sel["mae_std"].iloc[0]))
        assert row[col] == expected, (
            f"published cell {row['perturbation']} is {row[col]!r}, "
            f"but aggregate_metrics.csv implies {expected!r}"
        )


# ---------------------------------------------------------------------------
# The manuscript's relative degradation figures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "perturbation, published_percent",
    [
        # Section 3.6 quotes these as roughly +0.5% and +118.6% relative to the
        # unperturbed baseline. They live only in aggregate_metrics.csv --
        # table_perturbation.csv carries absolute MAE and no delta column.
        ("time_shift_2_samples", 0.5),
        ("block_loss_10ms", 118.6),
    ],
)
def test_perturbation_relative_degradation_is_traceable_to_aggregate_metrics(
    perturbation, published_percent
):
    agg = pd.read_csv(COMM_DIR / "aggregate_metrics.csv")
    sel = agg[agg["perturbation"] == perturbation]
    assert len(sel) == 1

    delta_rel_percent = float(sel["delta_mae_rel"].iloc[0]) * 100.0
    assert delta_rel_percent == pytest.approx(published_percent, abs=0.05), (
        f"{perturbation}: archived delta_mae_rel is {delta_rel_percent:.2f}%, "
        f"manuscript reports ~{published_percent}%"
    )

    # And the delta is itself consistent with the archived absolute MAEs.
    baseline = float(agg[agg["perturbation"] == "none"]["mae_mean"].iloc[0])
    expected = (float(sel["mae_mean"].iloc[0]) - baseline) / baseline
    assert float(sel["delta_mae_rel"].iloc[0]) == pytest.approx(expected, abs=TOL)


# ---------------------------------------------------------------------------
# What is deliberately absent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("directory", [OBS_DIR, COMM_DIR])
def test_per_window_predictions_are_documented_as_unavailable(directory):
    """Stage 1 never survived; guard against a later claim that it did.

    The generating scripts can write predictions.parquet / predictions.csv.gz,
    but those were never committed to either repository. The provenance record
    must keep saying so rather than quietly implying the tables are checkable
    all the way down to per-window output.
    """
    assert not list(directory.glob("predictions.*")), (
        f"{directory.name}: per-window predictions appeared -- if these were genuinely "
        "recovered, update PROVENANCE.md instead of leaving it stating they are unavailable"
    )

    provenance = (directory / "PROVENANCE.md").read_text(encoding="utf-8")
    assert re.search(r"never committed", provenance), (
        f"{directory.name}/PROVENANCE.md must state that per-window predictions were "
        "never committed"
    )
