"""Integration smoke test for the communication-perturbation runner (tiny synthetic, CPU).

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from dl_fault_analysis.scripts.run_perturbation import (
    PerturbationConfig,
    run_perturbation_experiment,
)

pytestmark = pytest.mark.integration

T_SMOKE, F = 16, 48
FAULT_LINES = ["Line_1_2_a", "Line_1_2_b", "Line_2_3_a", "Line_2_3_b"]
_POINTS = [
    (1, 1, 2, "A"),
    (1, 1, 2, "B"),
    (2, 1, 2, "A"),
    (2, 1, 2, "B"),
    (2, 2, 3, "A"),
    (2, 2, 3, "B"),
    (3, 2, 3, "A"),
    (3, 2, 3, "B"),
]


def _feature_names() -> list[str]:
    return [
        f"Bus_{b}_Line_{f:02d}_{t:02d}{c}_{q}_L{p}_{u}"
        for b, f, t, c in _POINTS
        for q, u in (("cur", "A"), ("vol", "V"))
        for p in (1, 2, 3)
    ]


def _labels() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows, sid = [], 0
    for line in FAULT_LINES:
        for _ in range(6):
            loc = float(rng.uniform(1.0, 99.0))
            for w in range(4):
                rows.append(
                    {
                        "sample_id": sid,
                        "window_idx": w,
                        "window_start_time": 0.1 + 0.005 * w,
                        "window_duration": 0.05,
                        "status": "fault_start" if w == 0 else "in_fault",
                        "y_fault_present": 1,
                        "y_fault_line": line,
                        "y_fault_location": loc,
                    }
                )
            sid += 1
    return pd.DataFrame(rows)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            model_name="gru_regressor",
            hidden_size=8,
            num_layers=1,
            bidirectional=False,
            dropout=0.0,
            input_size=None,
        ),
        dataset=SimpleNamespace(
            topology="hv_double_line_90kv", sampling_frequency=T_SMOKE
        ),
        window_extraction=SimpleNamespace(window_length=1.0),
        training=SimpleNamespace(target_label="y_fault_location"),
    )


@pytest.fixture(scope="module")
def pert_run(tmp_path_factory):
    labels = _labels()
    X = (
        np.random.default_rng(1)
        .standard_normal((len(labels), T_SMOKE, F))
        .astype(np.float32)
    )
    meta = {
        "feature_names": _feature_names(),
        "topology": "hv_double_line_90kv",
        "window_length": 1.0,
    }
    cfg = PerturbationConfig(
        n_folds=2, epochs=2, patience=2, batch_size=8, num_workers=0, seed=42
    )
    out_dir = str(tmp_path_factory.mktemp("perturbation_run"))
    result = run_perturbation_experiment(
        config=_config(),
        X=X,
        labels_df=labels,
        meta=meta,
        cfg=cfg,
        out_dir=out_dir,
        device=torch.device("cpu"),
        topology="hv_double_line_90kv",
    )
    return result, out_dir


def test_perturbation_outputs_exist(pert_run):
    _, out_dir = pert_run
    for fname in [
        "perturbation_metrics.csv",
        "aggregate_metrics.csv",
        "table_perturbation.csv",
        "table_perturbation.tex",
        "summary.md",
        "split_audit.json",
        "normalization_audit.json",
        "run_config.yaml",
    ]:
        assert os.path.exists(os.path.join(out_dir, fname)), f"missing {fname}"


def test_perturbation_schema_and_baseline(pert_run):
    result, out_dir = pert_run
    assert result["audits_passed"] is True
    agg = pd.read_csv(os.path.join(out_dir, "aggregate_metrics.csv"))
    for col in [
        "experiment",
        "perturbation",
        "model",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "delta_mae_abs",
        "delta_mae_rel",
        "n_episodes",
        "n_windows",
        "line_conditioned",
        "deployment_interpretation",
    ]:
        assert col in agg.columns, f"aggregate missing {col}"
    assert set(agg["perturbation"]) == {
        "none",
        "time_shift_2_samples",
        "block_loss_10ms",
    }
    none = agg[agg["perturbation"] == "none"].iloc[0]
    assert none["delta_mae_abs"] == 0.0 and none["delta_mae_rel"] == 0.0
    assert not bool(agg["line_conditioned"].iloc[0])  # full observability


def test_perturbation_predictions(pert_run):
    _, out_dir = pert_run
    pq = os.path.join(out_dir, "predictions.parquet")
    gz = os.path.join(out_dir, "predictions.csv.gz")
    path = pq if os.path.exists(pq) else gz
    assert os.path.exists(path)
    preds = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    assert {"perturbation", "fold", "sample_id", "y_true", "y_pred"} <= set(
        preds.columns
    )
    assert set(preds["perturbation"]) == {
        "none",
        "time_shift_2_samples",
        "block_loss_10ms",
    }
