"""Line-conditioned filtering + per-line metric aggregation for reduced-observability FL.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md. Manuscript Section 3.5 / Table 8.

For ``terminal_pair`` / ``single_ended`` we train one model per faulted line:
filter samples to a line, build episode-grouped CV on that line's episodes
only, then aggregate metrics across the per-line models.

Index spaces:
* ``labels_df`` here is the already target-filtered frame (``labels_df_used``
  in the main pipeline) -- for 90 kV FL that is ``status == "fault_start"``
  rows.
* :func:`filter_by_faulted_line` returns **positions into that frame**.
* :func:`build_line_conditioned_splits` returns CV splits whose indices are
  **positions into the per-line subset** (0..n_line-1), so the caller can
  build a per-line filtered space: ``y_line = y_all[line_pos]``,
  ``row_indices_line = valid_row_idx[line_pos]``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import dl_fault_analysis.data.labels as L
from dl_fault_analysis.data.observability import normalize_line_token


def filter_by_faulted_line(
    labels_df: pd.DataFrame,
    line_id: str,
    line_col: str = L.Y_FAULT_LINE,
) -> np.ndarray:
    """Return positional indices of rows whose faulted line == ``line_id``.

    Matching is done on the canonical
    :class:`~dl_fault_analysis.data.observability.LineCircuit` so label form
    (``"Line_1_2_a"``) and feature form (``"Line_01_02A"``) agree.
    """
    if line_col not in labels_df.columns:
        raise ValueError(f"line column '{line_col}' missing from labels_df")
    target = normalize_line_token(line_id)

    def _matches(v: Any) -> bool:
        if not isinstance(v, str):
            return False
        try:
            return normalize_line_token(v) == target
        except ValueError:
            return False

    mask = labels_df[line_col].map(_matches).to_numpy(dtype=bool)
    idx = np.flatnonzero(mask).astype(np.int64)
    if idx.size == 0:
        known = sorted({str(x) for x in labels_df[line_col].dropna().unique().tolist()})
        raise ValueError(
            f"no rows for faulted line {line_id!r} (={target}); present: {known}"
        )
    return idx


def unique_faulted_lines(
    labels_df: pd.DataFrame, line_col: str = L.Y_FAULT_LINE
) -> List[str]:
    """Distinct faulted-line labels present (raw label strings, sorted)."""
    if line_col not in labels_df.columns:
        raise ValueError(f"line column '{line_col}' missing from labels_df")
    return sorted({str(v) for v in labels_df[line_col].dropna().unique().tolist()})


def build_line_conditioned_splits(
    labels_df: pd.DataFrame,
    line_id: str,
    n_folds: int,
    seed: int,
    group_col: str = L.SAMPLE_ID,
    line_col: str = L.Y_FAULT_LINE,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Filter to ``line_id`` and build episode-grouped CV on that subset.

    Returns ``(line_pos, splits)`` where ``line_pos`` are positions into
    ``labels_df`` and each split's indices are positions into the per-line
    subset. Reuses
    :func:`dl_fault_analysis.utils.cv_utils.build_cv_splits_stratified`
    (GroupKFold for regression) so grouping by episode is identical to the
    main pipeline.
    """
    from dl_fault_analysis.utils.cv_utils import build_cv_splits_stratified

    line_pos = filter_by_faulted_line(labels_df, line_id, line_col=line_col)
    sub = labels_df.iloc[line_pos]
    groups_np = sub[group_col].to_numpy()

    n_groups = int(np.unique(groups_np).size)
    if n_groups < n_folds:
        raise ValueError(
            f"line {line_id!r} has only {n_groups} episodes < n_folds={n_folds}"
        )

    # FL is regression -> GroupKFold (y is ignored by GroupKFold).
    dummy_y = np.zeros(len(sub), dtype=np.float32)
    splits = build_cv_splits_stratified(
        y_all=dummy_y,
        groups_np=groups_np,
        task_type="regression",
        n_splits=int(n_folds),
        seed=int(seed),
    )
    return line_pos, splits


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------
def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def aggregate_line_metrics(
    per_line_fold_metrics: Mapping[str, Sequence[Mapping[str, float]]],
    metric_keys: Sequence[str] = ("mae", "rmse"),
) -> Dict[str, Any]:
    """Aggregate per-line, per-fold metrics.

    ``per_line_fold_metrics``: ``{line_id: [ {metric: value, ...} per fold ]}``.
    Returns per-line means and an overall mean/std taken **across all folds of
    all lines** (micro), plus the number of lines and total folds.
    """
    if not per_line_fold_metrics:
        raise ValueError("per_line_fold_metrics is empty")

    per_line: Dict[str, Dict[str, float]] = {}
    pooled: Dict[str, List[float]] = {k: [] for k in metric_keys}

    for line_id, fold_metrics in per_line_fold_metrics.items():
        line_summary: Dict[str, float] = {"n_folds": float(len(fold_metrics))}
        for k in metric_keys:
            vals = [float(fm[k]) for fm in fold_metrics if k in fm]
            mean, std = _mean_std(vals)
            line_summary[f"{k}_mean"] = mean
            line_summary[f"{k}_std"] = std
            pooled[k].extend(vals)
        per_line[str(line_id)] = line_summary

    out: Dict[str, Any] = {
        "n_lines": len(per_line),
        "n_folds_total": int(sum(len(v) for v in per_line_fold_metrics.values())),
        "per_line": per_line,
    }
    for k in metric_keys:
        mean, std = _mean_std(pooled[k])
        out[f"{k}_mean"] = mean
        out[f"{k}_std"] = std
    return out


def summarize_single_ended_sides(
    per_line_fold_metrics_by_side: Mapping[
        str, Mapping[str, Sequence[Mapping[str, float]]]
    ],
    metric_keys: Sequence[str] = ("mae", "rmse"),
) -> Dict[str, Any]:
    """Combine single-ended sides into mean-side and worst-side aggregates.

    Input: ``{"a": {line: [fold metrics...]}, "b": {line: [...]}}``. Per line,
    take each side's mean-over-folds, then per line compute mean-of-sides and
    worst (max, higher MAE/RMSE = worse) side; aggregate those across lines
    (mean/std).
    """
    sides = list(per_line_fold_metrics_by_side.keys())
    if not sides:
        raise ValueError("no sides provided")
    lines = sorted(
        set().union(*[set(per_line_fold_metrics_by_side[s].keys()) for s in sides])
    )

    mean_side: Dict[str, List[float]] = {k: [] for k in metric_keys}
    worst_side: Dict[str, List[float]] = {k: [] for k in metric_keys}
    per_line: Dict[str, Dict[str, float]] = {}

    for line in lines:
        entry: Dict[str, float] = {}
        for k in metric_keys:
            side_means: List[float] = []
            for s in sides:
                fm = per_line_fold_metrics_by_side[s].get(line, [])
                vals = [float(m[k]) for m in fm if k in m]
                if vals:
                    side_means.append(float(np.mean(vals)))
            if side_means:
                entry[f"{k}_mean_side"] = float(np.mean(side_means))
                entry[f"{k}_worst_side"] = float(np.max(side_means))
                mean_side[k].append(entry[f"{k}_mean_side"])
                worst_side[k].append(entry[f"{k}_worst_side"])
        per_line[str(line)] = entry

    out: Dict[str, Any] = {"n_lines": len(per_line), "per_line": per_line}
    for k in metric_keys:
        m_mean, m_std = _mean_std(mean_side[k])
        w_mean, w_std = _mean_std(worst_side[k])
        out[f"{k}_mean"] = m_mean  # headline = mean-side
        out[f"{k}_std"] = m_std
        out[f"{k}_worst_mean"] = w_mean
        out[f"{k}_worst_std"] = w_std
    return out
