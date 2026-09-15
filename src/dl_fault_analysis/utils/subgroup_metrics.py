from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ----------------------------
# existing helpers (unchanged)
# ----------------------------
def setup_console_logger(
    name: str = __name__, level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        h = logging.StreamHandler()
        h.setLevel(level)
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)

    logger.propagate = False
    return logger


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def binary_metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "support": float(tp + fp + tn + fn),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def subgroup_metrics_binary(
    labels_df: pd.DataFrame,
    row_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
    min_support: int = 1,
    dropna: bool = True,
) -> pd.DataFrame:
    if group_col not in labels_df.columns:
        raise ValueError(
            f"Column '{group_col}' not found. Available: {list(labels_df.columns)}"
        )

    groups = labels_df.iloc[row_idx][group_col].to_numpy()

    if dropna:
        keep = pd.notna(groups)
        groups = groups[keep]
        yt = y_true[keep].astype(np.int8, copy=False)
        yp = y_pred[keep].astype(np.int8, copy=False)
    else:
        yt = y_true.astype(np.int8, copy=False)
        yp = y_pred.astype(np.int8, copy=False)

    if len(groups) == 0:
        return pd.DataFrame()

    codes, uniques = pd.factorize(groups, sort=False)
    n_groups = len(uniques)

    support = np.bincount(codes, minlength=n_groups).astype(np.int64)
    tp = np.bincount(codes, weights=((yt == 1) & (yp == 1)), minlength=n_groups).astype(
        np.int64
    )
    fp = np.bincount(codes, weights=((yt == 0) & (yp == 1)), minlength=n_groups).astype(
        np.int64
    )
    tn = np.bincount(codes, weights=((yt == 0) & (yp == 0)), minlength=n_groups).astype(
        np.int64
    )
    fn = np.bincount(codes, weights=((yt == 1) & (yp == 0)), minlength=n_groups).astype(
        np.int64
    )

    rows: List[Dict[str, Any]] = []
    for i, cat in enumerate(uniques):
        if int(support[i]) < int(min_support):
            continue
        rows.append(
            {
                group_col: cat,
                **binary_metrics_from_counts(
                    int(tp[i]), int(fp[i]), int(tn[i]), int(fn[i])
                ),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["recall", "support"], ascending=[True, False]).reset_index(
        drop=True
    )


# ----------------------------
# NEW: regression subgroup metrics
# ----------------------------
def regression_metrics_from_errors(err: np.ndarray) -> Dict[str, float]:
    """
    err = y_pred - y_true
    """
    if err.size == 0:
        return {
            "support": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "bias": 0.0,
            "median_ae": 0.0,
        }

    ae = np.abs(err)
    return {
        "support": float(err.size),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "bias": float(np.mean(err)),
        "median_ae": float(np.median(ae)),
    }


def subgroup_metrics_regression(
    labels_df: pd.DataFrame,
    row_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
    min_support: int = 1,
    dropna: bool = True,
    eps: float = 1e-6,
    report_percent_points: bool = True,
    report_error_over_y: bool = False,  # diagnostic; can explode near 0
) -> pd.DataFrame:
    if group_col not in labels_df.columns:
        raise ValueError(
            f"Column '{group_col}' not found. Available: {list(labels_df.columns)}"
        )

    groups = labels_df.iloc[row_idx][group_col].to_numpy()

    yt = np.asarray(y_true, dtype=np.float32).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float32).reshape(-1)

    # filter invalid targets/preds as well
    valid = np.isfinite(yt) & np.isfinite(yp)
    if dropna:
        valid &= pd.notna(groups)

    groups = groups[valid]
    yt = yt[valid]
    yp = yp[valid]

    if len(groups) == 0:
        return pd.DataFrame()

    codes, uniques = pd.factorize(groups, sort=False)
    n_groups = len(uniques)

    support = np.bincount(codes, minlength=n_groups).astype(np.int64)

    # errors in relative units (0..1)
    err = (yp - yt).astype(np.float64, copy=False)
    abs_err = np.abs(err)
    sq_err = err * err

    sum_abs = np.bincount(codes, weights=abs_err, minlength=n_groups)
    sum_sq = np.bincount(codes, weights=sq_err, minlength=n_groups)
    sum_err = np.bincount(codes, weights=err, minlength=n_groups)

    # Optional: mean(|err| / |y|)
    if report_error_over_y:
        denom = np.maximum(np.abs(yt).astype(np.float64, copy=False), float(eps))
        abs_err_over_y = abs_err / denom
        sum_abs_over_y = np.bincount(codes, weights=abs_err_over_y, minlength=n_groups)

    rows: List[Dict[str, Any]] = []
    for i, cat in enumerate(uniques):
        n = int(support[i])
        if n < int(min_support):
            continue

        mae_rel = float(sum_abs[i] / n) if n else 0.0
        rmse_rel = float(np.sqrt(sum_sq[i] / n)) if n else 0.0
        bias_rel = float(sum_err[i] / n) if n else 0.0

        idx = codes == i
        median_ae_rel = float(np.median(abs_err[idx])) if np.any(idx) else 0.0

        row: Dict[str, Any] = {
            group_col: cat,
            "support": float(n),
            # relative units (0..1)
            "mae_rel": mae_rel,
            "rmse_rel": rmse_rel,
            "bias_rel": bias_rel,
            "median_ae_rel": median_ae_rel,
        }

        if report_percent_points:
            # percent-points (0..100) for readability
            row.update(
                {
                    "mae_pp": 100.0 * mae_rel,
                    "rmse_pp": 100.0 * rmse_rel,
                    "bias_pp": 100.0 * bias_rel,
                    "median_ae_pp": 100.0 * median_ae_rel,
                }
            )

        if report_error_over_y:
            row["mae_over_y"] = float(sum_abs_over_y[i] / n) if n else 0.0

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Sort by worst performance: MAE in relative units (or percent points if you prefer)
    return out.sort_values(
        ["mae_rel", "support"], ascending=[False, False]
    ).reset_index(drop=True)


def add_true_value_bins_for_regression(
    labels_df: pd.DataFrame,
    row_idx: np.ndarray,
    y_true: np.ndarray,
    out_col: str = "y_true_bin",
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Optional helper: adds a synthetic subgroup column with (rounded) unique y_true values.
    Useful when y_true has finite unique values (e.g., ~5-8 fault locations).
    Returns a *view-like* dataframe copy containing only the new column aligned to labels_df rows.
    """
    yt = np.asarray(y_true, dtype=np.float32)
    bins = np.full((len(labels_df),), np.nan, dtype=object)  # object for mixed/str
    vals = yt.copy()
    vals = np.round(vals, decimals=decimals)
    # only fill provided indices
    for j, ridx in enumerate(row_idx):
        v = vals[j]
        if np.isfinite(v):
            bins[int(ridx)] = float(v)
    df2 = labels_df.copy()
    df2[out_col] = bins
    return df2


def _sanitize_filename(s: str) -> str:
    # keep simple + predictable
    return "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in str(s))


def run_subgroup_analysis(
    labels_df: pd.DataFrame,
    preds: Dict[str, Any],
    group_cols: Sequence[str],
    out_dir: str,
    task_type: str = "binary",  # "binary" | "regression"
    min_support: int = 5,
    top_k: int = 10,
    logger: logging.Logger | None = None,
    regression_add_true_bins: bool = True,
    regression_true_bins_decimals: int = 3,
) -> Dict[str, pd.DataFrame]:
    logger = logger or setup_console_logger(__name__, logging.INFO)
    os.makedirs(out_dir, exist_ok=True)

    if task_type not in {"binary", "regression"}:
        raise ValueError(f"Unknown task_type='{task_type}'")

    row_idx = np.asarray(preds["row_idx"], dtype=np.int64).reshape(-1)
    y_true = np.asarray(preds["y_true"]).reshape(-1)
    y_pred = np.asarray(preds["y_pred"]).reshape(-1)

    if len(row_idx) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: row_idx={len(row_idx)} y_true={len(y_true)} y_pred={len(y_pred)}"
        )

    logger.info(
        "Subgroup analysis | task_type=%s | n=%d | min_support=%d | out_dir=%s",
        task_type,
        len(row_idx),
        min_support,
        out_dir,
    )

    # Optionally add y_true-based subgroup bins for regression
    labels_for_groups = labels_df
    extra_cols: List[str] = []
    if task_type == "regression" and regression_add_true_bins:
        labels_for_groups = add_true_value_bins_for_regression(
            labels_df=labels_df,
            row_idx=row_idx,
            y_true=y_true,
            out_col="y_true_bin",
            decimals=regression_true_bins_decimals,
        )
        extra_cols = ["y_true_bin"]

    results: Dict[str, pd.DataFrame] = {}
    for col in list(group_cols) + extra_cols:
        if col not in labels_for_groups.columns:
            logger.warning("Skipping '%s' (missing column).", col)
            continue

        if task_type == "binary":
            df = subgroup_metrics_binary(
                labels_df=labels_for_groups,
                row_idx=row_idx,
                y_true=y_true,
                y_pred=y_pred,
                group_col=col,
                min_support=min_support,
                dropna=True,
            )
            if df.empty:
                logger.warning(
                    "No subgroup rows for '%s' after min_support=%d", col, min_support
                )
                continue

            # subgroup_metrics_binary sorts with *lowest recall first* (worst-first)
            sort_label = "f1"
            cols_to_print = [
                col,
                "support",
                "recall",
                "precision",
                "f1",
                "accuracy",
                "tp",
                "fp",
                "fn",
                "tn",
            ]
            worst_df = df.head(top_k)

        else:
            df = subgroup_metrics_regression(
                labels_df=labels_for_groups,
                row_idx=row_idx,
                y_true=y_true,
                y_pred=y_pred,
                group_col=col,
                min_support=min_support,
                dropna=True,
                report_percent_points=True,  # adds mae_pp, rmse_pp, ...
                report_error_over_y=False,
            )
            if df.empty:
                logger.warning(
                    "No subgroup rows for '%s' after min_support=%d", col, min_support
                )
                continue

            if "mae_pp" in df.columns:
                sort_label = "mae_pp"
                cols_to_print = [
                    col,
                    "support",
                    "mae_pp",
                    "rmse_pp",
                    "bias_pp",
                    "median_ae_pp",
                    "mae_rel",
                    "rmse_rel",
                ]
            else:
                sort_label = "mae_rel"
                cols_to_print = [
                    col,
                    "support",
                    "mae_rel",
                    "rmse_rel",
                    "bias_rel",
                    "median_ae_rel",
                ]

            # Ensure the "Worst ... by sort_label" print is correct regardless of how df is sorted
            if sort_label in df.columns:
                worst_df = df.sort_values(
                    [sort_label, "support"], ascending=[False, False]
                ).head(top_k)
            else:
                worst_df = df.head(top_k)

        csv_path = os.path.join(out_dir, f"metrics_by_{_sanitize_filename(col)}.csv")
        df.to_csv(csv_path, index=False)

        logger.info("Wrote %s | groups=%d", csv_path, len(df))
        logger.info(
            "Worst %d '%s' groups by %s:\n%s",
            min(top_k, len(df)),
            col,
            sort_label,
            worst_df[cols_to_print].to_string(index=False),
        )

        results[col] = df

    return results


def maybe_run_subgroup_analysis(
    *,
    task_type: str,
    labels_df: pd.DataFrame,
    valid_row_idx: np.ndarray,
    idx_test: np.ndarray,
    y_true: Optional[np.ndarray],
    y_pred: Optional[np.ndarray],
    out_dir: str,
    group_cols: Sequence[str],
    logger,
    min_support: int = 5,
    regression_add_true_bins: bool = True,
    regression_true_bins_decimals: int = 3,
) -> None:
    """
    Thin wrapper around run_subgroup_analysis that:
    - computes row_idx in original labels_df
    - dispatches based on task_type
    """
    if y_true is None or y_pred is None:
        logger.info("Skipping subgroup analysis: missing predictions.")
        return

    row_idx_test_global = valid_row_idx[np.asarray(idx_test, dtype=np.int64)]

    if task_type == "binary":
        run_subgroup_analysis(
            labels_df=labels_df,
            preds={
                "row_idx": row_idx_test_global,
                "y_true": y_true,
                "y_pred": y_pred,
            },
            group_cols=list(group_cols),
            out_dir=str(out_dir),
            task_type="binary",
            min_support=int(min_support),
            logger=logger,
        )
        return

    if task_type == "regression":
        run_subgroup_analysis(
            labels_df=labels_df,
            preds={
                "row_idx": row_idx_test_global,
                "y_true": y_true,
                "y_pred": y_pred,
            },
            group_cols=list(group_cols),
            out_dir=str(out_dir),
            task_type="regression",
            min_support=int(min_support),
            logger=logger,
            regression_add_true_bins=bool(regression_add_true_bins),
            regression_true_bins_decimals=int(regression_true_bins_decimals),
        )
        return

    logger.info("Skipping subgroup analysis for task_type=%s.", task_type)
