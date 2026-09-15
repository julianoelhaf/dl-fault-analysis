#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate LaTeX tables from Weights & Biases runs.

Main-text tables:
- Classification (FD/FC/FLI): F1 + PR-AUC (+ optional ROC-AUC)
- Regression (FL): MAE + RMSE (appendix: + R^2 + P90)

Appendix tables:
- Classification: F1, P, R, PR-AUC, ROC-AUC
- Regression: MAE, RMSE, R^2, P90

Key handling:
- Recommended: store WANDB_API_KEY in a local .env file (NOT tracked) and load it.
  Example .env (do NOT commit):
      WANDB_API_KEY=wandb_v1_...
      WANDB_ENTITY=your_wandb_username_or_team
      WANDB_PROJECT=your_wandb_project_name

Offline mode (no W&B account needed): set WANDB_EXPORT_DIR to a local export
directory (configs.jsonl + summaries.jsonl + runs.csv) to regenerate tables
from an archived export instead of the live API. This repository's own
archived export lives at results/paper/main/raw/ -- see
results/paper/main/PROVENANCE.md. Example:
    WANDB_EXPORT_DIR=results/paper/main/raw OUT_DIR=/tmp/regenerated \
      python src/dl_fault_analysis/evaluation/make_tables_wandb.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
import wandb


# =========================
# Config
# =========================

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "your_wandb_username_or_team")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "your_wandb_project_name")
OUT_DIR = os.getenv("OUT_DIR", "reports/tables/wandb")
# When set, tables are regenerated from a local offline export (see
# wandb_runs_to_df_from_export) instead of hitting the live W&B API -- no
# WANDB_API_KEY needed. results/paper/main/raw/ is this repository's own
# archived export; see results/paper/main/PROVENANCE.md.
WANDB_EXPORT_DIR = os.getenv("WANDB_EXPORT_DIR")

# Window order used in paper
WINDOW_ORDER_MS = [10, 20, 30, 40, 50]

# Targets used in paper (map from your hydra labels)
FAULT_TARGETS = [
    "y_fault_present",  # FD (binary)
    "y_fault_class",  # FC (multiclass)
    "y_fault_line",  # FLI (multiclass)
    "y_fault_location",  # FL (regression)
]

# Model renaming for LaTeX
MODEL_RENAME_LATEX = {
    "cnn_classifier": "CNN",
    "cnn_regressor": "CNN",
    "dilated_cnn_classifier": "Dilated CNN",
    "dilated_cnn_regressor": "Dilated CNN",
    "cnn_lstm_classifier": "CNN--LSTM",
    "cnn_lstm_regressor": "CNN--LSTM",
    "rnn_classifier": "RNN",
    "rnn_regressor": "RNN",
    "lstm_classifier": "LSTM",
    "lstm_regressor": "LSTM",
    "gru_classifier": "GRU",
    "gru_regressor": "GRU",
    "tcn_classifier": "TCN",
    "tcn_regressor": "TCN",
    "tft_classifier": "TFT",
    "tft_regressor": "TFT",
    "inceptiontime_classifier": "InceptionTime",
    "inceptiontime_regressor": "InceptionTime",
}

# ----------------- Task typing -----------------
CLASSIFICATION_TARGETS = {"y_fault_present", "y_fault_class", "y_fault_line"}
REGRESSION_TARGETS = {"y_fault_location"}


def is_regression(target: str) -> bool:
    return target in REGRESSION_TARGETS


# =========================
# .env loading (no extra dependency)
# =========================


def load_dotenv(path: str = ".env") -> None:
    """
    Minimal .env loader (no python-dotenv dependency).
    Loads KEY=VALUE lines into os.environ if KEY not already set.
    Supports:
      - blank lines
      - comments starting with #
      - quoted values "..." or '...'
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except OSError:
        # If .env can't be read, just continue; W&B might still work via other env vars.
        return


def ensure_wandb_key() -> None:
    """
    Ensure WANDB_API_KEY is available via env.
    Priority:
      1) already set in environment
      2) loaded from .env
    """
    load_dotenv(".env")
    # If still not set, W&B may still work (e.g., already logged-in locally),
    # but most headless environments need WANDB_API_KEY.
    # We do not print the key.
    return


# =========================
# Metric specs
# =========================


@dataclass(frozen=True)
class MetricSpec:
    key: str
    mean_col: str
    std_col: Optional[str]
    higher_is_better: bool
    fmt: str
    cap_at_one: bool = False


def _cap_below_one(m: float, fmt: str) -> float:
    decimals = None
    if fmt.startswith(".") and fmt.endswith("f"):
        try:
            decimals = int(fmt[1:-1])
        except ValueError:
            decimals = None
    if decimals is None:
        return 0.999 if m >= 1.0 else m
    cap = 1.0 - (10 ** (-decimals))
    return min(m, cap)


def _get_first(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


# =========================
# W&B -> DataFrame
# =========================


def _row_from_config_summary(
    c: dict, s: dict, run_id: str, created_at: object
) -> Optional[Dict[str, object]]:
    """Build one output row from a single run's (config, summary) dicts.

    Shared by the live-API loader (`wandb_runs_to_df`) and the offline-export
    loader (`wandb_runs_to_df_from_export`) so both paths apply identical
    field extraction -- there is exactly one place that knows how to read a
    W&B run's config/summary shape.
    """
    row: Dict[str, object] = {}

    # ---- params (from config) ----
    def cfg(*keys, default=None):
        d = c
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    model_name = cfg("model", "model_name", default=c.get("model"))
    window_s = cfg(
        "window_extraction", "window_length", default=c.get("window_length_s")
    )
    target = cfg("training", "target_label", default=c.get("target"))
    dataset = c.get("dataset", c.get("topology"))

    if model_name is None or window_s is None or target is None:
        return None

    row["params.model"] = str(model_name)
    row["params.window_ms"] = int(round(float(window_s) * 1000))
    row["params.fault_target"] = str(target)
    row["params.dataset"] = str(dataset) if dataset is not None else ""

    # Optional: params / runtime for trade-off
    row["params.num_params"] = s.get("model/num_params", None)
    row["metrics.runtime_ms_per_window"] = s.get(
        "runtime/infer_ms_per_window", None
    ) or s.get("fold0/runtime_ms_per_window", None)

    # ---- metrics (from summary) ----
    # FD (binary): prefer non-macro keys if present.
    row["metrics.f1_fd_mean"] = _get_first(s, ["cv5_mean/f1", "cv5_mean/f1_macro"])
    row["metrics.f1_fd_std"] = _get_first(s, ["cv5_std/f1", "cv5_std/f1_macro"])

    row["metrics.pr_auc_fd_mean"] = _get_first(
        s, ["cv5_mean/pr_auc", "cv5_mean/pr_auc_macro"]
    )
    row["metrics.pr_auc_fd_std"] = _get_first(
        s, ["cv5_std/pr_auc", "cv5_std/pr_auc_macro"]
    )

    row["metrics.roc_auc_fd_mean"] = _get_first(
        s, ["cv5_mean/roc_auc", "cv5_mean/roc_auc_ovr"]
    )
    row["metrics.roc_auc_fd_std"] = _get_first(
        s, ["cv5_std/roc_auc", "cv5_std/roc_auc_ovr"]
    )

    # Accuracy (optional)
    row["metrics.accuracy_mean"] = s.get("cv5_mean/accuracy", None)
    row["metrics.accuracy_std"] = s.get("cv5_std/accuracy", None)

    # Multiclass defaults (macro / ovr)
    row["metrics.f1_macro_mean"] = s.get("cv5_mean/f1_macro", None)
    row["metrics.f1_macro_std"] = s.get("cv5_std/f1_macro", None)

    row["metrics.pr_auc_macro_mean"] = s.get("cv5_mean/pr_auc_macro", None)
    row["metrics.pr_auc_macro_std"] = s.get("cv5_std/pr_auc_macro", None)

    row["metrics.roc_auc_ovr_mean"] = s.get("cv5_mean/roc_auc_ovr", None)
    row["metrics.roc_auc_ovr_std"] = s.get("cv5_std/roc_auc_ovr", None)

    # Optional classification breakdown
    row["metrics.precision_macro_mean"] = s.get("cv5_mean/precision_macro", None)
    row["metrics.precision_macro_std"] = s.get("cv5_std/precision_macro", None)
    row["metrics.recall_macro_mean"] = s.get("cv5_mean/recall_macro", None)
    row["metrics.recall_macro_std"] = s.get("cv5_std/recall_macro", None)

    # Regression: scale from [0,1] to percent of line length
    mae_mean = s.get("cv5_mean/mae", None)
    mae_std = s.get("cv5_std/mae", None)
    row["metrics.mae_mean"] = mae_mean * 100 if mae_mean is not None else None
    row["metrics.mae_std"] = mae_std * 100 if mae_std is not None else None

    rmse_mean = s.get("cv5_mean/rmse", None)
    rmse_std = s.get("cv5_std/rmse", None)
    row["metrics.rmse_mean"] = rmse_mean * 100 if rmse_mean is not None else None
    row["metrics.rmse_std"] = rmse_std * 100 if rmse_std is not None else None

    row["metrics.r2_mean"] = s.get("cv5_mean/r2", None)
    row["metrics.r2_std"] = s.get("cv5_std/r2", None)

    p90_ae = s.get("cv5_mean/p90_ae", None)
    row["metrics.p90_ae"] = p90_ae * 100 if p90_ae is not None else None

    # meta for dedupe
    row["meta.run_id"] = run_id
    row["meta.created_at"] = created_at

    return row


def wandb_runs_to_df(entity: str, project: str) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    rows: List[Dict[str, object]] = []
    for r in runs:
        row = _row_from_config_summary(dict(r.config), dict(r.summary), r.id, r.created_at)
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


def wandb_runs_to_df_from_export(export_dir: str) -> pd.DataFrame:
    """Build the same DataFrame as `wandb_runs_to_df`, from a local offline
    export directory instead of the live W&B API.

    Expects `configs.jsonl` and `summaries.jsonl` (one JSON object per line,
    each with a "run_id" key plus "config"/"summary" respectively) and,
    optionally, `runs.csv` (with "run_id" and "created_at" columns, used only
    for `dedupe_latest`). This is the format produced by exporting a W&B
    project's runs via the API and dumping each run's `.config`/`.summary`/
    metadata -- see results/paper/main/PROVENANCE.md for how this
    repository's own archived export was produced.
    """
    import csv as csv_module
    import json

    configs: Dict[str, dict] = {}
    with open(os.path.join(export_dir, "configs.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            configs[d["run_id"]] = d["config"]

    summaries: Dict[str, dict] = {}
    with open(os.path.join(export_dir, "summaries.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            summaries[d["run_id"]] = d["summary"]

    created_at: Dict[str, object] = {}
    runs_csv_path = os.path.join(export_dir, "runs.csv")
    if os.path.exists(runs_csv_path):
        with open(runs_csv_path, encoding="utf-8", newline="") as f:
            for r in csv_module.DictReader(f):
                created_at[r["run_id"]] = r.get("created_at")

    rows: List[Dict[str, object]] = []
    for run_id, c in configs.items():
        s = summaries.get(run_id, {})
        row = _row_from_config_summary(c, s, run_id, created_at.get(run_id))
        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest run per (dataset, target, window_ms, model).

    A run whose `created_at` is missing or unparseable cannot be ordered, so it
    is treated as the *oldest*: any run with a real timestamp beats it. This is
    deliberate. `pd.to_datetime(..., errors="coerce")` turns such values into
    NaT, and `sort_values` places NaT last by default, so the naive
    `sort_values(...).drop_duplicates(keep="last")` would preferentially select
    exactly the runs we know least about -- including, potentially, a crashed
    attempt over its successful retry. `na_position="first"` inverts that.

    Ties (equal or absent timestamps) are broken by run id rather than by input
    order, so selection is reproducible and does not depend on how the runs
    happened to be enumerated by the API or ordered in an offline export.
    """
    keys = ["params.dataset", "params.fault_target", "params.window_ms", "params.model"]
    df2 = df.copy()
    df2["meta.created_at"] = pd.to_datetime(df2["meta.created_at"], errors="coerce")

    n_missing = int(df2["meta.created_at"].isna().sum())
    if n_missing:
        print(
            f"[WARN] {n_missing} of {len(df2)} runs have no usable created_at "
            "timestamp; they are treated as oldest and will only be selected if "
            "no timestamped run exists for the same combination. For an offline "
            "export, this usually means runs.csv is missing or incomplete."
        )

    df2 = df2.sort_values(
        ["meta.created_at", "meta.run_id"], na_position="first", kind="mergesort"
    ).drop_duplicates(subset=keys, keep="last")
    return df2


# =========================
# Rendering
# =========================


def latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def format_cell(mean: object, std: object, spec: MetricSpec) -> str:
    if mean is None or (isinstance(mean, float) and pd.isna(mean)):
        return "--"
    m = float(mean)
    if spec.cap_at_one:
        m = _cap_below_one(m, spec.fmt)
    m_txt = format(m, spec.fmt)

    if std is None or (isinstance(std, float) and pd.isna(std)):
        return m_txt
    s = float(std)
    s_txt = format(s, spec.fmt)
    return f"\\mstd{{{m_txt}}}{{{s_txt}}}"


def format_cell_plain(mean: object, std: object, spec: MetricSpec) -> str:
    """Same value formatting as `format_cell`, without LaTeX markup -- for CSV."""
    if mean is None or (isinstance(mean, float) and pd.isna(mean)):
        return ""
    m = float(mean)
    if spec.cap_at_one:
        m = _cap_below_one(m, spec.fmt)
    m_txt = format(m, spec.fmt)
    if std is None or (isinstance(std, float) and pd.isna(std)):
        return m_txt
    s_txt = format(float(std), spec.fmt)
    return f"{m_txt} ± {s_txt}"


def wide_to_csv(mean_wide: pd.DataFrame, std_wide: pd.DataFrame, specs: List[MetricSpec]) -> pd.DataFrame:
    """Flatten a prepare_wide() mean/std pair into one "model,<window>_<metric>,..." CSV frame."""
    if mean_wide.empty:
        return pd.DataFrame()
    ms_by_key = {ms.key: ms for ms in specs}
    out = pd.DataFrame(index=mean_wide.index)
    for w, k in mean_wide.columns:
        ms = ms_by_key[k]
        col_name = f"{w}ms_{k}"
        out[col_name] = [
            format_cell_plain(mean_wide.loc[model, (w, k)], std_wide.loc[model, (w, k)], ms)
            for model in mean_wide.index
        ]
    out.index.name = "model"
    return out.reset_index()


def best_mask(series: pd.Series, higher_is_better: bool) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.isna().all():
        return pd.Series([False] * len(series), index=series.index)
    best = vals.max() if higher_is_better else vals.min()
    tol = 1e-12
    return (vals - best).abs() <= tol


def prepare_wide(
    df: pd.DataFrame, target: str, specs: List[MetricSpec]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[df["params.fault_target"] == target].copy()
    sub = sub[sub["params.window_ms"].isin(WINDOW_ORDER_MS)]
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    mean_parts = []
    std_parts = []
    for ms in specs:
        m = sub.pivot(
            index="params.model", columns="params.window_ms", values=ms.mean_col
        ).reindex(columns=WINDOW_ORDER_MS)
        s = (
            sub.pivot(
                index="params.model", columns="params.window_ms", values=ms.std_col
            ).reindex(columns=WINDOW_ORDER_MS)
            if ms.std_col
            else None
        )

        m.columns = pd.MultiIndex.from_product(
            [m.columns.tolist(), [ms.key]], names=["window_ms", "metric"]
        )
        if s is None:
            s = m.copy() * float("nan")
        else:
            s.columns = pd.MultiIndex.from_product(
                [s.columns.tolist(), [ms.key]], names=["window_ms", "metric"]
            )

        mean_parts.append(m)
        std_parts.append(s)

    mean_wide = pd.concat(mean_parts, axis=1)
    std_wide = pd.concat(std_parts, axis=1)

    mean_wide.index = [MODEL_RENAME_LATEX.get(x, x) for x in mean_wide.index]
    std_wide.index = mean_wide.index

    primary = specs[0]
    primary_cols = [
        (w, primary.key)
        for w in WINDOW_ORDER_MS
        if (w, primary.key) in mean_wide.columns
    ]
    score = mean_wide[primary_cols].mean(axis=1)
    mean_wide = mean_wide.loc[
        score.sort_values(ascending=not primary.higher_is_better).index
    ]
    std_wide = std_wide.reindex(index=mean_wide.index)

    return mean_wide, std_wide


def render_transposed(
    mean_wide: pd.DataFrame,
    std_wide: pd.DataFrame,
    specs: List[MetricSpec],
    caption: str,
    label: str,
) -> str:
    if mean_wide.empty:
        return ""

    metrics = [ms.key for ms in specs]
    n_sub = len(metrics)

    ordered_cols = []
    for w in WINDOW_ORDER_MS:
        for ms in specs:
            col = (w, ms.key)
            if col in mean_wide.columns:
                ordered_cols.append(col)
    mean_wide = mean_wide.reindex(
        columns=pd.MultiIndex.from_tuples(ordered_cols, names=["window_ms", "metric"])
    )
    std_wide = std_wide.reindex(columns=mean_wide.columns)

    ms_by_key = {ms.key: ms for ms in specs}
    best_masks = {
        (w, k): best_mask(mean_wide[(w, k)], ms_by_key[k].higher_is_better)
        for (w, k) in mean_wide.columns
    }

    col_spec = "l" + ("c" * (len(WINDOW_ORDER_MS) * n_sub))

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    h1 = ["\\textbf{Model}"] + [
        f"\\multicolumn{{{n_sub}}}{{c}}{{\\textbf{{{w} ms}}}}" for w in WINDOW_ORDER_MS
    ]
    lines.append(" & ".join(h1) + " \\\\")

    start = 2
    cm = []
    for _ in WINDOW_ORDER_MS:
        end = start + n_sub - 1
        cm.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append("".join(cm))

    h2 = [" "] + [
        f"\\textbf{{{latex_escape(k)}}}" for _w in WINDOW_ORDER_MS for k in metrics
    ]
    lines.append(" & ".join(h2) + " \\\\")
    lines.append("\\midrule")

    for model in mean_wide.index:
        row = [latex_escape(model)]
        for w, k in mean_wide.columns:
            ms = ms_by_key[k]
            cell = format_cell(
                mean_wide.loc[model, (w, k)], std_wide.loc[model, (w, k)], ms
            )
            if cell != "--" and bool(best_masks[(w, k)].get(model, False)):
                cell = f"\\textbf{{{cell}}}"
            row.append(cell)
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# =========================
# Table definitions
# =========================


def specs_main(target: str) -> List[MetricSpec]:
    if not is_regression(target):
        if target == "y_fault_present":
            f1_mean_col, f1_std_col = "metrics.f1_fd_mean", "metrics.f1_fd_std"
            pr_mean_col, pr_std_col = "metrics.pr_auc_fd_mean", "metrics.pr_auc_fd_std"
        else:
            f1_mean_col, f1_std_col = "metrics.f1_macro_mean", "metrics.f1_macro_std"
            pr_mean_col, pr_std_col = (
                "metrics.pr_auc_macro_mean",
                "metrics.pr_auc_macro_std",
            )

        return [
            MetricSpec("F1", f1_mean_col, f1_std_col, True, ".3f", cap_at_one=True),
            MetricSpec("PR-AUC", pr_mean_col, pr_std_col, True, ".3f", cap_at_one=True),
        ]

    return [
        MetricSpec("MAE", "metrics.mae_mean", "metrics.mae_std", False, ".2f"),
        MetricSpec("RMSE", "metrics.rmse_mean", "metrics.rmse_std", False, ".2f"),
    ]


def specs_appendix(target: str) -> List[MetricSpec]:
    if not is_regression(target):
        if target == "y_fault_present":
            f1_mean_col, f1_std_col = "metrics.f1_fd_mean", "metrics.f1_fd_std"
            pr_mean_col, pr_std_col = "metrics.pr_auc_fd_mean", "metrics.pr_auc_fd_std"
            roc_mean_col, roc_std_col = (
                "metrics.roc_auc_fd_mean",
                "metrics.roc_auc_fd_std",
            )
            accuracy_mean_col, accuracy_std_col = (
                "metrics.accuracy_mean",
                "metrics.accuracy_std",
            )
        else:
            f1_mean_col, f1_std_col = "metrics.f1_macro_mean", "metrics.f1_macro_std"
            accuracy_mean_col, accuracy_std_col = (
                "metrics.accuracy_mean",
                "metrics.accuracy_std",
            )
            roc_mean_col, roc_std_col = (
                "metrics.roc_auc_ovr_mean",
                "metrics.roc_auc_ovr_std",
            )
            pr_mean_col, pr_std_col = (
                "metrics.pr_auc_macro_mean",
                "metrics.pr_auc_macro_std",
            )

        return [
            # NOTE: F1 and PR-AUC are intentionally omitted from the appendix tables
            # to avoid duplicating the main-text metrics; the appendix focuses on
            # Accuracy and ROC-AUC as summary classification metrics.
            # MetricSpec("F1", f1_mean_col, f1_std_col, True, ".3f", cap_at_one=True),
            # MetricSpec("PR-AUC", pr_mean_col, pr_std_col, True, ".3f", cap_at_one=True),
            MetricSpec(
                "Accuracy",
                accuracy_mean_col,
                accuracy_std_col,
                True,
                ".3f",
                cap_at_one=True,
            ),
            MetricSpec(
                "ROC-AUC", roc_mean_col, roc_std_col, True, ".3f", cap_at_one=True
            ),
        ]

    return [
        MetricSpec("MAE", "metrics.mae_mean", "metrics.mae_std", False, ".3f"),
        MetricSpec("RMSE", "metrics.rmse_mean", "metrics.rmse_std", False, ".3f"),
        MetricSpec(
            "$R^2$", "metrics.r2_mean", "metrics.r2_std", True, ".3f", cap_at_one=True
        ),
        MetricSpec("P90", "metrics.p90_ae", None, False, ".3f"),
    ]


def caption_label(target: str, which: str) -> Tuple[str, str]:
    if which == "main":
        if target == "y_fault_present":
            return (
                "Fault detection across decision windows (F1 and PR-AUC; mean$\\pm$std over 5 folds).",
                "tab:fd_main",
            )
        if target == "y_fault_class":
            return (
                "Fault classification across decision windows (macro-F1 and macro-PR-AUC; mean$\\pm$std over 5 folds).",
                "tab:fc_main",
            )
        if target == "y_fault_line":
            return (
                "Fault line identification across decision windows (macro-F1 and macro-PR-AUC; mean$\\pm$std over 5 folds).",
                "tab:fli_main",
            )
        if target == "y_fault_location":
            return (
                "Fault localization across decision windows (MAE and RMSE in \\% of line length; mean$\\pm$std over 5 folds).",
                "tab:fl_main",
            )
    else:
        if target == "y_fault_present":
            return (
                "Fault detection across decision windows (appendix metrics).",
                "tab:fd_app",
            )
        if target == "y_fault_class":
            return (
                "Fault classification across decision windows (appendix metrics).",
                "tab:fc_app",
            )
        if target == "y_fault_line":
            return (
                "Fault line identification across decision windows (appendix metrics).",
                "tab:fli_app",
            )
        if target == "y_fault_location":
            return (
                "Fault localization across decision windows (appendix metrics, MAE, RMSE, and P90 in \\% of line length).",
                "tab:fl_app",
            )
    return (latex_escape(target), f"tab:{target}_{which}")


# =========================
# Main
# =========================


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if WANDB_EXPORT_DIR:
        print(f"[INFO] loading offline export from {WANDB_EXPORT_DIR}")
        df = wandb_runs_to_df_from_export(WANDB_EXPORT_DIR)
    else:
        ensure_wandb_key()
        df = wandb_runs_to_df(WANDB_ENTITY, WANDB_PROJECT)
    if df.empty:
        raise SystemExit(
            "No runs found / could not parse W&B runs. Check entity/project and logging keys "
            "(or WANDB_EXPORT_DIR if using an offline export)."
        )

    df = dedupe_latest(df)
    print(f"[INFO] loaded {len(df)} runs after deduplication.")
    print(
        f"[INFO] fault targets in data: {df['params.fault_target'].unique().tolist()}"
    )
    print(f"[INFO] window_ms in data: {df['params.window_ms'].unique().tolist()}")
    print(f"[INFO] models in data: {df['params.model'].unique().tolist()}")
    print(f"[INFO] sample of data:\n{df.head().to_string(index=False)}")

    # Main-text tables
    for target in FAULT_TARGETS:
        specs = specs_main(target)
        mean_wide, std_wide = prepare_wide(df, target, specs)
        cap, lab = caption_label(target, "main")
        tex = render_transposed(mean_wide, std_wide, specs, cap, lab)
        outpath = os.path.join(OUT_DIR, f"table_{target}_main.tex")
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(tex + "\n")
        print(f"[INFO] wrote {outpath}")

        csv_df = wide_to_csv(mean_wide, std_wide, specs)
        csv_path = os.path.join(OUT_DIR, f"table_{target}_main.csv")
        csv_df.to_csv(csv_path, index=False)
        print(f"[INFO] wrote {csv_path}")

    # Appendix tables
    for target in FAULT_TARGETS:
        specs = specs_appendix(target)
        mean_wide, std_wide = prepare_wide(df, target, specs)
        cap, lab = caption_label(target, "app")
        tex = render_transposed(mean_wide, std_wide, specs, cap, lab)
        outpath = os.path.join(OUT_DIR, f"table_{target}_appendix.tex")
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(tex + "\n")
        print(f"[INFO] wrote {outpath}")

        csv_df = wide_to_csv(mean_wide, std_wide, specs)
        csv_path = os.path.join(OUT_DIR, f"table_{target}_appendix.csv")
        csv_df.to_csv(csv_path, index=False)
        print(f"[INFO] wrote {csv_path}")


if __name__ == "__main__":
    main()
