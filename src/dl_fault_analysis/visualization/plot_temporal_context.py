#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# =========================
# Style (ONE place)
# =========================
def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")

    plt.rcParams.update(
        {
            # resolution
            "figure.dpi": 150,
            "savefig.dpi": 300,

            # LaTeX fonts
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],

            # ---- FONT SIZES (bigger) ----
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,

            # ---- MAKE EVERYTHING BOLD ----
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",

            # ---- AXES ----
            "axes.linewidth": 1.2,

            # ---- LINES ----
            "lines.linewidth": 3.0,
            "lines.markersize": 6.0,

            # ---- TICKS ----
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 6,
            "ytick.major.size": 6,

            # ---- GRID ----
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,

            # ---- LEGEND ----
            "legend.frameon": False,
        }
    )


# =========================
# Config
# =========================
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "your_wandb_username_or_team")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "your_wandb_project_name")

OUT_DIR_DEFAULT = os.getenv("OUT_DIR", "reports/plots/wandb")

WINDOW_ORDER_MS = [10, 20, 30, 40, 50]

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

REGRESSION_TARGETS = {"y_fault_location"}


def is_regression(target: str) -> bool:
    return target in REGRESSION_TARGETS


# =========================
# .env loading (no dependency)
# =========================
def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
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


def ensure_wandb_key() -> None:
    if "WANDB_API_KEY" not in os.environ:
        load_dotenv(".env")


def _get_first(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


# =========================
# W&B -> DataFrame
# =========================
def wandb_runs_to_df(entity: str, project: str) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    rows: List[Dict[str, object]] = []
    for r in runs:
        s = dict(r.summary)
        c = dict(r.config)

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
            continue

        row: Dict[str, object] = {}
        row["params.model_raw"] = str(model_name)
        row["params.model"] = MODEL_RENAME_LATEX.get(str(model_name), str(model_name))
        row["params.window_ms"] = int(round(float(window_s) * 1000))
        row["params.fault_target"] = str(target)
        row["params.dataset"] = str(dataset) if dataset is not None else ""

        # FD metrics (binary): prefer non-macro keys, fallback to macro
        row["metrics.f1_fd_mean"] = _get_first(s, ["cv5_mean/f1", "cv5_mean/f1_macro"])

        # Multiclass metrics
        row["metrics.f1_macro_mean"] = s.get("cv5_mean/f1_macro", None)

        # Regression: scale to % line length (so 7.8% => 7.8)
        mae_mean = s.get("cv5_mean/mae", None)
        row["metrics.mae_mean_pct"] = (
            (mae_mean * 100.0) if mae_mean is not None else None
        )

        row["meta.created_at"] = r.created_at
        rows.append(row)

    return pd.DataFrame(rows)


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "params.dataset",
        "params.fault_target",
        "params.window_ms",
        "params.model_raw",
    ]
    df2 = df.copy()
    df2["meta.created_at"] = pd.to_datetime(df2["meta.created_at"], errors="coerce")
    df2 = df2.sort_values("meta.created_at").drop_duplicates(subset=keys, keep="last")
    return df2


# =========================
# Task specs
# =========================
@dataclass(frozen=True)
class TaskPlotSpec:
    target: str
    title: str
    y_label: str
    metric_col: str
    higher_is_better: bool


def get_task_specs() -> Dict[str, TaskPlotSpec]:
    # keys = CLI-friendly short names
    return {
        "fd": TaskPlotSpec(
            "y_fault_present", "Fault detection (FD)", "F1", "metrics.f1_fd_mean", True
        ),
        "fc": TaskPlotSpec(
            "y_fault_class",
            "Fault classification (FC)",
            "Macro-F1",
            "metrics.f1_macro_mean",
            True,
        ),
        "fli": TaskPlotSpec(
            "y_fault_line",
            "Fault line identification (FLI)",
            "Macro-F1",
            "metrics.f1_macro_mean",
            True,
        ),
        "fl": TaskPlotSpec(
            "y_fault_location",
            "Fault localization (FL)",
            r"MAE (\% of line length)",
            "metrics.mae_mean_pct",
            False,
        ),
    }


# =========================
# Aggregate + plot
# =========================
def normalize_model_name(model: str) -> str:
    """
    Accept either raw keys (e.g., 'gru_classifier') or pretty names (e.g., 'GRU').
    We compare against the pretty (renamed) column for robustness.
    """
    # If user passes a raw model key, map it. Otherwise keep as-is.
    return MODEL_RENAME_LATEX.get(model, model)


def aggregate_for_one_model(
    df: pd.DataFrame,
    target: str,
    metric_col: str,
    model_pretty: str,
) -> pd.DataFrame:
    sub = df[
        (df["params.fault_target"] == target) & (df["params.model"] == model_pretty)
    ].copy()
    sub = sub[sub["params.window_ms"].isin(WINDOW_ORDER_MS)]
    if sub.empty:
        return pd.DataFrame(columns=["window_ms", "mean", "std", "n"])

    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub = sub.dropna(subset=[metric_col])

    agg = (
        sub.groupby("params.window_ms")[metric_col]
        .agg(mean="mean", std="std", n="count")
        .reindex(WINDOW_ORDER_MS)
        .reset_index()
        .rename(columns={"params.window_ms": "window_ms"})
    )
    return agg


def _mask_valid(agg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = agg["window_ms"].to_numpy(dtype=float)
    y = agg["mean"].to_numpy(dtype=float)
    s = agg["std"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x_v = x[valid]
    y_v = y[valid]
    s_v = np.nan_to_num(s[valid], nan=0.0)
    return x_v, y_v, s_v


def plot_combined_like_example(
    model_pretty: str,
    series: List[tuple[TaskPlotSpec, pd.DataFrame]],
    out_dir: str,
    out_name: str = "performance_vs_temporal_context.svg",
) -> None:
    """
    Combined figure with two y-axes:
    - Left axis: F1 / Macro-F1 (higher is better)
    - Right axis: MAE (% of line length, lower is better)

    No annotations.
    """

    series = [(spec, agg) for (spec, agg) in series if not agg.empty]
    if len(series) == 0:
        raise SystemExit("[ERROR] No data to plot.")

    fig, ax1 = plt.subplots(figsize=(4.8, 4.0))
    ax2 = ax1.twinx()

    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]

    handles = []
    labels = []

    mae_values = []
    f1_values = []

    for i, (spec, agg) in enumerate(series):
        x = agg["window_ms"].to_numpy(dtype=float)
        y = agg["mean"].to_numpy(dtype=float)
        s = agg["std"].to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        s = np.nan_to_num(s[valid], nan=0.0)

        color = colors[i % len(colors)]

        if spec.higher_is_better:
            # F1 axis (left)
            h = ax1.plot(x, y, color=color)[0]
            ax1.fill_between(x, y - s, y + s, alpha=0.15, color=color)
            f1_values.extend(y)

            labels.append("Classification")

        else:
            # MAE axis (right)
            h = ax2.plot(x, y, color=color)[0]
            ax2.fill_between(x, y - s, y + s, alpha=0.15, color=color)
            mae_values.extend(y)

            labels.append("Localization")

        handles.append(h)

    # Labels
    ax1.set_xlabel(r"Temporal Context (ms)")
    ax1.set_ylabel(r"Performance (F1)")

    ax2.set_ylabel(r"Localization error (MAE)")

    ax1.set_xticks(WINDOW_ORDER_MS)

    # Y limits (tight and clean)
    if len(f1_values) > 0:
        lo = min(f1_values)
        hi = max(f1_values)
        ax1.set_ylim(max(0.0, lo - 0.01), min(1.0, hi + 0.01))
        ax1.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    if len(mae_values) > 0:
        lo = min(mae_values)
        hi = max(mae_values)
        margin = 0.05 * (hi - lo if hi > lo else 1)
        ax2.set_ylim(max(0.0, lo - margin), hi + margin)
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.25, zorder=0)

    # Combined legend
    legend = ax1.legend(handles, labels, frameon=False, loc="best")
    legend.set_zorder(100)

    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved combined plot to: {out_path}")

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    specs = get_task_specs()
    parser = argparse.ArgumentParser(
        description="Plot performance vs. temporal context from W&B."
    )
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    parser.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    parser.add_argument(
        "--model", default="GRU", help="Model to select (e.g., GRU or gru_classifier)."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["fc", "fl"],
        choices=sorted(specs.keys()),
        help="Which tasks to plot. Default: fc fl (matches the example: one saturating, one improving).",
    )
    parser.add_argument("--out_name", default="performance_vs_temporal_context.svg")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_wandb_key()
    set_plot_style()

    df = wandb_runs_to_df(args.entity, args.project)
    if df.empty:
        raise SystemExit(
            "No runs found / could not parse W&B runs. Check entity/project and logging keys."
        )

    df = dedupe_latest(df)

    model_pretty = normalize_model_name(args.model)
    specs = get_task_specs()

    series: List[tuple[TaskPlotSpec, pd.DataFrame]] = []
    for t in args.tasks:
        spec = specs[t]
        agg = aggregate_for_one_model(
            df, spec.target, spec.metric_col, model_pretty=model_pretty
        )
        series.append((spec, agg))

    plot_combined_like_example(
        model_pretty=model_pretty,
        series=series,
        out_dir=args.out_dir,
        out_name=args.out_name,
    )


if __name__ == "__main__":
    main()
