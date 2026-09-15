#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


# =========================
# Style
# =========================
def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "font.size": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "grid.linewidth": 0.6,
        }
    )


# =========================
# Config
# =========================
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "your_wandb_username_or_team")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "your_wandb_project_name")

OUT_DIR = os.getenv("OUT_DIR", "reports/plots/wandb")
os.makedirs(OUT_DIR, exist_ok=True)

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
        row["params.model"] = str(model_name)
        row["params.window_ms"] = int(round(float(window_s) * 1000))
        row["params.fault_target"] = str(target)
        row["params.dataset"] = str(dataset) if dataset is not None else ""

        # FD (binary): prefer non-macro keys
        row["metrics.f1_fd_mean"] = _get_first(s, ["cv5_mean/f1", "cv5_mean/f1_macro"])

        # FC/FLI: macro-F1
        row["metrics.f1_macro_mean"] = s.get("cv5_mean/f1_macro", None)

        # FL: MAE in % of line length
        mae_mean = s.get("cv5_mean/mae", None)
        row["metrics.mae_mean"] = mae_mean * 100 if mae_mean is not None else None

        row["meta.created_at"] = r.created_at
        rows.append(row)

    return pd.DataFrame(rows)


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["params.dataset", "params.fault_target", "params.window_ms", "params.model"]
    df2 = df.copy()
    df2["meta.created_at"] = pd.to_datetime(df2["meta.created_at"], errors="coerce")
    df2 = df2.sort_values("meta.created_at").drop_duplicates(subset=keys, keep="last")
    return df2


# =========================
# Specs
# =========================
@dataclass(frozen=True)
class TaskPlotSpec:
    target: str
    title: str
    y_label: str
    metric_col: str
    higher_is_better: bool


def get_task_specs() -> List[TaskPlotSpec]:
    return [
        TaskPlotSpec(
            "y_fault_present", "Fault detection (FD)", "F1", "metrics.f1_fd_mean", True
        ),
        TaskPlotSpec(
            "y_fault_class",
            "Fault classification (FC)",
            "Macro-F1",
            "metrics.f1_macro_mean",
            True,
        ),
        TaskPlotSpec(
            "y_fault_line",
            "Fault line identification (FLI)",
            "Macro-F1",
            "metrics.f1_macro_mean",
            True,
        ),
        TaskPlotSpec(
            "y_fault_location",
            "Fault localization (FL)",
            r"MAE (\% of line length)",
            "metrics.mae_mean",
            False,
        ),
    ]


# =========================
# Aggregate
# =========================
def aggregate_over_models(
    df: pd.DataFrame, target: str, metric_col: str
) -> pd.DataFrame:
    sub = df[df["params.fault_target"] == target].copy()
    sub = sub[sub["params.window_ms"].isin(WINDOW_ORDER_MS)]
    if sub.empty:
        return pd.DataFrame(columns=["window_ms", "mean", "std", "n"])

    sub["params.model"] = sub["params.model"].map(
        lambda x: MODEL_RENAME_LATEX.get(x, x)
    )
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


# =========================
# Plot helper
# =========================
def plot_on_ax(ax: plt.Axes, spec: TaskPlotSpec, agg: pd.DataFrame) -> None:
    x = agg["window_ms"].to_numpy(dtype=float)
    y = agg["mean"].to_numpy(dtype=float)
    s = agg["std"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    s = np.nan_to_num(s[valid], nan=0.0)

    ax.plot(x, y, marker="o", label=r"Mean")
    ax.fill_between(x, y - s, y + s, alpha=0.15, label=r"$\pm 1\sigma$ (models)")

    ax.set_title(spec.title)
    ax.set_xlabel("Window length (ms)")
    ax.set_ylabel(spec.y_label)
    ax.set_xticks(WINDOW_ORDER_MS)
    ax.grid(True, alpha=0.25)

    # --- axis policy ---
    if spec.higher_is_better:
        lo = float(np.min(y - s))
        hi = float(np.max(y + s))
        margin = 0.001  # tighter for near-1.0 curves
        ax.set_ylim(max(0.0, lo - margin), min(1.0, hi + margin))

        # FD in particular needs more decimals to avoid repeated "1.00"
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    else:
        ymax = float(np.max(y + s))
        ymin = float(np.min(y - s))
        ax.set_ylim(ymin - 0.0025 * abs(ymin), ymax + 0.0025 * abs(ymax))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


# =========================
# Main
# =========================
def main() -> None:
    ensure_wandb_key()
    set_plot_style()

    df = wandb_runs_to_df(WANDB_ENTITY, WANDB_PROJECT)
    if df.empty:
        raise SystemExit(
            "No runs found / could not parse W&B runs. Check entity/project and logging keys."
        )
    df = dedupe_latest(df)

    specs = get_task_specs()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True)

    axes = axes.ravel()

    for ax, spec in zip(axes, specs):
        agg = aggregate_over_models(df, spec.target, spec.metric_col)
        if agg.empty:
            ax.set_axis_off()
            continue
        plot_on_ax(ax, spec, agg)

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.92,
        bottom=0.20,  # space for legend
        hspace=0.35,
        wspace=0.28,
    )

    for ax in axes[:2]:
        ax.set_xlabel("")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.04),
    )

    out_pdf = os.path.join(OUT_DIR, "avg_over_models_2x2.pdf")
    out_svg = os.path.join(OUT_DIR, "avg_over_models_2x2.svg")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] wrote {out_pdf}")
    print(f"[INFO] wrote {out_svg}")


if __name__ == "__main__":
    main()
