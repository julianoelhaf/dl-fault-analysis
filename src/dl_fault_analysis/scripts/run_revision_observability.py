"""Runner for the EPSR reduced-observability fault-localization experiment.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md for exact source commit and integration
notes (namespace adapted from ``dl_psp`` to ``dl_fault_analysis``; the
scientific logic and archived result values in results/paper/observability/
are unchanged). Manuscript Section 3.5 / Table 8.

Grid: FL (regression), 50 ms, models x observability in
{full, terminal_pair, single_ended(a,b)}. For the line-conditioned modes a
**separate model is trained per faulted line** on that line's terminal channels.

The orchestration reuses the existing pipeline leaf functions
(``make_loaders``, ``create_model_from_name``, ``train_best_on_val``,
``evaluate``) plus the observability / line-conditioned / audit modules. It
deliberately does **not** import ``scripts.run_dl_experiment`` (which
requires ``wandb``); CV helpers come from ``utils.cv_utils``.

``run_experiment`` is data-injectable so it can be exercised on tiny synthetic
data (CPU) by the smoke tests; ``main`` wires up Hydra config + real windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import dl_fault_analysis.data.labels as L
from dl_fault_analysis.data import line_conditioned as lc
from dl_fault_analysis.data import observability as obs
from dl_fault_analysis.data.filters import (
    build_valid_row_indices,
    build_valid_row_indices_hv_double_line_90kv,
)
from dl_fault_analysis.data.targets import extract_target
from dl_fault_analysis.data.task_spec import get_task_spec
from dl_fault_analysis.models.model_utils import create_model_from_name, get_device
from dl_fault_analysis.utils.cv_utils import (
    build_cv_splits_stratified,
    split_train_val_from_train_pool,
)
from dl_fault_analysis.utils.eval_utils import evaluate
from dl_fault_analysis.utils.logging import get_logger
from dl_fault_analysis.utils.run_utils import get_env_info, set_seed
from dl_fault_analysis.utils.train_utils import make_loaders, train_best_on_val
from dl_fault_analysis.validation import (
    AuditCheck,
    AuditResult,
    assert_passed,
    audit_episode_disjointness,
    audit_line_conditioned_splits,
    audit_normalization,
    audit_window_grouping,
    write_normalization_audit,
    write_split_audit,
)

logger = get_logger(__name__)

METRIC_KEYS = ("mae", "rmse")
TASK_TYPE = "regression"
INTERPRETATION = {
    "line_conditioned": True,
    "uses_faulted_line_for_input_selection": True,
    "deployment_interpretation": "conditional_on_fault_line_identification",
}


@dataclass
class RevisionConfig:
    """Resolved experiment knobs (parsed from the revision yaml or built in tests)."""

    name: str = "reduced_observability_fl_50ms"
    target_label: str = L.Y_FAULT_LOCATION
    models: Sequence[str] = field(default_factory=lambda: ["gru_regressor"])
    modes: Sequence[str] = field(
        default_factory=lambda: ["full", "terminal_pair", "single_ended"]
    )
    single_ended_sides: Sequence[str] = field(default_factory=lambda: ["a", "b"])
    n_folds: int = 5
    inner_val_fraction: float = 0.15
    seed: int = 42
    epochs: int = 500
    patience: int = 15
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    save_predictions: bool = True
    save_checkpoints: bool = True
    out_root: str = "results/paper"

    @classmethod
    def from_block(cls, block: Mapping[str, Any]) -> "RevisionConfig":
        o = block.get("observability", {}) or {}
        s = block.get("splitting", {}) or {}
        t = block.get("training", {}) or {}
        r = block.get("reproducibility", {}) or {}
        out = block.get("outputs", {}) or {}
        return cls(
            name=str(block.get("name", cls.name)),
            target_label=str(block.get("target_label", L.Y_FAULT_LOCATION)),
            models=list(block.get("models", ["gru_regressor"])),
            modes=list(o.get("modes", ["full", "terminal_pair", "single_ended"])),
            single_ended_sides=list(o.get("single_ended_sides", ["a", "b"])),
            n_folds=int(s.get("n_folds", 5)),
            inner_val_fraction=float(s.get("inner_val_fraction", 0.15)),
            seed=int(r.get("seed", 42)),
            epochs=int(t.get("epochs", 500)),
            patience=int(t.get("patience", 15)),
            batch_size=int(t.get("batch_size", 256)),
            learning_rate=float(t.get("learning_rate", 1e-4)),
            weight_decay=float(t.get("weight_decay", 1e-4)),
            num_workers=int(t.get("num_workers", 4)),
            save_predictions=bool(out.get("save_predictions", True)),
            save_checkpoints=bool(out.get("save_checkpoints", True)),
            out_root=str(out.get("out_root", "results/paper")),
        )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _set_model_name(config: Any, name: str) -> None:
    try:
        config.model.model_name = name
    except Exception:  # DictConfig in struct mode
        import omegaconf

        with omegaconf.open_dict(config):
            config.model.model_name = name


def _topology_valid_rows(
    labels_df: pd.DataFrame, topology: str, target_label: str
) -> np.ndarray:
    """Target-valid memmap row indices, using the correct filter for the topology."""
    if topology == "hv_double_line_90kv":
        return build_valid_row_indices_hv_double_line_90kv(labels_df, target_label)
    return build_valid_row_indices(labels_df, target_label)


def _train_eval_one_fold(
    *,
    config: Any,
    X: np.ndarray,
    y_space: np.ndarray,
    row_indices: np.ndarray,
    feature_indices: Optional[Sequence[int]],
    n_features: int,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    rev: RevisionConfig,
    device: torch.device,
    ckpt_path: Optional[str] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Train (best-on-val) + evaluate one fold. Returns (metrics, y_true, y_pred).

    If ``ckpt_path`` is given, the trained (best-on-val) model is saved there so it
    can be reused later (e.g. perturbation eval without retraining).
    """
    set_seed(int(rev.seed))
    spec = get_task_spec(rev.target_label)

    train_loader, val_loader, test_loader = make_loaders(
        X_used=X,
        y_all=y_space,
        task_type=TASK_TYPE,
        feature_indices_for_ds=(
            list(feature_indices) if feature_indices is not None else None
        ),
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        batch_size=int(rev.batch_size),
        device=device,
        num_workers=int(rev.num_workers),
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
        row_indices=row_indices,
    )

    model = create_model_from_name(
        config, n_features=int(n_features), flattened_dim=None, out_dim=1
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(rev.learning_rate),
        weight_decay=float(rev.weight_decay),
    )
    train_best_on_val(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion=spec.criterion,
        device=device,
        task_type=TASK_TYPE,
        epochs=int(rev.epochs),
        primary_name=spec.primary_metric,
        higher_is_better=spec.higher_is_better,
        binary_threshold=0.5,
        patience=int(rev.patience),
        print_every=max(1, int(rev.epochs)),  # quiet
    )
    metrics = evaluate(model, test_loader, device, TASK_TYPE)

    if ckpt_path is not None:
        from dl_fault_analysis.utils.ckpt import save_checkpoint

        save_checkpoint(
            ckpt_path,
            model,
            meta={
                "model_name": str(config.model.model_name),
                "target_label": rev.target_label,
                "n_features": int(n_features),
                "seed": int(rev.seed),
                "test_metrics": dict(metrics),
            },
        )

    # predictions for this fold (test split), in normalized [0,1] units
    from dl_fault_analysis.utils.eval_utils import predict_on_loader

    y_true, y_pred, _ = predict_on_loader(
        model, test_loader, device, TASK_TYPE, binary_threshold=0.5
    )
    return metrics, np.asarray(y_true), np.asarray(y_pred)


# ---------------------------------------------------------------------------
# Cell runners
# ---------------------------------------------------------------------------
def _window_meta(
    labels_used: pd.DataFrame, abs_pos: np.ndarray
) -> Dict[str, np.ndarray]:
    """Per-window metadata (only columns present in labels)."""
    sub = labels_used.iloc[np.asarray(abs_pos, dtype=int)]
    out: Dict[str, np.ndarray] = {}
    if "window_idx" in sub.columns:
        out["window_id"] = sub["window_idx"].to_numpy()
    if "window_start_time" in sub.columns:
        ws = sub["window_start_time"].to_numpy()
        out["window_start"] = ws
        if "window_duration" in sub.columns:
            out["window_end"] = ws + sub["window_duration"].to_numpy()
    if L.Y_FAULT_LINE in sub.columns:
        out["faulted_line"] = sub[L.Y_FAULT_LINE].to_numpy()
    if L.Y_FAULT_LOCATION in sub.columns:
        out["fault_location"] = sub[L.Y_FAULT_LOCATION].to_numpy()
    return out


def _run_full(
    *,
    config,
    X,
    labels_used,
    y_all,
    valid_row_idx,
    feature_names,
    rev,
    device,
    collect_preds,
    ckpt_dir=None,
) -> Tuple[List[dict], List[dict], List[pd.DataFrame]]:
    """Full-observability global model. Returns (fold_rows, disjoint, preds)."""
    model_name = str(config.model.model_name)
    groups = labels_used[L.SAMPLE_ID]
    groups_np = groups.to_numpy()
    splits = build_cv_splits_stratified(
        y_all, groups_np, TASK_TYPE, rev.n_folds, rev.seed
    )
    n_features = int(X.shape[2])
    full_relays = ";".join(
        obs.describe_selection(feature_names, "full")["selected_relays"]
    )

    fold_rows: List[dict] = []
    disjoint_records: List[dict] = []
    preds: List[pd.DataFrame] = []

    for fold, (train_pool_idx, test_idx) in enumerate(splits):
        train_pool_idx = np.asarray(train_pool_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        idx_train, idx_val = split_train_val_from_train_pool(
            groups, train_pool_idx, rev.inner_val_fraction, rev.seed + fold
        )
        metrics, y_true, y_pred = _train_eval_one_fold(
            config=config,
            X=X,
            y_space=y_all,
            row_indices=valid_row_idx,
            feature_indices=None,
            n_features=n_features,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=test_idx,
            rev=rev,
            device=device,
            ckpt_path=(
                os.path.join(ckpt_dir, f"full__fold{fold}.pt") if ckpt_dir else None
            ),
        )
        fold_rows.append(
            {
                "observability": "full",
                "line_conditioned": False,
                "line": None,
                "terminal_side": None,
                "fold": fold,
                "n_relays": 8,
                "n_channels": n_features,
                "n_episodes_test": int(np.unique(groups_np[test_idx]).size),
                "n_windows_test": int(test_idx.size),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
            }
        )
        disjoint_records.append(
            {
                "line": "full",
                "fold": fold,
                "train": groups_np[idx_train],
                "val": groups_np[idx_val],
                "test": groups_np[test_idx],
            }
        )
        if collect_preds:
            preds.append(
                pd.DataFrame(
                    {
                        "observability": "full",
                        "observability_mode": "full",
                        "model": model_name,
                        "line": None,
                        "terminal_side": None,
                        "fold": fold,
                        "sample_id": groups_np[test_idx],
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "selected_relays": full_relays,
                        "line_conditioned": False,
                        **_window_meta(labels_used, test_idx),
                    }
                )
            )
    return fold_rows, disjoint_records, preds


def _run_line_conditioned(
    *,
    config,
    X,
    labels_used,
    y_all,
    valid_row_idx,
    feature_names,
    rev,
    device,
    mode,
    side,
    collect_preds,
    ckpt_dir=None,
) -> Tuple[List[dict], List[dict], List[pd.DataFrame]]:
    """One model per faulted line for terminal_pair / single_ended."""
    model_name = str(config.model.model_name)
    groups_all = labels_used[L.SAMPLE_ID].to_numpy()
    lines = lc.unique_faulted_lines(labels_used)
    fold_rows: List[dict] = []
    disjoint_records: List[dict] = []
    preds: List[pd.DataFrame] = []

    for line in lines:
        obs_idx = obs.select_observability(
            feature_names, mode, faulted_line=line, terminal_side=side
        )
        obs.validate_observability_selection(obs_idx, feature_names, mode)
        desc = obs.describe_selection(feature_names, mode, line, side)
        n_features = len(obs_idx)

        line_pos, splits = lc.build_line_conditioned_splits(
            labels_used, line, rev.n_folds, rev.seed
        )
        y_line = y_all[line_pos]
        row_indices_line = valid_row_idx[line_pos]
        groups_line = pd.Series(groups_all[line_pos])  # fresh RangeIndex 0..n_line-1

        for fold, (train_pool_idx, test_idx) in enumerate(splits):
            train_pool_idx = np.asarray(train_pool_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            idx_train, idx_val = split_train_val_from_train_pool(
                groups_line, train_pool_idx, rev.inner_val_fraction, rev.seed + fold
            )
            metrics, y_true, y_pred = _train_eval_one_fold(
                config=config,
                X=X,
                y_space=y_line,
                row_indices=row_indices_line,
                feature_indices=obs_idx,
                n_features=n_features,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=test_idx,
                rev=rev,
                device=device,
                ckpt_path=(
                    os.path.join(
                        ckpt_dir, f"{mode}__{line}__{side or 'x'}__fold{fold}.pt"
                    )
                    if ckpt_dir
                    else None
                ),
            )
            g = groups_line.to_numpy()
            fold_rows.append(
                {
                    "observability": mode,
                    "line_conditioned": True,
                    "line": line,
                    "terminal_side": side,
                    "fold": fold,
                    "n_relays": int(desc["n_relays"]),
                    "n_channels": n_features,
                    "selected_relays": ";".join(desc["selected_relays"]),
                    "n_episodes_test": int(np.unique(g[test_idx]).size),
                    "n_windows_test": int(test_idx.size),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                }
            )
            disjoint_records.append(
                {
                    "line": line,
                    "fold": fold,
                    "train": g[idx_train],
                    "val": g[idx_val],
                    "test": g[test_idx],
                }
            )
            if collect_preds:
                preds.append(
                    pd.DataFrame(
                        {
                            "observability": mode,
                            "observability_mode": mode,
                            "model": model_name,
                            "line": line,
                            "terminal_side": side,
                            "fold": fold,
                            "sample_id": g[test_idx],
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "selected_relays": ";".join(desc["selected_relays"]),
                            "line_conditioned": True,
                            **_window_meta(labels_used, line_pos[test_idx]),
                        }
                    )
                )
    return fold_rows, disjoint_records, preds


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _aggregate(fold_rows: List[dict]) -> pd.DataFrame:
    """Aggregate fold rows -> one row per (observability, model[, side]) with mean/std + delta."""
    df = pd.DataFrame(fold_rows)
    if df.empty:
        return df
    out: List[dict] = []

    # baseline (full) per model for delta computation
    full_mae = {
        m: g["mae"].mean()
        for m, g in df[df["observability"] == "full"].groupby("model")
    }

    for (obsv, model), g in df.groupby(["observability", "model"]):
        rows_for = g
        if obsv == "single_ended":
            # mean-side then worst-side aggregation across lines
            by_side: dict[str, dict[str, list]] = {}
            for side, gs in g.groupby("terminal_side"):
                per_line: dict[str, list] = {}
                for line, gl in gs.groupby("line"):
                    per_line[str(line)] = gl[["mae", "rmse"]].to_dict("records")
                by_side[str(side)] = per_line
            summ = lc.summarize_single_ended_sides(by_side, METRIC_KEYS)
            mae_mean, mae_std = summ["mae_mean"], summ["mae_std"]
            rmse_mean, rmse_std = summ["rmse_mean"], summ["rmse_std"]
            extra = {
                "mae_worst_mean": summ["mae_worst_mean"],
                "rmse_worst_mean": summ["rmse_worst_mean"],
            }
        else:
            mae_mean = float(rows_for["mae"].mean())
            mae_std = float(rows_for["mae"].std(ddof=1)) if len(rows_for) > 1 else 0.0
            rmse_mean = float(rows_for["rmse"].mean())
            rmse_std = float(rows_for["rmse"].std(ddof=1)) if len(rows_for) > 1 else 0.0
            extra = {}

        base = full_mae.get(model, float("nan"))
        delta_abs = mae_mean - base
        delta_rel = (
            (delta_abs / base)
            if base and np.isfinite(base) and base != 0
            else float("nan")
        )
        # Coverage: single_ended evaluates BOTH terminal sides over the same episodes;
        # count one side so n_episodes/n_windows aren't double-counted vs full/terminal_pair.
        cov = rows_for
        if obsv == "single_ended":
            cov = rows_for[
                rows_for["terminal_side"] == rows_for["terminal_side"].iloc[0]
            ]
        lc_flag = bool(rows_for["line_conditioned"].iloc[0])
        out.append(
            {
                "experiment": rows_for["experiment"].iloc[0],
                "observability": obsv,
                "model": model,
                "line_conditioned": lc_flag,
                "uses_faulted_line_for_input_selection": lc_flag,
                "deployment_interpretation": (
                    INTERPRETATION["deployment_interpretation"]
                    if lc_flag
                    else "centralized_full_observability"
                ),
                "n_relays": int(rows_for["n_relays"].iloc[0]),
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "delta_mae_abs": delta_abs,
                "delta_mae_rel": delta_rel,
                "n_episodes": int(cov["n_episodes_test"].sum()),
                "n_windows": int(cov["n_windows_test"].sum()),
                **extra,
            }
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Tables / summary
# ---------------------------------------------------------------------------
def _fmt(mean: float, std: float, scale: float = 100.0) -> str:
    return f"{mean * scale:.2f} ± {std * scale:.2f}"


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Markdown table without requiring the optional `tabulate` dependency."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(map(str, cols)) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)


def write_table_and_summary(
    out_dir: str, agg: pd.DataFrame, rev: RevisionConfig, audits_passed: bool
) -> None:
    """Compact paper table (csv + tex) and a summary.md (MAE in % of line length)."""
    models = list(dict.fromkeys(agg["model"].tolist()))
    modes = ["full", "full_line_conditioned", "terminal_pair", "single_ended"]
    relays = {
        "full": 8,
        "full_line_conditioned": 8,
        "terminal_pair": 2,
        "single_ended": 1,
    }

    header = ["observability", "relays"] + [f"{m}_mae" for m in models]
    rows: List[List[str]] = []
    for mode in modes:
        if mode not in set(agg["observability"]):
            continue
        cells = [mode, str(relays.get(mode, ""))]
        for m in models:
            sel = agg[(agg["observability"] == mode) & (agg["model"] == m)]
            cells.append(
                _fmt(sel["mae_mean"].iloc[0], sel["mae_std"].iloc[0])
                if len(sel)
                else "—"
            )
        rows.append(cells)

    table_df = pd.DataFrame(rows, columns=header)
    table_df.to_csv(
        os.path.join(out_dir, "table_reduced_observability.csv"), index=False
    )

    # LaTeX
    tex = [
        "% MAE in % of line length (mean ± std across CV folds / per-line models)",
        "\\begin{tabular}{l" + "r" * (len(header) - 1) + "}",
        "\\toprule",
        " & ".join(h.replace("_", "\\_") for h in header) + " \\\\",
        "\\midrule",
    ]
    for r in rows:
        tex.append(
            " & ".join(c.replace("_", "\\_").replace("±", "$\\pm$") for c in r)
            + " \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out_dir, "table_reduced_observability.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    # summary.md
    md = [
        "# Reduced-Observability Fault Localization Summary",
        "",
        "## Goal",
        "",
        "Assess how much 50 ms fault localization depends on full centralized "
        "eight-relay waveform access.",
        "",
        "## Scope",
        "",
        f"Task: FL  \nWindow: 50 ms  \nModels: {', '.join(models)}  \n"
        "Observability: full, full-line-conditioned, terminal-pair, single-ended",
        "",
        "## Optimizer / baseline caveat",
        "",
        "All cells use a **fixed optimizer (lr = wd = 1e-4, no hyperparameter "
        "tuning)** so the observability conditions are directly comparable. The "
        "`full` row is therefore an **internal fixed-optimizer reference**, not a "
        "replacement for the tuned main-paper 50 ms FL table (which uses LR/WD "
        "search and reports slightly lower MAE). `full_line_conditioned` (per-line "
        "model, all 8 relays) is the control that isolates the per-line-modelling "
        "effect from the relay-count effect; compare it to `terminal_pair`/"
        "`single_ended` to attribute degradation to reduced sensing alone.",
        "",
        "## Leakage audit",
        "",
        f"Episode leakage / window grouping / train-only normalization: "
        f"{'pass' if audits_passed else 'FAIL'} (see split_audit.json, normalization_audit.json)",
        "",
        "## Main results (MAE, % of line length)",
        "",
        _df_to_markdown(table_df),
        "",
        "## Interpretation",
        "",
        "Higher MAE under reduced observability indicates greater dependence on "
        "centralized sensing for accurate localization.",
        "",
        "## Important limitation",
        "",
        "Terminal-pair and single-ended settings are **line-conditioned**: they assume "
        "the faulted line is known (from a prior FLI stage or line-specific deployment) "
        "and use it to select terminal measurements. They are **not** a complete "
        "line-agnostic local-relay deployment setting "
        "(deployment_interpretation: conditional_on_fault_line_identification).",
        "",
        "## Files",
        "",
        "run_config.yaml, git_commit.txt, environment.txt, split_audit.json, "
        "normalization_audit.json, fold_metrics.csv, aggregate_metrics.csv, "
        "table_reduced_observability.{csv,tex}, predictions.csv (if enabled).",
        "",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md))


def merge_results(in_dirs: Sequence[str], out_dir: str) -> str:
    """Combine per-model result dirs into one aggregate table + summary.

    Used after a per-model job split: concatenates each dir's
    aggregate_metrics.csv / fold_metrics.csv and regenerates the combined paper
    table (models as columns). audits_passed is the AND of the per-model audits.
    """
    os.makedirs(out_dir, exist_ok=True)
    folds: List[pd.DataFrame] = []
    passed = True
    for d in in_dirs:
        fp = os.path.join(d, "fold_metrics.csv")
        if not os.path.exists(fp):
            raise ValueError(f"no fold_metrics.csv in {d}")
        fold = pd.read_csv(fp)
        folds.append(fold)
        # Re-aggregate each dir from its raw fold metrics so corrections to
        # _aggregate (e.g. single_ended coverage) propagate without retraining.
        _aggregate(fold.to_dict("records")).to_csv(
            os.path.join(d, "aggregate_metrics.csv"), index=False
        )
        for af in ("split_audit.json", "normalization_audit.json"):
            p = os.path.join(d, af)
            if os.path.exists(p):
                with open(p) as fh:
                    passed = passed and bool(json.load(fh).get("passed", False))
    # Combined aggregate from ALL fold rows (groupby is per observability x model).
    all_fold = pd.concat(folds, ignore_index=True)
    all_fold.to_csv(os.path.join(out_dir, "fold_metrics.csv"), index=False)
    agg = _aggregate(all_fold.to_dict("records"))
    agg.to_csv(os.path.join(out_dir, "aggregate_metrics.csv"), index=False)
    name = (
        str(agg["experiment"].iloc[0])
        if "experiment" in agg.columns and len(agg)
        else "reduced_observability_fl_50ms"
    )
    rev = RevisionConfig(name=name, models=list(dict.fromkeys(agg["model"].tolist())))
    write_table_and_summary(out_dir, agg, rev, passed)
    # combined audit provenance (per-model audit JSONs live in the per-model subdirs)
    audits: Dict[str, Any] = {"audits_passed": bool(passed), "per_model": {}}
    for d in in_dirs:
        m = os.path.basename(os.path.normpath(d))
        entry = {}
        for af in ("split_audit.json", "normalization_audit.json"):
            p = os.path.join(d, af)
            if os.path.exists(p):
                with open(p) as fh:
                    entry[af] = bool(json.load(fh).get("passed", False))
        audits["per_model"][m] = entry
    with open(os.path.join(out_dir, "audits_summary.json"), "w") as f:
        json.dump(audits, f, indent=2, sort_keys=True)
    # Keep ONE provenance copy at the experiment level (per-model subdirs are
    # gitignored). run_config documents the knobs; dataset_hash the data version.
    import shutil

    for fname in ("run_config.yaml", "dataset_hash.txt"):
        src = os.path.join(in_dirs[0], fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))
    logger.info(
        "Merged %d dirs -> %s (audits_passed=%s)", len(in_dirs), out_dir, passed
    )
    return out_dir


# ---------------------------------------------------------------------------
# Reproducibility artifacts
# ---------------------------------------------------------------------------
def _write_predictions(out_dir: str, preds_df: pd.DataFrame) -> str:
    """Write per-window predictions (normalized [0,1] units). Parquet, csv.gz fallback."""
    path = os.path.join(out_dir, "predictions.parquet")
    try:
        preds_df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    except (ImportError, ModuleNotFoundError, ValueError):
        path = os.path.join(out_dir, "predictions.csv.gz")
        preds_df.to_csv(path, index=False, compression="gzip")
    return path


def _write_reproducibility(
    out_dir: str, rev: RevisionConfig, config: Any, meta: Mapping[str, Any]
) -> None:
    env = get_env_info()
    if env.get("git_commit"):
        with open(os.path.join(out_dir, "git_commit.txt"), "w") as f:
            f.write(str(env["git_commit"]) + "\n")
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        with open(os.path.join(out_dir, "environment.txt"), "w") as f:
            f.write(freeze)
    except Exception as e:  # pragma: no cover
        logger.warning("pip freeze failed: %s", e)

    run_cfg = {
        "revision_experiment": asdict(rev),
        "env": env,
        "dataset": {
            "topology": meta.get("topology"),
            "window_length": meta.get("window_length"),
            "n_features": len(meta.get("feature_names", [])),
        },
    }
    try:
        import yaml

        with open(os.path.join(out_dir, "run_config.yaml"), "w") as f:
            yaml.safe_dump(run_cfg, f, sort_keys=False)
    except Exception:
        with open(os.path.join(out_dir, "run_config.yaml"), "w") as f:
            json.dump(run_cfg, f, indent=2, default=str)

    # dataset hash (best-effort: hash the windows manifest next to the memmap)
    try:
        from dl_fault_analysis.data.window_io import generate_paths

        x_path, _ = generate_paths(config)
        manifest = str(x_path) + ".manifest.json"
        if os.path.exists(manifest):
            h = hashlib.sha256(open(manifest, "rb").read()).hexdigest()
            with open(os.path.join(out_dir, "dataset_hash.txt"), "w") as f:
                f.write(f"sha256(manifest)={h}\n{manifest}\n")
    except Exception as e:  # pragma: no cover
        logger.info("dataset hash unavailable: %s", e)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_experiment(
    *,
    config: Any,
    X: np.ndarray,
    labels_df: pd.DataFrame,
    meta: Mapping[str, Any],
    rev: RevisionConfig,
    out_dir: str,
    device: Optional[torch.device] = None,
    topology: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full reduced-observability grid and write all artifacts."""
    device = device or get_device()
    topology = topology or str(
        meta.get("topology", getattr(getattr(config, "dataset", None), "topology", ""))
    )
    feature_names = list(meta.get("feature_names", []))
    if not feature_names:
        raise ValueError(
            "meta['feature_names'] is required for observability selection"
        )
    os.makedirs(out_dir, exist_ok=True)

    # FL-valid rows (90 kV -> status == 'fault_start'); build target once.
    valid_row_idx = _topology_valid_rows(labels_df, topology, rev.target_label)
    labels_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
    y_all, _ = extract_target(labels_used, rev.target_label)
    y_all = np.asarray(y_all, dtype=np.float32)

    all_fold_rows: List[dict] = []
    all_disjoint: List[dict] = []
    pred_frames: List[pd.DataFrame] = []
    norm_seed_ids: Optional[dict] = None
    collect = bool(rev.save_predictions)
    ckpt_dir = os.path.join(out_dir, "checkpoints") if rev.save_checkpoints else None

    for model_name in rev.models:
        _set_model_name(config, model_name)
        for mode in rev.modes:
            logger.info("=== model=%s | observability=%s ===", model_name, mode)
            if mode == "full":
                fr, dj, pr = _run_full(
                    config=config,
                    X=X,
                    labels_used=labels_used,
                    y_all=y_all,
                    valid_row_idx=valid_row_idx,
                    feature_names=feature_names,
                    rev=rev,
                    device=device,
                    collect_preds=collect,
                    ckpt_dir=ckpt_dir,
                )
                if norm_seed_ids is None and dj:
                    norm_seed_ids = dj[0]
            elif mode in ("full_line_conditioned", "terminal_pair"):
                # both are per-line single-side cells; full_line_conditioned uses
                # all 8 relays (control isolating per-line modelling from relay count).
                fr, dj, pr = _run_line_conditioned(
                    config=config,
                    X=X,
                    labels_used=labels_used,
                    y_all=y_all,
                    valid_row_idx=valid_row_idx,
                    feature_names=feature_names,
                    rev=rev,
                    device=device,
                    mode=mode,
                    side=None,
                    collect_preds=collect,
                    ckpt_dir=ckpt_dir,
                )
            elif mode == "single_ended":
                fr, dj, pr = [], [], []
                for side in rev.single_ended_sides:
                    fr_s, dj_s, pr_s = _run_line_conditioned(
                        config=config,
                        X=X,
                        labels_used=labels_used,
                        y_all=y_all,
                        valid_row_idx=valid_row_idx,
                        feature_names=feature_names,
                        rev=rev,
                        device=device,
                        mode=mode,
                        side=side,
                        collect_preds=collect,
                        ckpt_dir=ckpt_dir,
                    )
                    for row in fr_s:
                        row["terminal_side"] = side
                    fr += fr_s
                    dj += dj_s
                    pr += pr_s
            else:
                raise ValueError(f"unknown observability mode {mode!r}")
            pred_frames += pr

            for row in fr:
                row["experiment"] = rev.name
                row["model"] = model_name
                row.setdefault("terminal_side", None)
                # Carry the declared interpretation on line-conditioned rows.
                if row.get("line_conditioned"):
                    row["uses_faulted_line_for_input_selection"] = INTERPRETATION[
                        "uses_faulted_line_for_input_selection"
                    ]
                    row["deployment_interpretation"] = INTERPRETATION[
                        "deployment_interpretation"
                    ]
            all_fold_rows += fr
            all_disjoint += dj

    # ---- audits ----
    split_audit = AuditResult(name="split_audit")
    # Window grouping is verified per CV scheme (full + each faulted line) so the
    # check is present for ANY combination of modes (not only when 'full' runs).
    # Mixing schemes would falsely flag an episode tested under both full and its
    # line partition, so we group test episodes by scheme first.
    schemes: dict[str, list] = {}
    for rec in all_disjoint:
        key = "full" if rec["line"] == "full" else f"line:{rec['line']}"
        for ep in np.unique(np.asarray(rec["test"])):
            schemes.setdefault(key, []).append(
                {"sample_id": ep, "fold_id": rec["fold"]}
            )
    if not schemes:
        split_audit.add(
            AuditCheck(
                name="window_grouping_present",
                passed=False,
                details={"reason": "no test records to audit"},
            )
        )
    for key, rows in schemes.items():
        for c in audit_window_grouping(pd.DataFrame(rows)).checks:
            split_audit.add(
                AuditCheck(name=f"{key}_{c.name}", passed=c.passed, details=c.details)
            )
    # per (line/full, fold) disjointness
    line_records = [r for r in all_disjoint if r["line"] != "full"]
    full_records = [r for r in all_disjoint if r["line"] == "full"]
    for rec in full_records:
        sub = audit_episode_disjointness(rec["train"], rec["val"], rec["test"])
        split_audit.add(
            AuditCheck(
                name=f"full_fold{rec['fold']}_disjoint",
                passed=sub.passed,
                details={"checks": [c.to_dict() for c in sub.checks]},
            )
        )
    if line_records:
        for c in audit_line_conditioned_splits(line_records).checks:
            split_audit.add(c)
    write_split_audit(os.path.join(out_dir, "split_audit.json"), split_audit)

    norm_seed_ids = norm_seed_ids or (
        all_disjoint[0] if all_disjoint else {"train": [], "val": [], "test": []}
    )
    norm_audit = audit_normalization(
        list(np.asarray(norm_seed_ids["train"]).tolist()),
        list(np.asarray(norm_seed_ids["val"]).tolist()),
        list(np.asarray(norm_seed_ids["test"]).tolist()),
    )
    write_normalization_audit(
        os.path.join(out_dir, "normalization_audit.json"), norm_audit
    )
    audits_passed = split_audit.passed and norm_audit.passed

    # ---- metrics ----
    fold_df = pd.DataFrame(all_fold_rows)
    fold_df.to_csv(os.path.join(out_dir, "fold_metrics.csv"), index=False)
    agg = _aggregate(all_fold_rows)
    agg.to_csv(os.path.join(out_dir, "aggregate_metrics.csv"), index=False)

    if collect and pred_frames:
        preds_df = pd.concat(pred_frames, ignore_index=True)
        preds_df.insert(0, "experiment", rev.name)
        _write_predictions(out_dir, preds_df)

    _write_reproducibility(out_dir, rev, config, meta)
    write_table_and_summary(out_dir, agg, rev, audits_passed)

    return {
        "out_dir": out_dir,
        "audits_passed": audits_passed,
        "n_fold_rows": len(all_fold_rows),
        "aggregate": agg,
        "split_audit": split_audit,
        "normalization_audit": norm_audit,
    }


# ---------------------------------------------------------------------------
# Entry point (real data via Hydra compose)
# ---------------------------------------------------------------------------
def _load_revision_block(path: str) -> dict:
    from omegaconf import DictConfig, OmegaConf

    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    block = cfg.get("revision_experiment", cfg)
    return OmegaConf.to_container(block, resolve=True)  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser(description="Reduced-observability FL experiment")
    ap.add_argument(
        "--revision-config",
        default="config/paper/observability/reduced_observability_fl_50ms.yaml",
    )
    ap.add_argument(
        "--windows-local-dir",
        default=None,
        help="override window_extraction.windows_local_dir",
    )
    ap.add_argument("--out-dir", default=None, help="override output dir")
    # Optional overrides for quick validation runs / per-model job splitting.
    ap.add_argument("--models", nargs="*", default=None, help="override model list")
    ap.add_argument(
        "--modes", nargs="*", default=None, help="override observability modes"
    )
    ap.add_argument("--n-folds", type=int, default=None, help="override CV folds")
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs")
    ap.add_argument(
        "--patience", type=int, default=None, help="override early-stop patience"
    )
    ap.add_argument(
        "--merge",
        nargs="*",
        default=None,
        help="merge these per-model result dirs into a combined table (no training)",
    )
    args = ap.parse_args()

    if args.merge:
        out_dir = args.out_dir or os.path.join("results/paper", "observability")
        merge_results(args.merge, out_dir)
        return

    block = _load_revision_block(args.revision_config)
    rev = RevisionConfig.from_block(block)
    if args.models:
        rev.models = list(args.models)
    if args.modes:
        rev.modes = list(args.modes)
    if args.n_folds:
        rev.n_folds = int(args.n_folds)
    if args.epochs:
        rev.epochs = int(args.epochs)
    if args.patience:
        rev.patience = int(args.patience)
    topology = str(block.get("topology", "hv_double_line_90kv"))
    window_ms = int(block.get("window_ms", 50))
    window_s = window_ms / 1000.0

    # Compose the base pipeline config (dataset/model/window/training) via Hydra.
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config")
    )
    overrides = [
        f"dataset={topology}",
        f"window_extraction.window_length={window_s}",
        f"training.target_label={rev.target_label}",
        f"model.model_name={rev.models[0]}",
    ]
    if args.windows_local_dir:
        overrides.append(
            f"window_extraction.windows_local_dir={args.windows_local_dir}"
        )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name="main-config", overrides=overrides)
    with open_dict(config):
        config.tracking.use_wandb = False  # runner does not use wandb

    from dl_fault_analysis.data.data_utils import load_windowed_dataset

    X, labels_df, meta = load_windowed_dataset(config)
    out_dir = args.out_dir or os.path.join(rev.out_root, "observability")

    result = run_experiment(
        config=config,
        X=X,
        labels_df=labels_df,
        meta=meta,
        rev=rev,
        out_dir=out_dir,
        topology=topology,
    )
    logger.info(
        "Done. audits_passed=%s | outputs in %s", result["audits_passed"], out_dir
    )
    assert_passed(result["split_audit"])
    assert_passed(result["normalization_audit"])


if __name__ == "__main__":
    main()
