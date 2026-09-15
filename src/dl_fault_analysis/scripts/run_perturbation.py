"""Runner for the minimal test-time communication-perturbation experiment.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md for exact source commit and integration
notes (namespace adapted from ``dl_psp`` to ``dl_fault_analysis``; the
scientific logic and archived result values in results/paper/communication/
are unchanged). Manuscript Section 3.6 / Table 9.

Scope (exactly, per the manuscript): task FL, 50 ms, **GRU only**, **full
observability**, **no retraining**; the trained model is evaluated on
**perturbed test inputs** for each of {none, time_shift_2_samples,
block_loss_10ms}.

Reuses the same leaf building blocks as the reduced-observability runner
(make_loaders / create_model_from_name / train_best_on_val) and the shared CV
helpers; it does not import the wandb-coupled main entrypoint.
``run_perturbation_experiment`` is data-injectable for the smoke test.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

import dl_fault_analysis.data.labels as L
from dl_fault_analysis.data.data_utils import scale_sample_to_minus1_1
from dl_fault_analysis.data.perturbations import PERTURBATIONS, apply_perturbation
from dl_fault_analysis.data.targets import extract_target
from dl_fault_analysis.data.task_spec import get_task_spec
from dl_fault_analysis.models.model_utils import create_model_from_name, get_device
from dl_fault_analysis.scripts.run_revision_observability import (
    _df_to_markdown,
    _topology_valid_rows,
)
from dl_fault_analysis.utils.cv_utils import (
    build_cv_splits_stratified,
    split_train_val_from_train_pool,
)
from dl_fault_analysis.utils.logging import get_logger
from dl_fault_analysis.utils.run_utils import get_env_info, set_seed
from dl_fault_analysis.utils.train_utils import make_loaders, train_best_on_val
from dl_fault_analysis.validation import (
    AuditCheck,
    AuditResult,
    audit_episode_disjointness,
    audit_normalization,
    audit_window_grouping,
    write_normalization_audit,
    write_split_audit,
)

logger = get_logger(__name__)

TASK_TYPE = "regression"
METRIC_KEYS = ("mae", "rmse")
INTERPRETATION = {
    "line_conditioned": False,
    "uses_faulted_line_for_input_selection": False,
    "deployment_interpretation": "centralized_full_observability_test_time_perturbation",
}


@dataclass
class PerturbationConfig:
    name: str = "minimal_communication_perturbation_fl_50ms"
    target_label: str = L.Y_FAULT_LOCATION
    model: str = "gru_regressor"
    perturbations: Sequence[str] = field(default_factory=lambda: list(PERTURBATIONS))
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
    out_root: str = "results/paper"

    @classmethod
    def from_block(cls, block: Mapping[str, Any]) -> "PerturbationConfig":
        s = block.get("splitting", {}) or {}
        t = block.get("training", {}) or {}
        r = block.get("reproducibility", {}) or {}
        out = block.get("outputs", {}) or {}
        return cls(
            name=str(block.get("name", cls.name)),
            target_label=str(block.get("target_label", L.Y_FAULT_LOCATION)),
            model=str(block.get("model", "gru_regressor")),
            perturbations=list(block.get("perturbations", list(PERTURBATIONS))),
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
            out_root=str(out.get("out_root", "results/paper")),
        )


def _set_model_name(config: Any, name: str) -> None:
    try:
        config.model.model_name = name
    except Exception:  # DictConfig struct mode
        import omegaconf

        with omegaconf.open_dict(config):
            config.model.model_name = name


@torch.no_grad()
def _eval_perturbed(
    model, X, valid_row_idx, test_idx, *, kind, sampling_frequency, device, batch_size
) -> np.ndarray:
    """Predict on the test set with ``kind`` applied to each raw window (then scaled)."""
    model.eval()
    preds: List[np.ndarray] = []
    test_idx = np.asarray(test_idx, dtype=int)
    for s in range(0, len(test_idx), batch_size):
        rows = np.asarray(valid_row_idx)[test_idx[s : s + batch_size]]
        xs = np.empty((len(rows), X.shape[1], X.shape[2]), dtype=np.float32)
        for k, r in enumerate(rows):
            xr = np.asarray(X[int(r)], dtype=np.float32)
            xr = apply_perturbation(xr, kind, sampling_frequency)
            xs[k] = scale_sample_to_minus1_1(xr)
        out = model(torch.from_numpy(xs).to(device))
        preds.append(out.squeeze(-1).detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds) if preds else np.array([])


def run_perturbation_experiment(
    *,
    config: Any,
    X: np.ndarray,
    labels_df: pd.DataFrame,
    meta: Mapping[str, Any],
    cfg: PerturbationConfig,
    out_dir: str,
    device: Optional[torch.device] = None,
    topology: Optional[str] = None,
) -> Dict[str, Any]:
    """Train full-obs GRU per fold (no retraining), evaluate under each perturbation."""
    device = device or get_device()
    topology = topology or str(
        meta.get("topology", getattr(getattr(config, "dataset", None), "topology", ""))
    )
    sampling_frequency = float(config.dataset.sampling_frequency)
    os.makedirs(out_dir, exist_ok=True)
    _set_model_name(config, cfg.model)

    valid_row_idx = _topology_valid_rows(labels_df, topology, cfg.target_label)
    labels_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
    y_all, _ = extract_target(labels_used, cfg.target_label)
    y_all = np.asarray(y_all, dtype=np.float32)
    groups = labels_used[L.SAMPLE_ID]
    groups_np = groups.to_numpy()
    n_features = int(X.shape[2])
    spec = get_task_spec(cfg.target_label)

    splits = build_cv_splits_stratified(
        y_all, groups_np, TASK_TYPE, cfg.n_folds, cfg.seed
    )

    fold_rows: List[dict] = []
    disjoint: List[dict] = []
    pred_frames: List[pd.DataFrame] = []
    norm_seed_ids: Optional[dict] = None

    for fold, (train_pool_idx, test_idx) in enumerate(splits):
        train_pool_idx = np.asarray(train_pool_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        idx_train, idx_val = split_train_val_from_train_pool(
            groups, train_pool_idx, cfg.inner_val_fraction, cfg.seed + fold
        )
        set_seed(int(cfg.seed))
        train_loader, val_loader, _ = make_loaders(
            X_used=X,
            y_all=y_all,
            task_type=TASK_TYPE,
            feature_indices_for_ds=None,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=test_idx,
            batch_size=int(cfg.batch_size),
            device=device,
            num_workers=int(cfg.num_workers),
            pin_memory=(device.type == "cuda"),
            prefetch_factor=2,
            row_indices=valid_row_idx,
        )
        model = create_model_from_name(
            config, n_features=n_features, flattened_dim=None, out_dim=1
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )
        train_best_on_val(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion=spec.criterion,
            device=device,
            task_type=TASK_TYPE,
            epochs=int(cfg.epochs),
            primary_name=spec.primary_metric,
            higher_is_better=spec.higher_is_better,
            binary_threshold=0.5,
            patience=int(cfg.patience),
            print_every=max(1, int(cfg.epochs)),
        )

        y_true = y_all[test_idx].astype(float)
        n_ep = int(np.unique(groups_np[test_idx]).size)
        for kind in cfg.perturbations:
            y_pred = _eval_perturbed(
                model,
                X,
                valid_row_idx,
                test_idx,
                kind=kind,
                sampling_frequency=sampling_frequency,
                device=device,
                batch_size=int(cfg.batch_size),
            )
            fold_rows.append(
                {
                    "experiment": cfg.name,
                    "perturbation": kind,
                    "model": cfg.model,
                    "observability": "full",
                    "fold": fold,
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    "n_episodes_test": n_ep,
                    "n_windows_test": int(test_idx.size),
                    **INTERPRETATION,
                }
            )
            if cfg.save_predictions:
                pred_frames.append(
                    pd.DataFrame(
                        {
                            "experiment": cfg.name,
                            "perturbation": kind,
                            "model": cfg.model,
                            "fold": fold,
                            "sample_id": groups_np[test_idx],
                            "y_true": y_true,
                            "y_pred": y_pred,
                        }
                    )
                )
        disjoint.append(
            {
                "fold": fold,
                "train": groups_np[idx_train],
                "val": groups_np[idx_val],
                "test": groups_np[test_idx],
            }
        )
        if norm_seed_ids is None:
            norm_seed_ids = disjoint[0]

    # ---- audits (single CV scheme: full observability) ----
    split_audit = AuditResult(name="split_audit")
    wr = [
        {"sample_id": ep, "fold_id": rec["fold"]}
        for rec in disjoint
        for ep in np.unique(np.asarray(rec["test"]))
    ]
    for c in audit_window_grouping(pd.DataFrame(wr)).checks:
        split_audit.add(c)
    for rec in disjoint:
        sub = audit_episode_disjointness(rec["train"], rec["val"], rec["test"])
        split_audit.add(
            AuditCheck(
                name=f"fold{rec['fold']}_disjoint",
                passed=sub.passed,
                details={"checks": [c.to_dict() for c in sub.checks]},
            )
        )
    write_split_audit(os.path.join(out_dir, "split_audit.json"), split_audit)
    seeds = norm_seed_ids or {"train": [], "val": [], "test": []}
    norm_audit = audit_normalization(
        list(np.asarray(seeds["train"]).tolist()),
        list(np.asarray(seeds["val"]).tolist()),
        list(np.asarray(seeds["test"]).tolist()),
    )
    write_normalization_audit(
        os.path.join(out_dir, "normalization_audit.json"), norm_audit
    )
    audits_passed = split_audit.passed and norm_audit.passed

    # ---- metrics + aggregate ----
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(out_dir, "perturbation_metrics.csv"), index=False)
    agg = _aggregate(fold_rows)
    agg.to_csv(os.path.join(out_dir, "aggregate_metrics.csv"), index=False)
    if cfg.save_predictions and pred_frames:
        _write_predictions(out_dir, pd.concat(pred_frames, ignore_index=True))

    _write_table_and_summary(out_dir, agg, cfg, audits_passed)
    _write_provenance(out_dir, cfg, config, meta)
    return {
        "out_dir": out_dir,
        "audits_passed": audits_passed,
        "aggregate": agg,
        "split_audit": split_audit,
        "normalization_audit": norm_audit,
    }


def _aggregate(fold_rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(fold_rows)
    if df.empty:
        return df
    base = df[df["perturbation"] == "none"]["mae"].mean()
    out: List[dict] = []
    for kind, g in df.groupby("perturbation"):
        mae_mean = float(g["mae"].mean())
        out.append(
            {
                "experiment": g["experiment"].iloc[0],
                "perturbation": kind,
                "model": g["model"].iloc[0],
                "mae_mean": mae_mean,
                "mae_std": float(g["mae"].std(ddof=1)) if len(g) > 1 else 0.0,
                "rmse_mean": float(g["rmse"].mean()),
                "rmse_std": float(g["rmse"].std(ddof=1)) if len(g) > 1 else 0.0,
                "delta_mae_abs": mae_mean - base,
                "delta_mae_rel": (mae_mean - base) / base if base else float("nan"),
                "n_episodes": int(g["n_episodes_test"].sum()),
                "n_windows": int(g["n_windows_test"].sum()),
                **INTERPRETATION,
            }
        )
    # deterministic perturbation order: none first
    order = {"none": 0, "time_shift_2_samples": 1, "block_loss_10ms": 2}
    return (
        pd.DataFrame(out)
        .sort_values("perturbation", key=lambda s: s.map(order))
        .reset_index(drop=True)
    )


def _write_predictions(out_dir: str, preds_df: pd.DataFrame) -> None:
    try:
        preds_df.to_parquet(
            os.path.join(out_dir, "predictions.parquet"),
            index=False,
            engine="pyarrow",
            compression="snappy",
        )
    except (ImportError, ModuleNotFoundError, ValueError):
        preds_df.to_csv(
            os.path.join(out_dir, "predictions.csv.gz"), index=False, compression="gzip"
        )


def _write_table_and_summary(
    out_dir: str, agg: pd.DataFrame, cfg: PerturbationConfig, audits_passed: bool
) -> None:
    def fmt(m, s):
        return f"{m * 100:.2f} ± {s * 100:.2f}"

    rows = [
        [r["perturbation"], fmt(r["mae_mean"], r["mae_std"])] for _, r in agg.iterrows()
    ]
    table = pd.DataFrame(rows, columns=["perturbation", f"{cfg.model}_mae"])
    table.to_csv(os.path.join(out_dir, "table_perturbation.csv"), index=False)
    tex = [
        "% MAE in % of line length (mean ± std across CV folds)",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "perturbation & " + cfg.model.replace("_", "\\_") + "\\_mae \\\\",
        "\\midrule",
    ]
    for r in rows:
        tex.append(
            r[0].replace("_", "\\_") + " & " + r[1].replace("±", "$\\pm$") + " \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(out_dir, "table_perturbation.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    md = [
        "# Minimal Communication-Perturbation Summary (FL, 50 ms)",
        "",
        "## Goal",
        "",
        "Probe how a trained 50 ms fault-localization model (GRU, full 8-relay "
        "observability) degrades under small **test-time** input perturbations -- "
        "**no retraining**.",
        "",
        "## Scope",
        "",
        f"Task: FL  \nWindow: 50 ms  \nModel: {cfg.model}  \nObservability: full  \n"
        "Perturbations: none, time_shift_2_samples, block_loss_10ms (test inputs only)",
        "",
        "## Leakage audit",
        "",
        f"Episode disjointness / window grouping / train-only normalization: "
        f"{'pass' if audits_passed else 'FAIL'} (see split_audit.json, normalization_audit.json)",
        "",
        "## Main results (MAE, % of line length)",
        "",
        _df_to_markdown(table),
        "",
        "## Interpretation",
        "",
        "Compare each perturbation's MAE to `none` (the unperturbed full-observability "
        "baseline): a 2-sample time shift probes synchronisation sensitivity; a 10 ms "
        "block loss probes a brief communication dropout.",
        "",
        "## Important limitation",
        "",
        "This is a **compact, test-time-only** sensitivity probe (no retraining, full "
        "observability). It is **not** a broad robustness grid and covers only "
        "these three perturbations by design.",
        "",
        "## Files",
        "",
        "perturbation_metrics.csv, aggregate_metrics.csv, table_perturbation.{csv,tex}, "
        "split_audit.json, normalization_audit.json, run_config.yaml, dataset_hash.txt, "
        "predictions.parquet (if enabled).",
        "",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md))


def _write_provenance(
    out_dir: str, cfg: PerturbationConfig, config: Any, meta: Mapping[str, Any]
) -> None:
    env = get_env_info()
    run_cfg = {
        "perturbation_experiment": asdict(cfg),
        "env": env,
        "dataset": {
            "topology": meta.get("topology"),
            "window_length": meta.get("window_length"),
        },
    }
    try:
        import yaml

        with open(os.path.join(out_dir, "run_config.yaml"), "w") as f:
            yaml.safe_dump(run_cfg, f, sort_keys=False)
    except Exception:
        with open(os.path.join(out_dir, "run_config.yaml"), "w") as f:
            json.dump(run_cfg, f, indent=2, default=str)
    try:
        import hashlib

        from dl_fault_analysis.data.window_io import generate_paths

        x_path, _ = generate_paths(config)
        manifest = str(x_path) + ".manifest.json"
        if os.path.exists(manifest):
            h = hashlib.sha256(open(manifest, "rb").read()).hexdigest()
            with open(os.path.join(out_dir, "dataset_hash.txt"), "w") as f:
                f.write(f"sha256(manifest)={h}\n{manifest}\n")
    except Exception as e:  # pragma: no cover
        logger.info("dataset hash unavailable: %s", e)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Minimal communication-perturbation FL experiment"
    )
    ap.add_argument(
        "--config",
        default="config/paper/communication/minimal_communication_perturbation_fl_50ms.yaml",
    )
    ap.add_argument("--windows-local-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-folds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    from omegaconf import OmegaConf

    block = OmegaConf.to_container(
        OmegaConf.load(args.config).get("perturbation_experiment"), resolve=True
    )
    cfg = PerturbationConfig.from_block(block)
    if args.n_folds:
        cfg.n_folds = int(args.n_folds)
    if args.epochs:
        cfg.epochs = int(args.epochs)
    topology = str(block.get("topology", "hv_double_line_90kv"))
    window_s = int(block.get("window_ms", 50)) / 1000.0

    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config")
    )
    overrides = [
        f"dataset={topology}",
        f"window_extraction.window_length={window_s}",
        f"training.target_label={cfg.target_label}",
        f"model.model_name={cfg.model}",
    ]
    if args.windows_local_dir:
        overrides.append(
            f"window_extraction.windows_local_dir={args.windows_local_dir}"
        )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name="main-config", overrides=overrides)
    with open_dict(config):
        config.tracking.use_wandb = False

    from dl_fault_analysis.data.data_utils import load_windowed_dataset

    X, labels_df, meta = load_windowed_dataset(config)
    out_dir = args.out_dir or os.path.join(cfg.out_root, "communication")
    result = run_perturbation_experiment(
        config=config,
        X=X,
        labels_df=labels_df,
        meta=meta,
        cfg=cfg,
        out_dir=out_dir,
        topology=topology,
    )
    logger.info(
        "Done. audits_passed=%s | outputs in %s", result["audits_passed"], out_dir
    )


if __name__ == "__main__":
    main()
