from __future__ import annotations

import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch

import dl_fault_analysis.data.labels as L
import wandb
from dl_fault_analysis.config import MainConfig
from dl_fault_analysis.data.data_utils import load_windowed_dataset
from dl_fault_analysis.data.features import maybe_filter_features
from dl_fault_analysis.data.filters import (
    build_valid_row_indices,
    build_valid_row_indices_hv_double_line_90kv,
)
from dl_fault_analysis.data.targets import extract_target
from dl_fault_analysis.data.task_spec import get_task_spec, infer_task_type_from_spec
from dl_fault_analysis.models.model_utils import create_model_from_name, get_device
from dl_fault_analysis.utils.cv_utils import (
    build_cv_splits_stratified,
    split_train_val_from_train_pool,
)
from dl_fault_analysis.utils.eval_utils import evaluate, predict_on_loader
from dl_fault_analysis.utils.logging import get_logger
from dl_fault_analysis.utils.run_utils import (
    get_env_info,
    infer_input_dims,
    log_classification_report,
    save_checkpoint,
    save_confusion_matrix_report,
    save_fold_predictions,
    save_fold_splits,
    set_seed,
    set_torch_perf_flags,
)
from dl_fault_analysis.utils.subgroup_metrics import maybe_run_subgroup_analysis
from dl_fault_analysis.utils.train_utils import make_loaders, train_best_on_val
from dl_fault_analysis.utils.tuning_utils import tune_lr_wd_on_single_fold

logger = get_logger(__name__)


# =============================================================================
# Helper functions for multiclass CV diagnostics and stratification
# =============================================================================


def canonicalize_multiclass_encoding(y_all: np.ndarray, task_type: str):
    if task_type != "multiclass":
        return y_all, {}

    unique_val = np.unique(y_all)

    # sort deterministically
    if unique_val.dtype.kind in {"U", "S", "O"}:
        sorted_classes = sorted([str(x) for x in unique_val])
        canonical_mapping = {c: i for i, c in enumerate(sorted_classes)}
        y_encoded = np.array([canonical_mapping[str(v)] for v in y_all], dtype=np.int64)
    else:
        sorted_classes = sorted([int(x) for x in unique_val])
        canonical_mapping = {int(c): i for i, c in enumerate(sorted_classes)}
        y_encoded = np.array([canonical_mapping[int(v)] for v in y_all], dtype=np.int64)

    return y_encoded, canonical_mapping


def log_class_distribution(
    y_subset: np.ndarray,
    class_to_idx: dict,
    fold_idx: int,
    split_name: str,
    n_splits: int = 5,
) -> None:
    """
    Log per-class distribution for a data split.

    Args:
        y_subset: Labels for the split
        class_to_idx: Class name to index mapping
        fold_idx: Fold number (for logging)
        split_name: "train", "val", "test"
        n_splits: Total number of folds (for logging format)
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    unique_classes, counts = np.unique(y_subset, return_counts=True)

    dist_parts = []
    for class_idx, count in zip(unique_classes, counts):
        class_name = idx_to_class.get(int(class_idx), f"unknown_{class_idx}")
        pct = 100.0 * count / len(y_subset)
        dist_parts.append(f"{class_name}: {int(count)} ({pct:.1f}%)")

    dist_str = " | ".join(dist_parts)
    logger.info(
        "[fold %d/%d] %s class distribution (%d samples): %s",
        fold_idx,
        n_splits,
        split_name.upper(),
        len(y_subset),
        dist_str,
    )


def check_missing_classes(
    y_subset: np.ndarray,
    out_dim: int,
    fold_idx: int,
    split_name: str,
) -> None:
    """
    Check if training data is missing any classes needed for multiclass prediction.

    Args:
        y_subset: Labels for the split
        out_dim: Expected number of output classes
        fold_idx: Fold number (for logging)
        split_name: Split name

    Raises:
        ValueError if training data is missing classes
    """
    classes_present = len(np.unique(y_subset))

    if split_name == "train" and classes_present < out_dim:
        missing_count = out_dim - classes_present
        logger.warning(
            "[fold %d] WARNING: Training set has only %d/%d classes (missing %d)! "
            "Model may fail to predict missing classes.",
            fold_idx,
            classes_present,
            out_dim,
            missing_count,
        )

    if split_name == "test" and classes_present < out_dim:
        missing_count = out_dim - classes_present
        logger.info(
            "[fold %d] Test set has only %d/%d classes (missing %d). "
            "Metrics will only be computed for present classes.",
            fold_idx,
            classes_present,
            out_dim,
            missing_count,
        )


def ckpt_path_for(
    out_dir: str,
    topology: str,
    target_label: str,
    model_name: str,
    window_s: float,
    fold_idx: int,
    seed: int,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    window_ms = int(round(1000.0 * float(window_s)))
    return os.path.join(
        out_dir,
        f"{topology}__{target_label}__{model_name}__W{window_ms}ms__fold{fold_idx}__seed{seed}.pt",
    )


def load_checkpoint_into_model(
    ckpt_path: str, model: torch.nn.Module, device: torch.device
) -> dict:
    """Load checkpoint into model and return checkpoint metadata."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt


def validate_checkpoint_metadata(
    ckpt: dict,
    expected_topology: str,
    expected_target: str,
    expected_model: str,
    expected_window_s: float,
) -> None:
    """Validate that checkpoint metadata matches expected configuration."""
    meta = ckpt.get("meta", {})
    config_dict = ckpt.get("config", {})

    # Extract values from checkpoint
    ckpt_topology = meta.get("topology") or config_dict.get("dataset", {}).get(
        "topology"
    )
    ckpt_target = meta.get("target_label") or config_dict.get("training", {}).get(
        "target_label"
    )
    ckpt_model = meta.get("model_name") or config_dict.get("model", {}).get(
        "model_name"
    )
    ckpt_window = meta.get("window_length_s")

    # Validate critical fields
    mismatches = []
    if ckpt_topology and str(ckpt_topology) != str(expected_topology):
        mismatches.append(
            f"topology: checkpoint={ckpt_topology}, expected={expected_topology}"
        )
    if ckpt_target and str(ckpt_target) != str(expected_target):
        mismatches.append(
            f"target: checkpoint={ckpt_target}, expected={expected_target}"
        )
    if ckpt_model and str(ckpt_model) != str(expected_model):
        mismatches.append(f"model: checkpoint={ckpt_model}, expected={expected_model}")
    if ckpt_window and abs(float(ckpt_window) - float(expected_window_s)) > 1e-6:
        mismatches.append(
            f"window_length: checkpoint={ckpt_window}, expected={expected_window_s}"
        )

    if mismatches:
        logger.warning(
            "Checkpoint metadata mismatch detected:\n  %s\n"
            "Loading may produce incorrect results if model architecture or data preprocessing differs.",
            "\n  ".join(mismatches),
        )


@hydra.main(
    version_base=None, config_path="../../../config", config_name="main-config.yaml"
)
def main(config: MainConfig) -> None:
    """
    Global benchmark training entrypoint (episode-grouped 5-fold CV protocol).

    Pipeline:
      (1) load memmap windows + labels
      (2) optional feature-group selection
      (3) apply target-specific row filtering (memmap-safe via row indices)
      (4) infer task + build CV5 (GroupKFold) splits/loaders
      (5) train best-on-val, evaluate on test
      (6) subgroup analysis + checkpoint
      (7) aggregate mean±std over folds
    """
    # ----------------------------
    # Reproducibility + perf knobs
    # ----------------------------
    set_torch_perf_flags()

    # ----------------------------
    # Load dataset (global view)
    # ----------------------------
    X, labels_df, meta = load_windowed_dataset(config)
    device = get_device()
    logger.info("Device: %s", device)

    # ----------------------------
    # Feature selection (optional)
    # ----------------------------
    include_groups = config.training.feature_groups_include
    materialize = config.training.materialize_feature_filters
    X_used, feature_indices_for_ds = maybe_filter_features(
        X=X,
        meta=meta,
        include_groups=include_groups,
        materialize=materialize,
    )

    # ----------------------------
    # Target + task setup
    # ----------------------------
    target_label = str(config.training.target_label)
    if target_label not in L.ALL_TARGETS:
        raise ValueError(f"Unknown target_label='{target_label}'")

    spec = get_task_spec(target_label)
    task_type = infer_task_type_from_spec(spec)
    criterion = spec.criterion
    primary_name = spec.primary_metric
    higher_is_better = spec.higher_is_better

    # ----------------------------
    # Valid rows for this target
    # ----------------------------
    # valid_row_idx maps from labels_df_used-row -> original memmap row in X.
    if config.dataset.topology == "hv_double_line_90kv":
        valid_row_idx = build_valid_row_indices_hv_double_line_90kv(
            labels_df, target_label
        )
        if valid_row_idx is None:
            labels_df_used = labels_df.reset_index(drop=True)
            logger.info(
                "No custom filtering applied for hv_double_line_90kv (valid_row_idx=None)."
            )
        else:
            labels_df_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
            logger.info(
                "Applied custom filtering for hv_double_line_90kv: kept %d/%d rows (%.2f%%).",
                len(labels_df_used),
                len(labels_df),
                100.0 * (len(labels_df_used) / max(1, len(labels_df))),
            )
    else:
        valid_row_idx = build_valid_row_indices(labels_df, target_label=target_label)
        if valid_row_idx is None:
            labels_df_used = labels_df.reset_index(drop=True)
            logger.info("No target-specific filtering applied (valid_row_idx=None).")
        else:
            labels_df_used = labels_df.iloc[valid_row_idx].reset_index(drop=True)
            logger.info(
                "Applied target-specific filtering: kept %d/%d rows (%.2f%%).",
                len(labels_df_used),
                len(labels_df),
                100.0 * (len(labels_df_used) / max(1, len(labels_df))),
            )

    groups_used = labels_df_used[L.SAMPLE_ID]

    # Extract target
    y_all, class_to_idx = extract_target(labels_df_used, target_label=target_label)

    # ---- CRITICAL FIX #1: Canonicalize multiclass encoding ----
    # Ensures class indices are consistent across all folds (sorted order)
    if task_type == "multiclass":
        y_all, class_to_idx = canonicalize_multiclass_encoding(y_all, task_type)
        logger.info(
            "After canonicalization: %d classes in mapping",
            len(class_to_idx),
        )

    # Guard against unexpected NaNs (ideally handled by FILTER_RULES upstream)
    if np.issubdtype(y_all.dtype, np.floating):
        num_nans = int(np.isnan(y_all).sum())
        if num_nans > 0:
            logger.warning(
                "Target '%s' contains %d NaN values out of %d samples (%.2f%%).",
                target_label,
                num_nans,
                len(y_all),
                (num_nans / len(y_all)) * 100.0,
            )
            raise ValueError("NaN values found in target labels.")

    if task_type == "multiclass":
        if class_to_idx is None:
            raise ValueError("Multiclass target requires class_to_idx mapping.")
        out_dim = len(class_to_idx)
    else:
        out_dim = 1

    # ----------------------------
    # Model input dims (respects feature filtering)
    # ----------------------------
    T, F_eff, flat_dim = infer_input_dims(X_used, feature_indices_for_ds)

    model_name = str(config.model.model_name)

    window_s = float(config.window_extraction.window_length)  # seconds (given by you)
    window_ms = int(round(1000.0 * window_s))
    step_s = float(getattr(config.window_extraction, "step_length_seconds", np.nan))
    logger.info(
        "Setup: target=%s | task_type=%s | n=%d | T=%d F_eff=%d out_dim=%d | window=%dms (%.3fs) | step=%.3fs | spec=%s",
        target_label,
        task_type,
        len(y_all),
        T,
        F_eff,
        out_dim,
        window_ms,
        window_s,
        step_s,
        spec.log_msg,
    )

    # ----------------------------
    # Seed selection (single seed for CV)
    # ----------------------------
    seeds = list(map(int, config.training.seeds))
    if len(seeds) == 0:
        raise RuntimeError("No training seeds provided.")
    seed = int(seeds[0])
    if len(seeds) > 1:
        logger.info("Multiple seeds configured; using first only for CV: %d", seed)

    # ----------------------------
    # Reproducibility tracking
    # ----------------------------
    env_info = get_env_info()
    

    # ----------------------------
    # W&B init (CV-aware; do NOT log a single split)
    # ----------------------------
    wb_run = setup_wandb_logging(
        config,
        target_label,
        task_type,
        labels_df_used,
        groups_used,
        F_eff,
        model_name,
        window_s,
        window_ms,
        seed,
        env_info,
    )

    # ----------------------------
    # Run-specific output directory (under configured out_dir)
    # ----------------------------
    top_out_dir = str(getattr(config.training, "out_dir", "outputs"))
    run_id = wb_run.id if wb_run is not None else env_info["timestamp"]
    run_out_dir = os.path.join(top_out_dir, f"run_{run_id}")
    os.makedirs(run_out_dir, exist_ok=True)

    # ----------------------------
    # CV: episode-grouped 5-fold
    # ----------------------------
    n_splits = int(getattr(config.training, "n_splits", 5))

    groups_np = (
        groups_used.to_numpy()
        if hasattr(groups_used, "to_numpy")
        else np.asarray(groups_used)
    )
    idx_all = np.arange(len(labels_df_used), dtype=int)

    # Use stratified CV splitting for multiclass to ensure balanced class distribution
    splits = build_cv_splits_stratified(
        y_all=y_all,
        groups_np=groups_np,
        task_type=task_type,
        n_splits=n_splits,
        seed=seed,
    )

    best_lr, best_wd, eval_only, resave_eval_only = select_best_lr_wd(
        config,
        X_used,
        list(feature_indices_for_ds) if feature_indices_for_ds is not None else None,
        task_type,
        criterion,
        primary_name,
        higher_is_better,
        valid_row_idx,
        labels_df_used,
        y_all,
        out_dim,
        F_eff,
        flat_dim,
        seed,
        n_splits,
    )

    if wb_run is not None:
        wb_run.summary["protocol/n_splits"] = int(n_splits)
        wb_run.config.update(
            {"lr_used": float(best_lr), "weight_decay_used": float(best_wd)},
            allow_val_change=True,
        )
        wb_run.summary["opt/lr_used"] = float(best_lr)
        wb_run.summary["opt/wd_used"] = float(best_wd)

    # ----------------------------
    # Run bookkeeping
    # ----------------------------
    all_fold_metrics: list[dict] = []

    ckpt_dir = str(getattr(config.training, "ckpt_dir", "outputs/checkpoints"))

    # Iterate folds (GroupKFold CV5, group-safe val split inside each fold)
    for fold_idx, (train_pool_idx, test_idx) in enumerate(splits, start=0):
        train_pool_idx = np.asarray(train_pool_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        set_seed(int(seed))

        # Group-safe val split from train_pool (no leakage across episodes)
        # Note: split_seed is offset by fold_idx to ensure different val splits per fold,
        # which is standard practice to maximize data utilization and reduce variance.
        # The base split_seed still determines reproducibility for a given fold.
        idx_train, idx_val = split_train_val_from_train_pool(
            groups_used=groups_used,
            train_pool_idx=train_pool_idx,
            val_size=float(config.training.val_size),
            split_seed=int(config.training.split_seed) + int(fold_idx),
        )

        logger.info(
            "[fold %d/%d] split sizes: train=%d | val=%d | test=%d",
            fold_idx + 1,
            n_splits,
            len(idx_train),
            len(idx_val),
            len(test_idx),
        )

        # loaders for this fold (memmap-safe via row_indices=valid_row_idx)
        train_loader, val_loader, test_loader = make_loaders(
            X_used=X_used,
            y_all=y_all,
            task_type=task_type,
            feature_indices_for_ds=feature_indices_for_ds,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=test_idx,
            batch_size=int(config.training.batch_size),
            device=device,
            num_workers=int(config.training.num_workers),
            pin_memory=bool(config.training.pin_memory),
            prefetch_factor=int(config.training.prefetch_factor),
            row_indices=valid_row_idx,
        )

        logger.info(
            "=== fold %d/%d | seed %d ===",
            fold_idx + 1,
            n_splits,
            int(seed),
        )

        # Log class distribution for multiclass tasks
        if task_type == "multiclass" and isinstance(class_to_idx, dict):
            log_class_distribution(
                y_all[idx_train], class_to_idx, fold_idx + 1, "train", n_splits
            )
            log_class_distribution(
                y_all[idx_val], class_to_idx, fold_idx + 1, "val", n_splits
            )
            log_class_distribution(
                y_all[test_idx], class_to_idx, fold_idx + 1, "test", n_splits
            )
            check_missing_classes(y_all[idx_train], out_dim, fold_idx + 1, "train")
            check_missing_classes(y_all[test_idx], out_dim, fold_idx + 1, "test")

        model = create_model_from_name(
            config,
            n_features=int(F_eff),
            flattened_dim=int(flat_dim),
            out_dim=int(out_dim),
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(best_lr),
            weight_decay=float(best_wd),
        )

        ckpt_path = ckpt_path_for(
            out_dir=ckpt_dir,
            topology=str(config.dataset.topology),
            target_label=target_label,
            model_name=str(config.model.model_name),
            window_s=float(config.window_extraction.window_length),  # seconds
            fold_idx=int(fold_idx),  # keep 0-based in filenames if you want
            seed=int(seed),
        )

        # train or load
        if eval_only:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"eval_only=true but checkpoint not found: {ckpt_path}"
                )
            ckpt = load_checkpoint_into_model(ckpt_path, model, device)
            # Validate checkpoint metadata to catch configuration mismatches
            validate_checkpoint_metadata(
                ckpt=ckpt,
                expected_topology=str(config.dataset.topology),
                expected_target=target_label,
                expected_model=str(config.model.model_name),
                expected_window_s=float(config.window_extraction.window_length),
            )
            logger.info("Loaded checkpoint: %s", ckpt_path)
        else:
            train_best_on_val(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                task_type=task_type,
                epochs=int(config.training.epochs),
                primary_name=primary_name,
                higher_is_better=higher_is_better,
                binary_threshold=float(config.training.binary_threshold),
                patience=int(getattr(config.training, "patience", 15)),
            )

        # evaluate
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            task_type,
            binary_threshold=float(config.training.binary_threshold),
        )
        logger.info(
            "[fold %d/%d seed %d] test_metrics=%s",
            fold_idx + 1,
            n_splits,
            int(seed),
            test_metrics,
        )

        # store fold+seed metrics for later aggregation
        metrics_row = {
            "fold": int(fold_idx),  # or fold_idx+1 if you prefer 1-based
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
            "n_test": int(len(test_idx)),
            **{f"test/{k}": float(v) for k, v in test_metrics.items()},
        }
        all_fold_metrics.append(metrics_row)

        # detailed outputs (optional)
        y_true_np, y_pred_np, y_score_np = predict_on_loader(
            model=model,
            loader=test_loader,
            device=device,
            task_type=task_type,
            binary_threshold=float(config.training.binary_threshold),
        )
        
        # Define output directory
        out_dir = run_out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        # Save fold split identity (group IDs for reproducibility)
        try:
            split_path = save_fold_splits(
                out_dir=out_dir,
                fold_idx=fold_idx,
                idx_train=idx_train,
                idx_val=idx_val,
                idx_test=test_idx,
                labels_df=labels_df_used,
                group_col=L.SAMPLE_ID,
            )
            logger.info("Saved fold splits: %s", split_path)
        except Exception as e:
            logger.warning("Failed to save fold splits: %s", e)
        
        # Save predictions with metadata
        try:
            meta_cols = [L.EVENT_TYPE, L.STATUS, L.Y_FAULT_LINE, L.DT_START]
            pred_path = save_fold_predictions(
                out_dir=out_dir,
                fold_idx=fold_idx,
                idx_test=test_idx,
                y_true=y_true_np,
                y_pred=y_pred_np,
                y_score=y_score_np,
                labels_df=labels_df_used,
                task_type=task_type,
                meta_cols=meta_cols,
            )
            logger.info("Saved fold predictions: %s", pred_path)
        except Exception as e:
            logger.warning("Failed to save fold predictions: %s", e)
        
        if task_type in {"binary", "multiclass"}:
            log_classification_report(
                y_true_np, y_pred_np, class_to_idx=class_to_idx, logger=logger
            )
            
            # Save confusion matrix and classification report as JSON
            try:
                cm_path, report_path = save_confusion_matrix_report(
                    out_dir=out_dir,
                    fold_idx=fold_idx,
                    y_true=y_true_np,
                    y_pred=y_pred_np,
                    class_to_idx=class_to_idx,
                )
                logger.info("Saved confusion matrix: %s", cm_path)
                logger.info("Saved classification report: %s", report_path)
                
                # Log to W&B if enabled
                if wb_run is not None:
                    import json
                    # Log confusion matrix as W&B table
                    with open(cm_path, "r") as f:
                        cm_data = json.load(f)
                    cm_array = np.array(cm_data["confusion_matrix"])
                    target_names = cm_data["target_names"]
                    
                    # Create W&B confusion matrix table
                    cm_table = wandb.Table(
                        columns=["True\\Pred"] + target_names,
                        data=[[target_names[i]] + row.tolist() 
                              for i, row in enumerate(cm_array)]
                    )
                    wandb.log({f"fold{fold_idx}/confusion_matrix": cm_table})
                    
                    # Log classification report metrics
                    with open(report_path, "r") as f:
                        report_data = json.load(f)
                    
                    # Log per-class metrics
                    for class_name in target_names:
                        if class_name in report_data:
                            class_metrics = report_data[class_name]
                            wandb.log({
                                f"fold{fold_idx}/class_{class_name}/precision": class_metrics.get("precision", 0),
                                f"fold{fold_idx}/class_{class_name}/recall": class_metrics.get("recall", 0),
                                f"fold{fold_idx}/class_{class_name}/f1-score": class_metrics.get("f1-score", 0),
                            })
                    
            except Exception as e:
                logger.warning("Failed to save/log confusion matrix and report: %s", e)

            # subgroup analysis (labels_df is original; idx_test indexes labels_df_used)
            if bool(config.analysis.run_subgroup):
                base_df = labels_df_used if valid_row_idx is None else labels_df
                maybe_run_subgroup_analysis(
                    task_type=task_type,
                    labels_df=base_df,
                    valid_row_idx=valid_row_idx,
                    idx_test=test_idx,
                    y_true=y_true_np,
                    y_pred=y_pred_np,
                    out_dir=out_dir,
                    group_cols=[L.EVENT_TYPE, L.STATUS, L.Y_FAULT_LINE],
                    logger=logger,
                )

        # save checkpoint (after evaluation so metrics included)
        if (not eval_only) or resave_eval_only:
            save_checkpoint(
                ckpt_path,
                model=model,
                config=config,
                meta=meta,
                task_type=task_type,
                target_label=target_label,
                include_groups=include_groups,
                feature_indices_for_ds=feature_indices_for_ds,
                class_to_idx=class_to_idx,
                test_metrics=test_metrics,
                X_used_shape=tuple(X_used.shape),
                fold_idx=int(fold_idx),
                seed=int(seed),
            )
            logger.info("Saved checkpoint: %s", ckpt_path)

    # ============================================================
    # Aggregate results: mean±std over folds
    # ============================================================
    if len(all_fold_metrics) == 0:
        raise RuntimeError("No metrics collected (did you set training.seeds?).")

    df = pd.DataFrame(all_fold_metrics)

    size_cols = ["n_train", "n_val", "n_test"]
    df_fold_sizes = df[size_cols + ["fold"]].sort_values("fold")
    logger.info("Fold test sizes: %s", df_fold_sizes["n_test"].tolist())

    # metric columns are those starting with "test/"
    metric_cols = [c for c in df.columns if c.startswith("test/")]
    if len(metric_cols) == 0:
        raise RuntimeError("No 'test/*' metric columns found in collected results.")

    # df already has one row per fold (single seed)
    df_fold = df.sort_values("fold")

    # mean±std over folds
    mean_metrics = {f"cv5_mean/{c[5:]}": float(df_fold[c].mean()) for c in metric_cols}
    std_metrics = {
        f"cv5_std/{c[5:]}": float(df_fold[c].std(ddof=1)) for c in metric_cols
    }

    # Pretty log: "metric=mean±std"
    pretty = {
        c[5:]: f"{float(df_fold[c].mean()):.6g}±{float(df_fold[c].std(ddof=1)):.3g}"
        for c in metric_cols
    }
    logger.info(
        "=== CV5 results (mean±std over %d folds) ===\n%s", len(df_fold), pretty
    )

    # Optional: save a CSV summary next to outputs
    out_dir = run_out_dir
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "cv5_fold_metrics.csv"), index=False)

    # W&B logging
    if wb_run is not None:
        # Fold table
        try:
            fold_table = wandb.Table(dataframe=df_fold)
            wandb.log({"cv5/fold_metrics": fold_table})
        except Exception:
            # fallback if wandb.Table(dataframe=...) is not supported in your version
            fold_table = wandb.Table(
                columns=list(df_fold.columns),
                data=df_fold.values.tolist(),
            )
            wandb.log({"cv5/fold_metrics": fold_table})

        # Summary scalars
        for k, v in mean_metrics.items():
            wb_run.summary[k] = v
        for k, v in std_metrics.items():
            wb_run.summary[k] = v

    if wb_run is not None:
        wandb.finish()


def select_best_lr_wd(
    config: MainConfig,
    X_used: np.ndarray,
    feature_indices_for_ds: list[int] | np.ndarray | None,
    task_type: str,
    criterion: torch.nn.Module,
    primary_name: str,
    higher_is_better: bool,
    valid_row_idx: np.ndarray | None,
    labels_df_used: pd.DataFrame,
    y_all: np.ndarray,
    out_dim: int,
    F_eff: int,
    flat_dim: int,
    seed: int,
    n_splits: int,
):
    best_lr = float(config.training.learning_rate)
    best_wd = float(config.training.weight_decay)

    eval_only = bool(getattr(config.training, "eval_only", False))
    resave_eval_only = bool(getattr(config.training, "resave_in_eval_only", False))

    tuning_enabled = bool(getattr(config.training, "tune_lr_wd", False))
    if tuning_enabled and (not eval_only):
        lr_grid = list(
            map(float, getattr(config.training, "lr_grid", [1e-4, 3e-4, 1e-3]))
        )
        wd_grid = list(
            map(float, getattr(config.training, "wd_grid", [0.0, 1e-4, 1e-3]))
        )
        calib_fold = int(getattr(config.training, "calib_fold", 0))
        cache_dir = str(getattr(config.training, "tune_cache_dir", "outputs/tuning"))
        subsample_ratio = getattr(config.training, "tune_subsample_ratio", None)
        max_epochs = getattr(config.training, "tune_max_epochs", None)

        best_lr, best_wd, _tune_details = tune_lr_wd_on_single_fold(
            config=config,
            X_used=X_used,
            y_all=y_all,
            task_type=task_type,
            feature_indices_for_ds=feature_indices_for_ds,
            valid_row_idx=valid_row_idx,
            labels_df_used=labels_df_used,
            fold_idx=calib_fold,
            seed=seed,
            F_eff=int(F_eff),
            flat_dim=int(flat_dim),
            out_dim=int(out_dim),
            criterion=criterion,
            primary_name=primary_name,
            higher_is_better=higher_is_better,
            lrs=lr_grid,
            wds=wd_grid,
            cache_dir=cache_dir,
            subsample_ratio=(
                float(subsample_ratio) if subsample_ratio is not None else None
            ),
            max_epochs=int(max_epochs) if max_epochs is not None else None,
            n_splits=int(n_splits),
        )
        logger.info(
            "Tuning selected lr=%g wd=%g (calib_fold=%d)", best_lr, best_wd, calib_fold
        )

    return best_lr, best_wd, eval_only, resave_eval_only


def setup_wandb_logging(
    config: MainConfig,
    target_label: str,
    task_type: str,
    labels_df_used: pd.DataFrame,
    groups_used: pd.Series | np.ndarray,
    F_eff: int,
    model_name: str,
    window_s: float,
    window_ms: int,
    seed: int,
    env_info: dict,
):
    wb_run = None
    if bool(config.tracking.use_wandb):
        wb_run = wandb.init(
            project=config.tracking.project,
            entity=config.tracking.entity,
            mode=config.tracking.mode,
            name=f"{config.dataset.topology}__{target_label}__{model_name}__W{window_ms}ms",
            config={
                "topology": str(config.dataset.topology),
                "target": target_label,
                "task_type": task_type,
                "model": model_name,
                "n_features": int(F_eff),
                "window_length_s": float(window_s),
                "window_length_ms": int(window_ms),
                "batch_size": int(config.training.batch_size),
                "lr": float(config.training.learning_rate),
                "weight_decay": float(config.training.weight_decay),
                "epochs": int(config.training.epochs),
                "split_seed": int(config.training.split_seed),
                "n_seeds": 1,
                # Reproducibility info
                "git_commit": env_info.get("git_commit"),
                "git_status": env_info.get("git_status"),
                "python_version": env_info["python_version"].split()[0],
                "torch_version": env_info["torch_version"],
                "numpy_version": env_info["numpy_version"],
                "hostname": env_info["hostname"],
            },
        )

    if wb_run is not None:
        wb_run.summary["data/n_samples_used"] = int(len(labels_df_used))
        n_groups = (
            int(groups_used.nunique())
            if isinstance(groups_used, pd.Series)
            else len(np.unique(groups_used))
        )
        wb_run.summary["data/n_groups_used"] = int(n_groups)
        wb_run.summary["protocol/split_seed"] = int(config.training.split_seed)
        wb_run.summary["protocol/training_seeds"] = [int(seed)]
        # Environment reproducibility info
        wb_run.summary["env/timestamp"] = env_info["timestamp"]
        wb_run.summary["env/platform"] = env_info["platform"]
        wb_run.summary["env/cuda_available"] = env_info["cuda_available"]
        if env_info["cuda_available"]:
            wb_run.summary["env/cuda_version"] = env_info["cuda_version"]
    return wb_run


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
