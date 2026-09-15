# src/dl_fault_analysis/utils/run_utils.py
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from dl_fault_analysis.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Best-effort reproducibility (does not force full determinism)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_torch_perf_flags() -> None:
    """Optional perf knobs; safe no-ops on older torch."""
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_git_commit() -> Optional[str]:
    """Get current git commit hash for reproducibility tracking."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_status() -> Optional[str]:
    """Check if working directory is clean for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            status = result.stdout.strip()
            return "clean" if not status else "dirty"
    except Exception:
        pass
    return None


def get_env_info() -> Dict[str, Any]:
    """Collect environment information for reproducibility."""
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "unknown"

    try:
        import pandas

        pandas_version = pandas.__version__
    except Exception:
        pandas_version = "unknown"

    env_info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn_version,
        "pandas_version": pandas_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        ),
        "git_commit": get_git_commit(),
        "git_status": get_git_status(),
    }

    logger.debug(
        "Git commit: %s (%s)",
        env_info.get("git_commit", "unknown"),
        env_info.get("git_status", "unknown"),
    )
    logger.debug("Python: %s", env_info["python_version"].split()[0])
    logger.debug(
        "PyTorch: %s | NumPy: %s", env_info["torch_version"], env_info["numpy_version"]
    )
    logger.debug(
        "Platform: %s | Hostname: %s", env_info["platform"], env_info["hostname"]
    )
    if env_info["git_status"] == "dirty":
        logger.warning(
            "Git working directory is dirty - results may not be fully reproducible!"
        )

    return env_info


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def infer_input_dims(
    X_used: np.ndarray,
    feature_indices_for_ds: Optional[Sequence[int]],
) -> Tuple[int, int, int]:
    """
    Infer (T, F_eff, flat_dim) for model construction.
    X_used shape: (N, T, F_full)
    """
    _, T, F_full = X_used.shape
    F_eff = (
        int(len(feature_indices_for_ds))
        if feature_indices_for_ds is not None
        else int(F_full)
    )
    flat_dim = int(T) * int(F_eff)
    return int(T), int(F_eff), int(flat_dim)


def log_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_to_idx: Optional[Dict[Any, int]],
    logger: logging.Logger,
    digits: int = 4,
) -> None:
    """Logs confusion matrix + classification report with stable target_names."""
    if class_to_idx is not None:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(idx_to_class)
        target_names = [str(idx_to_class[i]) for i in range(num_classes)]
    else:
        target_names = ["0", "1"]

    logger.info("Confusion matrix:\n%s", confusion_matrix(y_true, y_pred))
    logger.info(
        "Classification report:\n%s",
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=digits,
            zero_division=0,
        ),
    )


def _to_jsonable(obj: Any) -> Any:
    """Convert common Hydra/dataclass-ish objects to JSON-able structures."""
    if obj is None:
        return None

    # dataclass instance vs dataclass class
    if is_dataclass(obj):
        if isinstance(obj, type):
            # dataclass *class* -> stringify or return its name
            return f"{obj.__module__}.{obj.__qualname__}"
        return asdict(obj)  # dataclass *instance*

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    try:
        import omegaconf  # type: ignore

        if isinstance(obj, (omegaconf.DictConfig, omegaconf.ListConfig)):
            return _to_jsonable(omegaconf.OmegaConf.to_container(obj, resolve=True))
    except Exception:
        pass

    return str(obj)


def save_checkpoint(
    path: str,
    *,
    model: nn.Module,
    config: Any,
    meta: Dict[str, Any],
    task_type: str,
    target_label: str,
    include_groups: Sequence[str],
    feature_indices_for_ds: Optional[Sequence[int]],
    class_to_idx: Optional[Dict[Any, int]],
    test_metrics: Dict[str, float],
    X_used_shape: Tuple[int, int, int],
    fold_idx: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Save a compact checkpoint + metadata with full reproducibility info.
    Keeps feature indices truncated to avoid huge payloads.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    env_info = get_env_info()

    payload = {
        "state_dict": model.state_dict(),
        "task_type": task_type,
        "target_label": target_label,
        "feature_groups_include": list(include_groups),
        "selected_feature_indices": (
            list(feature_indices_for_ds)[:5000] if feature_indices_for_ds else []
        ),
        "class_to_idx": class_to_idx,
        "meta": {
            "topology": meta.get("topology"),
            "sampling_frequency": meta.get("sampling_frequency"),
            "window_length": meta.get("window_length"),
            "X_shape": tuple(X_used_shape),
        },
        "test_metrics": dict(test_metrics),
        "train_cfg": _to_jsonable(getattr(config, "training", None)),
        "model_cfg": _to_jsonable(getattr(config, "model", None)),
        "dataset_cfg": _to_jsonable(getattr(config, "dataset", None)),
        "window_cfg": _to_jsonable(getattr(config, "window_extraction", None)),
        "reproducibility": {
            **env_info,
            "fold_idx": fold_idx,
            "seed": seed,
            "model_params": count_params(model),
        },
    }

    torch.save(payload, path)

    # Optional: write tiny sidecar JSON for quick grep/diff (no tensors)
    sidecar_path = os.path.splitext(path)[0] + ".meta.json"
    sidecar = {k: v for k, v in payload.items() if k != "state_dict"}
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)


def save_fold_predictions(
    out_dir: str,
    fold_idx: int,
    idx_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    labels_df: Any,
    task_type: str,
    meta_cols: Optional[list[str]] = None,
) -> str:
    """
    Save per-fold predictions as parquet with metadata.

    Args:
        out_dir: Output directory for predictions
        fold_idx: Fold number
        idx_test: Test indices (rows in labels_df)
        y_true: True labels
        y_pred: Predicted labels
        y_score: Prediction scores/probabilities
        labels_df: Labels dataframe with metadata
        task_type: "binary", "multiclass", or "regression"
        meta_cols: Optional metadata columns to include

    Returns:
        Path to saved file
    """
    import pandas as pd

    preds_dir = os.path.join(out_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    # Build predictions dataframe
    pred_data = {
        "fold": fold_idx,
        "idx_test": idx_test,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    # Add scores/probabilities
    if task_type == "binary":
        pred_data["y_score"] = y_score.astype(np.float32)
    elif task_type == "multiclass":
        # Save full probability matrix as float16 to save space
        if y_score.ndim == 2:
            for class_idx in range(y_score.shape[1]):
                pred_data[f"proba_class_{class_idx}"] = y_score[:, class_idx].astype(
                    np.float16
                )
        else:
            pred_data["y_score"] = y_score.astype(np.float32)
    else:  # regression
        pred_data["y_score"] = y_score.astype(np.float32)

    # Add metadata columns if available
    if meta_cols:
        for col in meta_cols:
            if col in labels_df.columns:
                test_meta = labels_df.iloc[idx_test][col].values
                pred_data[col] = test_meta

    # Always include sample_id if available
    if hasattr(labels_df, "columns") and "sample_id" in labels_df.columns:
        pred_data["sample_id"] = labels_df.iloc[idx_test]["sample_id"].values

    pred_df = pd.DataFrame(pred_data)

    # Save as parquet (compressed by default), fallback to CSV if pyarrow not available
    out_path = os.path.join(preds_dir, f"fold{fold_idx}.parquet")
    try:
        pred_df.to_parquet(
            out_path, index=False, engine="pyarrow", compression="snappy"
        )
    except (ImportError, ModuleNotFoundError):
        # Fallback to CSV if pyarrow not available
        out_path = os.path.join(preds_dir, f"fold{fold_idx}.csv.gz")
        pred_df.to_csv(out_path, index=False, compression="gzip")

    return out_path


def save_fold_splits(
    out_dir: str,
    fold_idx: int,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    labels_df: Any,
    group_col: str = "sample_id",
) -> str:
    """
    Save fold split identity (group IDs) for reproducibility.

    Args:
        out_dir: Output directory
        fold_idx: Fold number
        idx_train: Training indices
        idx_val: Validation indices
        idx_test: Test indices
        labels_df: Labels dataframe with group column
        group_col: Column name for grouping (e.g., "sample_id")

    Returns:
        Path to saved file
    """
    splits_dir = os.path.join(out_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    # Extract group IDs for each split
    split_data = {
        "fold": fold_idx,
        "train_groups": sorted(labels_df.iloc[idx_train][group_col].unique().tolist()),
        "val_groups": sorted(labels_df.iloc[idx_val][group_col].unique().tolist()),
        "test_groups": sorted(labels_df.iloc[idx_test][group_col].unique().tolist()),
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
    }

    out_path = os.path.join(splits_dir, f"fold{fold_idx}_splits.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)

    return out_path


def save_confusion_matrix_report(
    out_dir: str,
    fold_idx: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_to_idx: Optional[Dict[Any, int]] = None,
) -> tuple[str, str]:
    """
    Save confusion matrix and classification report as JSON.

    Args:
        out_dir: Output directory
        fold_idx: Fold number
        y_true: True labels
        y_pred: Predicted labels
        class_to_idx: Class name to index mapping

    Returns:
        Tuple of (confusion_matrix_path, classification_report_path)
    """
    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Prepare target names
    if class_to_idx is not None:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(idx_to_class)
        target_names = [str(idx_to_class[i]) for i in range(num_classes)]
    else:
        target_names = ["0", "1"]

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_data = {
        "fold": fold_idx,
        "confusion_matrix": cm.tolist(),
        "target_names": target_names,
    }
    cm_path = os.path.join(reports_dir, f"fold{fold_idx}_confusion_matrix.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm_data, f, indent=2)

    # Classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    if isinstance(report_dict, dict):
        report_dict["fold"] = fold_idx
    else:
        report_dict = {"fold": fold_idx, "report": str(report_dict)}

    report_path = os.path.join(
        reports_dir, f"fold{fold_idx}_classification_report.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    return cm_path, report_path
