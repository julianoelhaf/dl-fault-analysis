# dl_fault_analysis/utils/tuning_utils.py
from __future__ import annotations

import itertools
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import dl_fault_analysis.data.labels as L
from dl_fault_analysis.config import MainConfig
from dl_fault_analysis.data.task_spec import get_task_spec, infer_task_type_from_spec
from dl_fault_analysis.models.model_utils import create_model_from_name, get_device
from dl_fault_analysis.utils.cv_utils import (
    build_cv_splits_stratified,
    split_train_val_from_train_pool,
)
from dl_fault_analysis.utils.eval_utils import evaluate
from dl_fault_analysis.utils.logging import get_logger
from dl_fault_analysis.utils.run_utils import set_seed
from dl_fault_analysis.utils.train_utils import make_loaders, train_best_on_val

logger = get_logger(__name__)


def _json_dump(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _json_load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _grid_hash(lrs: List[float], wds: List[float]) -> str:
    # stable, filesystem-friendly id
    s = (
        "lr="
        + ",".join([f"{float(x):.2g}" for x in lrs])
        + "__wd="
        + ",".join([f"{float(x):.2g}" for x in wds])
    )
    return s.replace(".", "p")


def tune_lr_wd_on_single_fold(
    *,
    config: MainConfig,
    X_used: Any,
    y_all: np.ndarray,
    task_type: str,
    feature_indices_for_ds: Any,
    valid_row_idx: Optional[np.ndarray],
    labels_df_used: pd.DataFrame,
    fold_idx: int,
    seed: int,
    F_eff: int,
    flat_dim: int,
    out_dim: int,
    criterion: torch.nn.Module,
    primary_name: str,
    higher_is_better: bool,
    lrs: List[float],
    wds: List[float],
    cache_dir: str,
    subsample_ratio: Optional[float] = None,
    max_epochs: Optional[int] = None,
    n_splits: int = 5,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Calibrate (learning_rate, weight_decay) on ONE predefined fold.
    Selection is based on VAL metric (primary_name), using early stopping inside train_best_on_val().

    - Uses GroupKFold fold_idx to define the train_pool.
    - Uses a group-safe train/val split inside train_pool.
    - Never touches any test fold, so it doesn't contaminate outer CV.

    Returns:
      best_lr, best_wd, details_dict (includes per-trial val metrics and cache metadata)
    """
    topology = str(config.dataset.topology)
    target_label = str(config.training.target_label)
    model_name = str(config.model.model_name)

    window_s = float(config.window_extraction.window_length)
    window_ms = int(round(1000.0 * window_s))

    grid_id = _grid_hash(lrs, wds)
    cache_path = os.path.join(
        str(cache_dir),
        f"tune__{topology}__{target_label}__{model_name}__W{window_ms}ms__fold{int(fold_idx)}__seed{int(seed)}__{grid_id}.json",
    )

    # --------------------------
    # Cache
    # --------------------------
    if os.path.exists(cache_path):
        d = _json_load(cache_path)
        logger.info(
            "[tuning] Cache hit: %s | best_lr=%g best_wd=%g best_%s=%g",
            cache_path,
            float(d.get("best_lr", np.nan)),
            float(d.get("best_wd", np.nan)),
            str(d.get("primary_name", primary_name)),
            float(d.get("best_score", np.nan)),
        )
        return float(d["best_lr"]), float(d["best_wd"]), d

    logger.info(
        "[tuning] Cache miss: %s | starting calibration on fold=%d seed=%d",
        cache_path,
        int(fold_idx),
        int(seed),
    )

    # --------------------------
    # Early cache directory validation (before expensive computation)
    # --------------------------
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # Test write permission with a temporary file
        test_file = cache_path + ".tmp"
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except (OSError, IOError, PermissionError) as exc:
        raise RuntimeError(
            f"Cannot create or write to cache directory: {os.path.dirname(cache_path)}. "
            f"Please check permissions or set a valid tune_cache_dir in config."
        ) from exc

    # --------------------------
    # Groups / fold definition
    # --------------------------
    if L.SAMPLE_ID not in labels_df_used.columns:
        raise ValueError(
            "labels_df_used must contain column L.SAMPLE_ID (episode/group id)."
        )

    groups_used = labels_df_used[L.SAMPLE_ID].astype(int)
    groups_np = groups_used.to_numpy()

    # Extract target to get task type
    from dl_fault_analysis.data.targets import extract_target

    y_fold, _ = extract_target(labels_df_used, config.training.target_label)
    task_type_fold = infer_task_type_from_spec(
        get_task_spec(config.training.target_label)
    )

    splits = build_cv_splits_stratified(
        y_all=y_fold,
        groups_np=groups_np,
        task_type=task_type_fold,
        n_splits=int(n_splits),
        seed=int(seed),
    )
    if not (0 <= int(fold_idx) < len(splits)):
        raise ValueError(f"fold_idx={fold_idx} out of range for n_splits={n_splits}")

    train_pool_idx, _test_idx = splits[int(fold_idx)]
    train_pool_idx = np.asarray(train_pool_idx, dtype=int)

    idx_train, idx_val = split_train_val_from_train_pool(
        groups_used=groups_used,
        train_pool_idx=train_pool_idx,
        val_size=float(config.training.val_size),
        split_seed=int(config.training.split_seed) + int(fold_idx),
    )

    n_train_before = int(len(idx_train))
    n_val = int(len(idx_val))

    # Optional subsampling of TRAIN only (keep VAL intact for stable selection)
    did_subsample = False
    if subsample_ratio is not None and 0.0 < float(subsample_ratio) < 1.0:
        did_subsample = True
        rng = np.random.default_rng(int(seed) + 1337 + int(fold_idx))
        n_take = max(1, int(round(float(subsample_ratio) * len(idx_train))))
        idx_train = rng.choice(idx_train, size=n_take, replace=False).astype(int)

    logger.info(
        "[tuning] Fold=%d split: train_pool=%d | train=%d%s | val=%d | val_size=%.2f | split_seed=%d",
        int(fold_idx),
        int(len(train_pool_idx)),
        int(len(idx_train)),
        f" (subsampled from {n_train_before})" if did_subsample else "",
        n_val,
        float(config.training.val_size),
        int(config.training.split_seed) + int(fold_idx),
    )

    # --------------------------
    # Loaders
    # --------------------------
    device = get_device()
    train_loader, val_loader, _unused = make_loaders(
        X_used=X_used,
        y_all=y_all,
        task_type=task_type,
        feature_indices_for_ds=feature_indices_for_ds,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_val,  # unused
        batch_size=int(config.training.batch_size),
        device=device,
        num_workers=int(config.training.num_workers),
        pin_memory=bool(config.training.pin_memory),
        prefetch_factor=int(config.training.prefetch_factor),
        row_indices=valid_row_idx,
    )

    patience = int(getattr(config.training, "patience", 15))
    epochs = int(config.training.epochs) if max_epochs is None else int(max_epochs)

    grid_size = int(len(lrs) * len(wds))
    logger.info(
        "[tuning] Grid size=%d | primary=%s (%s) | epochs=%d | patience=%d | lrs=%s | wds=%s",
        grid_size,
        str(primary_name),
        "higher=better" if bool(higher_is_better) else "lower=better",
        int(epochs),
        int(patience),
        [float(x) for x in lrs],
        [float(x) for x in wds],
    )

    # --------------------------
    # Run grid
    # --------------------------
    trials: List[Dict[str, Any]] = []
    best_trial: Optional[Dict[str, Any]] = None

    for t_idx, (lr, wd) in enumerate(itertools.product(lrs, wds), start=1):
        set_seed(int(seed))

        model = create_model_from_name(
            config,
            n_features=int(F_eff),
            flattened_dim=int(flat_dim),
            out_dim=int(out_dim),
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(wd),
        )

        logger.info(
            "[tuning] Trial %d/%d | lr=%g wd=%g",
            int(t_idx),
            grid_size,
            float(lr),
            float(wd),
        )

        train_best_on_val(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            task_type=task_type,
            epochs=epochs,
            primary_name=primary_name,
            higher_is_better=higher_is_better,
            binary_threshold=float(config.training.binary_threshold),
            patience=patience,
            print_every=10,  # override default to reduce verbosity during tuning
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            task_type=task_type,
            binary_threshold=float(config.training.binary_threshold),
        )

        if primary_name not in val_metrics:
            logger.warning(
                "[tuning] primary metric '%s' missing in val_metrics keys=%s",
                str(primary_name),
                list(val_metrics.keys()),
            )

        score = float(val_metrics.get(primary_name, np.nan))
        if np.isnan(score):
            logger.warning(
                "[tuning] Trial %d/%d | lr=%g wd=%g produced NaN score for primary '%s'. Full val_metrics=%s",
                int(t_idx),
                grid_size,
                float(lr),
                float(wd),
                str(primary_name),
                {k: float(v) for k, v in val_metrics.items()},
            )

        trial = {
            "lr": float(lr),
            "wd": float(wd),
            "score": score,
            "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        }
        trials.append(trial)

        if best_trial is None:
            best_trial = trial
        else:
            best_score = float(best_trial["score"])
            improved = (higher_is_better and score > best_score) or (
                (not higher_is_better) and score < best_score
            )
            if improved:
                best_trial = trial

        if best_trial is not None:
            logger.info(
                "[tuning] Trial %d/%d done | score(%s)=%g | best so far: lr=%g wd=%g score=%g",
                int(t_idx),
                grid_size,
                str(primary_name),
                float(score),
                float(best_trial["lr"]),
                float(best_trial["wd"]),
                float(best_trial["score"]),
            )

        logger.debug("[tuning] Full val_metrics: %s", trial["val_metrics"])

    if best_trial is None:
        raise RuntimeError("Tuning failed: no trials evaluated.")

    logger.info(
        "[tuning] Selected best: lr=%g wd=%g | %s=%g | wrote cache=%s",
        float(best_trial["lr"]),
        float(best_trial["wd"]),
        str(primary_name),
        float(best_trial["score"]),
        cache_path,
    )

    out: Dict[str, Any] = {
        "topology": topology,
        "target_label": target_label,
        "model_name": model_name,
        "window_ms": int(window_ms),
        "fold_idx": int(fold_idx),
        "seed": int(seed),
        "primary_name": str(primary_name),
        "higher_is_better": bool(higher_is_better),
        "grid": {
            "learning_rate": [float(x) for x in lrs],
            "weight_decay": [float(x) for x in wds],
        },
        "subsample_ratio": subsample_ratio,
        "max_epochs": max_epochs,
        "best_lr": float(best_trial["lr"]),
        "best_wd": float(best_trial["wd"]),
        "best_score": float(best_trial["score"]),
        "trials": trials,
    }

    _json_dump(cache_path, out)
    return float(out["best_lr"]), float(out["best_wd"]), out
