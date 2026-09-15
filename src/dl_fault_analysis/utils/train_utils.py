from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from dl_fault_analysis.data.data_utils import WindowedDataset
from dl_fault_analysis.data.filters import build_group_strat_labels, get_strat_key
from dl_fault_analysis.utils.eval_utils import evaluate
from dl_fault_analysis.utils.logging import get_logger

logger = get_logger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    task_type: str,
) -> float:
    model.train()
    losses: List[float] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(xb)

        if task_type == "multiclass":
            loss = criterion(out, yb)
        else:
            loss = criterion(out.squeeze(-1), yb)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float("nan")


def train_best_on_val(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    task_type: str,
    epochs: int,
    primary_name: str,
    higher_is_better: bool,
    binary_threshold: float,
    patience: int = 15,
    print_every: int = 1,
) -> None:
    best_val: Optional[float] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    stop_epoch: int = 0
    early_stopped: bool = False

    logger.info(
        "=== Training start | epochs=%d | patience=%d | primary=%s (%s) ===",
        int(epochs),
        int(patience),
        str(primary_name),
        "higher=better" if higher_is_better else "lower=better",
    )

    for epoch in range(1, epochs + 1):
        stop_epoch = epoch

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, task_type
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            task_type,
            binary_threshold=binary_threshold,
        )

        # Guard: missing primary metric is a hard error (better than silently selecting NaN)
        if primary_name not in val_metrics:
            raise KeyError(
                f"primary_name='{primary_name}' not found in val_metrics keys={list(val_metrics.keys())}"
            )

        primary = float(val_metrics[primary_name])
        is_better = (best_val is None) or (
            (primary > best_val) if higher_is_better else (primary < best_val)
        )

        if is_better:
            best_val = primary
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % int(print_every) == 0:
            suffix = (
                " (best)"
                if is_better
                else f" | no_improve={epochs_without_improvement}/{patience}"
            )
            logger.info(
                "Epoch %03d | train_loss=%.5f | val_loss=%.5f | %s=%.5f%s",
                epoch,
                float(train_loss),
                float(val_metrics["loss"]),
                primary_name,
                primary,
                suffix,
            )

        # Early stopping
        if epochs_without_improvement >= int(patience):
            early_stopped = True
            logger.info(
                "Early stopping at epoch %d | best_epoch=%d | best_%s=%.5f",
                int(epoch),
                int(best_epoch),
                str(primary_name),
                float(best_val) if best_val is not None else float("nan"),
            )
            break

    hit_cap = (stop_epoch >= int(epochs)) and (not early_stopped)

    if hit_cap:
        logger.info(
            "Reached epoch cap (%d) without early stopping | best_epoch=%d | best_%s=%.5f",
            int(epochs),
            int(best_epoch),
            str(primary_name),
            float(best_val) if best_val is not None else float("nan"),
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(
            "Restored best model | best_epoch=%d | stop_epoch=%d | early_stopped=%s | hit_cap=%s | best_%s=%.5f",
            int(best_epoch),
            int(stop_epoch),
            str(early_stopped),
            str(hit_cap),
            str(primary_name),
            float(best_val) if best_val is not None else float("nan"),
        )
    else:
        logger.warning(
            "No best_state captured (unexpected) | stop_epoch=%d | early_stopped=%s | hit_cap=%s",
            int(stop_epoch),
            str(early_stopped),
            str(hit_cap),
        )


def split_indices(
    groups: pd.Series,
    labels_df: pd.DataFrame,
    target_label: str,
    test_size: float,
    val_size: float,
    random_state: int,
    stratify: bool = True,  # default ON
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Grouped train/val/test split (episode-level), optionally stratified by an
    episode-constant key defined in STRAT_RULES (via build_group_strat_labels).
    """

    # ----------------------------
    # Sanity checks
    # ----------------------------
    assert len(groups) > 0, "empty input"
    assert len(labels_df) == len(groups), "labels_df and groups must have same length"
    assert 0.0 < test_size < 1.0
    assert 0.0 < val_size < 1.0
    assert test_size + val_size < 1.0

    group_ids = groups.to_numpy()
    unique_groups = np.unique(group_ids)
    assert unique_groups.size >= 2, "need at least two groups"

    # ----------------------------
    # Logging: split setup
    # ----------------------------
    logger.info(
        "Episode-level split: test_size=%.2f, val_size=%.2f (seed=%d), n_groups=%d",
        test_size,
        val_size,
        random_state,
        unique_groups.size,
    )

    group_counts = np.unique(group_ids, return_counts=True)[1]
    logger.debug(
        "Group size statistics: min=%d, max=%d, mean=%.2f",
        group_counts.min(),
        group_counts.max(),
        group_counts.mean(),
    )

    # ----------------------------
    # Stratification (episode-level)
    # ----------------------------
    strat = None
    if stratify:
        strat = build_group_strat_labels(
            labels_df=labels_df, groups=groups, target_label=target_label
        )
        strat_key = get_strat_key(target_label)
        if strat is not None:
            logger.info(
                "Using episode-level stratification for target '%s' by %s (n_classes=%d)",
                target_label,
                strat_key,
                len(np.unique(strat)),
            )
            logger.debug(
                "Strat class counts: %s",
                dict(zip(*np.unique(strat, return_counts=True))),
            )
            assert len(strat) == len(
                unique_groups
            ), "strat labels must align with unique groups"
        else:
            logger.info(
                "No stratification defined for target '%s' — proceeding unstratified",
                target_label,
            )
    else:
        logger.info("Stratification disabled explicitly for target '%s'", target_label)

    # ----------------------------
    # Split: train+val / test
    # ----------------------------
    g_trainval, g_test = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=strat,
    )

    # ----------------------------
    # Split: train / val
    # ----------------------------
    val_rel = val_size / (1.0 - test_size)
    strat_trainval = None
    if strat is not None:
        strat_trainval = strat[np.isin(unique_groups, g_trainval)]

    g_train, g_val = train_test_split(
        g_trainval,
        test_size=val_rel,
        random_state=random_state,
        shuffle=True,
        stratify=strat_trainval,
    )

    # ----------------------------
    # Map back to row indices
    # ----------------------------
    idx_train = np.flatnonzero(np.isin(group_ids, g_train)).astype(np.int64)
    idx_val = np.flatnonzero(np.isin(group_ids, g_val)).astype(np.int64)
    idx_test = np.flatnonzero(np.isin(group_ids, g_test)).astype(np.int64)

    assert idx_train.size and idx_val.size and idx_test.size, "empty split"

    # ----------------------------
    # Leakage checks + debug
    # ----------------------------
    train_groups = set(group_ids[idx_train])
    val_groups = set(group_ids[idx_val])
    test_groups = set(group_ids[idx_test])

    assert train_groups.isdisjoint(val_groups), "group leakage train ↔ val"
    assert train_groups.isdisjoint(test_groups), "group leakage train ↔ test"
    assert val_groups.isdisjoint(test_groups), "group leakage val ↔ test"

    logger.debug(
        "Final split groups: train=%d | val=%d | test=%d",
        len(train_groups),
        len(val_groups),
        len(test_groups),
    )

    return idx_train, idx_val, idx_test


def make_loaders(
    *,
    X_used: np.ndarray,
    y_all: np.ndarray,
    task_type: str,
    feature_indices_for_ds: Optional[Sequence[int]],
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    row_indices: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test loaders.

    Index spaces:
      - idx_* are indices into the *filtered* space (0..N_filtered-1), where N_filtered=len(y_all).
      - If row_indices is provided (shape (N_filtered,)), each filtered index j maps to X_used[row_indices[j]].
      - If row_indices is None, filtered index j maps directly to X_used[j].
    """
    # ---- normalize arrays ----
    y_all = np.asarray(y_all)
    idx_train = np.asarray(idx_train, dtype=np.int64)
    idx_val = np.asarray(idx_val, dtype=np.int64)
    idx_test = np.asarray(idx_test, dtype=np.int64)

    if y_all.ndim != 1:
        raise ValueError(f"y_all must be 1D, got shape {y_all.shape}")

    n_filtered = int(y_all.shape[0])
    if n_filtered == 0:
        raise ValueError("y_all must be non-empty")

    if row_indices is not None:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1:
            raise ValueError("row_indices must be 1D")
        if len(row_indices) != n_filtered:
            raise ValueError(
                f"Length mismatch: row_indices has {len(row_indices)}, y_all has {n_filtered}"
            )
    else:
        # Without mapping, y_all must align 1:1 with X_used
        if X_used.shape[0] != n_filtered:
            raise ValueError(
                f"Length mismatch: X_used has {X_used.shape[0]} rows, y_all has {n_filtered}"
            )

    # ---- split index safety ----
    def _check_split(name: str, idx: np.ndarray) -> None:
        if idx.ndim != 1:
            raise ValueError(f"{name} indices must be 1D")
        if idx.size == 0:
            raise ValueError(f"{name} split is empty")
        if idx.min() < 0:
            raise ValueError(f"{name} indices contain negative values")
        if idx.max() >= n_filtered:
            raise ValueError(
                f"{name} indices out of range: max={idx.max()} >= n_filtered={n_filtered}"
            )

    _check_split("train", idx_train)
    _check_split("val", idx_val)
    _check_split("test", idx_test)

    # ---- datasets ----
    def _ds(split_idx: np.ndarray) -> WindowedDataset:
        return WindowedDataset(
            X=X_used,
            y=y_all,
            indices=split_idx,
            task_type=task_type,
            feature_indices=feature_indices_for_ds,
            row_indices=row_indices,
        )

    train_ds = _ds(idx_train)
    val_ds = _ds(idx_val)
    test_ds = _ds(idx_test)

    # ---- loader kwargs ----
    nw = int(num_workers)
    pin = bool(pin_memory) and (device.type == "cuda")

    loader_kwargs: Dict[str, Any] = dict(
        batch_size=int(batch_size),
        num_workers=nw,
        persistent_workers=(nw > 0),
        pin_memory=pin,
        drop_last=False,
    )
    if nw > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
