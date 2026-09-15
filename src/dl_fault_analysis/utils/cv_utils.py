"""Episode-grouped cross-validation helpers.

Factored out of `scripts/run_dl_experiment.py` and `utils/tuning_utils.py`,
which each defined an identical copy of this logic. Behavior-preserving
refactor only -- see docs/PROVENANCE.md and docs/REPRODUCIBILITY.md for the
CV-split reproducibility limitation (the exact published episode<->fold
assignment was not recoverable; only this algorithm and seed are frozen).
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from dl_fault_analysis.utils.logging import get_logger

logger = get_logger(__name__)


def build_cv_splits_stratified(
    y_all: np.ndarray,
    groups_np: np.ndarray,
    task_type: str,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build episode-grouped outer CV splits.

    Multiclass tasks (FC, FLI) use `StratifiedGroupKFold` so each fold has a
    similar class distribution; binary/regression tasks (FD, FL) use
    `GroupKFold`. Both group by episode (`sample_id`), never by window, so no
    episode's windows can straddle a train/test boundary.

    Args:
        y_all: Labels (already canonically encoded for multiclass).
        groups_np: Group assignments (episode/sample IDs).
        task_type: "binary", "multiclass", or "regression".
        n_splits: Number of CV folds.
        seed: Random seed for shuffling.

    Returns:
        List of (train_idx, test_idx) tuples.
    """
    idx_all = np.arange(len(y_all), dtype=int)

    if task_type == "multiclass":
        logger.info(
            "Using StratifiedGroupKFold (stratified by class) for %s task", task_type
        )
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        splits = list(sgkf.split(idx_all, y=y_all, groups=groups_np))
    else:
        logger.info("Using GroupKFold (not stratified) for %s task", task_type)
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(idx_all, y=None, groups=groups_np))

    logger.info("Built %d CV splits with %s", len(splits), type(splits[0]).__name__)
    return splits


def split_train_val_from_train_pool(
    groups_used: Union[pd.Series, np.ndarray],
    train_pool_idx: np.ndarray,
    val_size: float = 0.2,
    split_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Group-safe (episode-safe) train/val split within a CV fold's train pool."""
    if isinstance(groups_used, pd.Series):
        pool_groups = groups_used.iloc[train_pool_idx]
    else:
        pool_groups = groups_used[train_pool_idx]

    uniq = np.unique(pool_groups)
    rng = np.random.default_rng(int(split_seed))
    rng.shuffle(uniq)

    n_val_groups = max(1, int(round(float(val_size) * len(uniq))))
    val_groups = set(uniq[:n_val_groups])

    def _group_of(i: int):
        return groups_used.iloc[i] if isinstance(groups_used, pd.Series) else groups_used[i]

    idx_val = [i for i in train_pool_idx if _group_of(i) in val_groups]
    idx_val_set = set(idx_val)
    idx_train = [i for i in train_pool_idx if i not in idx_val_set]
    return np.asarray(idx_train, dtype=int), np.asarray(idx_val, dtype=int)
