"""Split-integrity tests for the episode-grouped 5-fold CV mechanism.

These validate the *mechanism* frozen in config/paper/splits/SPLIT_LIMITATION.md
(StratifiedGroupKFold/GroupKFold, seed 42, grouped by episode) -- not that it
reproduces the exact published fold assignment, which is not recoverable (see
that file and docs/PROVENANCE.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from dl_fault_analysis.utils.cv_utils import (
    build_cv_splits_stratified,
    split_train_val_from_train_pool,
)

N_EPISODES = 200
WINDOWS_PER_EPISODE = 4
N_SPLITS = 5
SEED = 42


def _make_grouped_data(task_type: str):
    rng = np.random.default_rng(SEED)
    episode_ids = np.repeat(np.arange(N_EPISODES), WINDOWS_PER_EPISODE)
    if task_type == "multiclass":
        # one class per episode (constant within an episode), 4 classes
        episode_classes = rng.integers(0, 4, size=N_EPISODES)
        y = np.repeat(episode_classes, WINDOWS_PER_EPISODE)
    else:
        y = rng.standard_normal(episode_ids.shape[0])
    return y, episode_ids


@pytest.mark.parametrize("task_type", ["multiclass", "binary", "regression"])
def test_every_episode_is_a_test_episode_exactly_once(task_type):
    y, groups = _make_grouped_data(task_type)
    splits = build_cv_splits_stratified(y, groups, task_type, N_SPLITS, SEED)

    assert len(splits) == N_SPLITS

    test_episode_counts: dict[int, int] = {}
    for _, test_idx in splits:
        for ep in np.unique(groups[test_idx]):
            test_episode_counts[ep] = test_episode_counts.get(ep, 0) + 1

    assert set(test_episode_counts) == set(np.unique(groups))
    assert all(count == 1 for count in test_episode_counts.values())


@pytest.mark.parametrize("task_type", ["multiclass", "binary", "regression"])
def test_no_train_test_episode_overlap(task_type):
    y, groups = _make_grouped_data(task_type)
    splits = build_cv_splits_stratified(y, groups, task_type, N_SPLITS, SEED)

    for train_idx, test_idx in splits:
        train_episodes = set(np.unique(groups[train_idx]))
        test_episodes = set(np.unique(groups[test_idx]))
        assert train_episodes.isdisjoint(test_episodes)


def test_split_train_val_from_train_pool_is_group_safe():
    _, groups = _make_grouped_data("regression")
    train_pool_idx = np.arange(len(groups))

    idx_train, idx_val = split_train_val_from_train_pool(
        groups_used=groups, train_pool_idx=train_pool_idx, val_size=0.2, split_seed=SEED
    )

    train_episodes = set(np.unique(groups[idx_train]))
    val_episodes = set(np.unique(groups[idx_val]))
    assert train_episodes.isdisjoint(val_episodes)
    assert len(idx_train) + len(idx_val) == len(train_pool_idx)


def test_multiclass_split_is_deterministic_for_a_fixed_seed():
    y, groups = _make_grouped_data("multiclass")
    splits_a = build_cv_splits_stratified(y, groups, "multiclass", N_SPLITS, SEED)
    splits_b = build_cv_splits_stratified(y, groups, "multiclass", N_SPLITS, SEED)

    for (train_a, test_a), (train_b, test_b) in zip(splits_a, splits_b):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)
