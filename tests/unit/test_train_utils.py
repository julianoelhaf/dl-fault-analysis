"""Regression test for the group-size min/max logging in split_indices().

Guards against a bug where ``np.unique(group_ids, return_counts=True)[1]``
(counts sorted by group *id*) was indexed with ``[[0, -1]]`` and logged as
"min"/"max" group size. That reports the counts of whichever group id sorts
first/last, not the true smallest/largest group -- silently wrong numbers
rather than a crash. The fix sorts the counts themselves before taking
min/max.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from dl_fault_analysis.utils.train_utils import split_indices


def _make_uneven_groups() -> tuple[pd.Series, pd.DataFrame]:
    # Group ids intentionally NOT sorted by size: the smallest group has the
    # highest id and the largest group has a middling id, so indexing counts
    # sorted-by-id at [0]/[-1] would not recover the true min/max.
    group_sizes = {0: 5, 1: 20, 2: 8, 3: 2, 4: 12}
    group_ids = np.concatenate(
        [np.full(size, gid) for gid, size in group_sizes.items()]
    )
    rng = np.random.default_rng(0)
    rng.shuffle(group_ids)
    groups = pd.Series(group_ids, name="sample_id")
    labels_df = pd.DataFrame({"y_fault_present": rng.integers(0, 2, size=len(groups))})
    return groups, labels_df


def test_group_size_log_reports_true_min_max(caplog: pytest.LogCaptureFixture) -> None:
    groups, labels_df = _make_uneven_groups()

    with caplog.at_level(logging.DEBUG, logger="dl_fault_analysis.utils.train_utils"):
        split_indices(
            groups=groups,
            labels_df=labels_df,
            target_label="y_fault_present",
            test_size=0.2,
            val_size=0.2,
            random_state=0,
            stratify=False,
        )

    messages = [r.getMessage() for r in caplog.records if "Group size statistics" in r.getMessage()]
    assert messages, "expected a 'Group size statistics' debug log line"
    # True min=2 (group 3), true max=20 (group 1) -- not the id-sorted-first/last values.
    assert "min=2" in messages[0]
    assert "max=20" in messages[0]
