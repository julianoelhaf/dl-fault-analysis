"""Episode-level leakage audit for CV / line-conditioned splits.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md.

Formalizes and *persists* the disjointness guarantees that the training
pipeline otherwise only asserts inline (see ``utils/train_utils.py``
``split_indices``).

Leakage rules enforced:
* an ``episode_id`` (``sample_id``) appears in at most one of train/val/test;
* all windows of an episode share one outer fold;
* per-line (line-conditioned) splits are themselves episode-disjoint.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from dl_fault_analysis.validation._audit_base import (
    AuditCheck,
    AuditError,
    AuditResult,
    assert_passed,
    write_audit_json,
)

__all__ = [
    "audit_episode_disjointness",
    "audit_window_grouping",
    "audit_line_conditioned_splits",
    "write_split_audit",
    "assert_passed",
    "AuditError",
    "AuditResult",
]


def _as_group_ids(meta: Any, group_col: str = "sample_id") -> np.ndarray:
    """Coerce a DataFrame (use ``group_col``) or an array-like into a 1D id array."""
    if isinstance(meta, pd.DataFrame):
        if group_col not in meta.columns:
            raise ValueError(f"group column '{group_col}' missing from metadata frame")
        return meta[group_col].to_numpy()
    return np.asarray(meta)


def audit_episode_disjointness(
    train_meta: Any,
    val_meta: Any,
    test_meta: Any,
    group_col: str = "sample_id",
) -> AuditResult:
    """Check pairwise episode disjointness across train/val/test."""
    train = set(_as_group_ids(train_meta, group_col).tolist())
    val = set(_as_group_ids(val_meta, group_col).tolist())
    test = set(_as_group_ids(test_meta, group_col).tolist())

    result = AuditResult(name="episode_disjointness")
    for a_name, a, b_name, b in (
        ("train", train, "val", val),
        ("train", train, "test", test),
        ("val", val, "test", test),
    ):
        overlap = sorted(a & b)
        result.add(
            AuditCheck(
                name=f"{a_name}_vs_{b_name}_disjoint",
                passed=len(overlap) == 0,
                details={
                    "n_overlap": len(overlap),
                    "overlap_examples": overlap[:10],
                    f"n_{a_name}": len(a),
                    f"n_{b_name}": len(b),
                },
            )
        )
    return result


def audit_window_grouping(
    metadata: pd.DataFrame,
    group_col: str = "sample_id",
    fold_col: str = "fold_id",
) -> AuditResult:
    """Check every episode's windows are assigned to exactly one outer fold."""
    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("audit_window_grouping requires a pandas DataFrame")
    for col in (group_col, fold_col):
        if col not in metadata.columns:
            raise ValueError(f"column '{col}' missing from metadata frame")

    folds_per_episode = metadata.groupby(group_col)[fold_col].nunique()
    offenders = folds_per_episode[folds_per_episode > 1]
    result = AuditResult(name="window_grouping")
    result.add(
        AuditCheck(
            name="windows_share_one_fold",
            passed=len(offenders) == 0,
            details={
                "n_episodes": int(folds_per_episode.shape[0]),
                "n_offending_episodes": int(offenders.shape[0]),
                "offending_examples": [str(x) for x in offenders.index[:10].tolist()],
            },
        )
    )
    return result


def audit_line_conditioned_splits(
    splits: Iterable[Mapping[str, Any]],
    group_col: str = "sample_id",
) -> AuditResult:
    """Audit per-line, per-fold splits for episode disjointness.

    ``splits`` is an iterable of records, each with keys ``line``, ``fold``
    and ``train``/``val``/``test`` (array-likes of episode ids). Each record
    must be train/val/test disjoint; additionally, no episode may appear
    under two different lines (a line-conditioned dataset must partition
    episodes by line).
    """
    result = AuditResult(name="line_conditioned_splits")
    episode_to_line: dict[Any, str] = {}
    cross_line_conflicts: list[dict[str, Any]] = []

    records = list(splits)
    if not records:
        result.add(
            AuditCheck(
                name="has_records", passed=False, details={"reason": "no splits given"}
            )
        )
        return result

    for rec in records:
        line = str(rec["line"])
        fold = rec.get("fold")
        sub = audit_episode_disjointness(
            rec["train"], rec["val"], rec["test"], group_col=group_col
        )
        result.add(
            AuditCheck(
                name=f"line={line}_fold={fold}_disjoint",
                passed=sub.passed,
                details={"checks": [c.to_dict() for c in sub.checks]},
            )
        )
        # episode<->line exclusivity (use the test set, which is unique per fold)
        for ep in set(_as_group_ids(rec["test"], group_col).tolist()):
            prev = episode_to_line.get(ep)
            if prev is not None and prev != line:
                cross_line_conflicts.append({"episode": str(ep), "lines": [prev, line]})
            episode_to_line[ep] = line

    result.add(
        AuditCheck(
            name="episodes_exclusive_to_one_line",
            passed=len(cross_line_conflicts) == 0,
            details={
                "n_conflicts": len(cross_line_conflicts),
                "examples": cross_line_conflicts[:10],
            },
        )
    )
    return result


def write_split_audit(path: str, result: AuditResult) -> str:
    """Serialize a split-audit result to JSON."""
    return write_audit_json(path, result)
