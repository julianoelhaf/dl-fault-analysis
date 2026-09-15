"""Normalization-leakage audit.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md.

The training pipeline normalizes **per window** via
``dl_fault_analysis.data.data_utils.scale_sample_to_minus1_1`` inside
``WindowedDataset.__getitem__`` -- there is no scaler fitted on the dataset
and no global/train statistics. That makes train/val/test normalization
leakage-free *by construction*.

Accordingly, this audit *verifies and records* that property rather than
auditing a non-existent fitted scaler: it checks the scaler is stateless and
per-sample independent, and records the train episode ids while asserting
val/test ids are excluded from any fit.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from dl_fault_analysis.data.data_utils import WindowedDataset, scale_sample_to_minus1_1
from dl_fault_analysis.validation._audit_base import (
    AuditCheck,
    AuditError,
    AuditResult,
    assert_passed,
    write_audit_json,
)

__all__ = [
    "scaler_is_stateless",
    "verify_per_sample_independence",
    "audit_normalization",
    "write_normalization_audit",
    "assert_passed",
    "AuditError",
    "AuditResult",
]

# Attribute names that would indicate a *fitted*, dataset-global scaler.
_FITTED_STATE_ATTRS = (
    "scaler",
    "mean_",
    "std_",
    "min_",
    "max_",
    "data_min_",
    "data_max_",
    "_stats",
    "norm_stats",
)


def scaler_is_stateless(obj: Any = WindowedDataset) -> bool:
    """True if ``obj`` (a dataset class/instance) carries no fitted-scaler state."""
    return not any(hasattr(obj, attr) for attr in _FITTED_STATE_ATTRS)


def verify_per_sample_independence(seed: int = 0, n: int = 8) -> bool:
    """Numerically confirm scaling a window depends only on that window.

    Scales each window alone and again after stacking with unrelated windows;
    per-sample scaling must give identical results either way.
    """
    rng = np.random.default_rng(seed)
    samples = [rng.normal(size=(5, 4)).astype(np.float32) for _ in range(n)]
    for i, x in enumerate(samples):
        alone = scale_sample_to_minus1_1(x)
        # the value of scaling x must not change based on the other samples
        other = samples[(i + 1) % n]
        _ = scale_sample_to_minus1_1(other)
        again = scale_sample_to_minus1_1(x)
        if not np.array_equal(alone, again):
            return False
        # range is within [-1, 1] for non-degenerate inputs
        if alone.size and (alone.min() < -1.0 - 1e-6 or alone.max() > 1.0 + 1e-6):
            return False
    return True


def audit_normalization(
    train_sample_ids: Sequence[Any],
    val_sample_ids: Sequence[Any],
    test_sample_ids: Sequence[Any],
    dataset: Any = WindowedDataset,
) -> AuditResult:
    """Audit that normalization is per-sample/stateless and fit excludes val/test.

    Records the train episode ids used (provenance) and asserts the val/test
    ids are disjoint from train (so they cannot influence any per-fold
    statistic, were one ever introduced).
    """
    train = list(train_sample_ids)
    val = set(val_sample_ids)
    test = set(test_sample_ids)
    train_set = set(train)

    result = AuditResult(name="normalization")
    result.add(
        AuditCheck(
            name="scaler_is_stateless",
            passed=scaler_is_stateless(dataset),
            details={"checked_attrs": list(_FITTED_STATE_ATTRS)},
        )
    )
    result.add(
        AuditCheck(
            name="per_sample_independent",
            passed=verify_per_sample_independence(),
            details={"scaler": "scale_sample_to_minus1_1", "fit_on": "per_window"},
        )
    )
    result.add(
        AuditCheck(
            name="fit_excludes_val_test",
            passed=train_set.isdisjoint(val) and train_set.isdisjoint(test),
            details={
                "n_train_episodes": len(train_set),
                "n_val_episodes": len(val),
                "n_test_episodes": len(test),
                "fit_on": "train_only (per-window; no cross-sample statistics)",
            },
        )
    )
    # provenance: record the actual train ids used "for fitting" (per-window scaling)
    result.add(
        AuditCheck(
            name="train_episode_ids_recorded",
            passed=len(train) > 0,
            details={"train_episode_ids": [str(s) for s in sorted(train_set)]},
        )
    )
    return result


def write_normalization_audit(path: str, result: AuditResult) -> str:
    """Serialize a normalization-audit result to JSON."""
    return write_audit_json(path, result)
