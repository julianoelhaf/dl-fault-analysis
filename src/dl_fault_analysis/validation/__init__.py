"""Leakage / reproducibility audits for the reviewer-added revision experiments.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md.
"""

from dl_fault_analysis.validation._audit_base import (
    AuditCheck,
    AuditError,
    AuditResult,
    assert_passed,
    write_audit_json,
)
from dl_fault_analysis.validation.normalization_audit import (
    audit_normalization,
    write_normalization_audit,
)
from dl_fault_analysis.validation.split_audit import (
    audit_episode_disjointness,
    audit_line_conditioned_splits,
    audit_window_grouping,
    write_split_audit,
)

__all__ = [
    "AuditCheck",
    "AuditError",
    "AuditResult",
    "assert_passed",
    "write_audit_json",
    "audit_episode_disjointness",
    "audit_window_grouping",
    "audit_line_conditioned_splits",
    "write_split_audit",
    "audit_normalization",
    "write_normalization_audit",
]
