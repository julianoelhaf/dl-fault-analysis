"""Shared primitives for leakage/reproducibility audits.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md.

A small, dependency-light result model so audits can both (a) be asserted on
(fail loudly) and (b) be serialized to JSON for the run record.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


class AuditError(AssertionError):
    """Raised when an audit that must pass does not."""


@dataclass
class AuditCheck:
    name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "passed": bool(self.passed), "details": self.details}


@dataclass
class AuditResult:
    name: str
    checks: List[AuditCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) and len(self.checks) > 0

    def add(self, check: AuditCheck) -> "AuditResult":
        self.checks.append(check)
        return self

    def failed_checks(self) -> List[AuditCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit": self.name,
            "passed": self.passed,
            "n_checks": len(self.checks),
            "checks": [c.to_dict() for c in self.checks],
        }


def assert_passed(result: AuditResult) -> None:
    """Raise :class:`AuditError` if any check in ``result`` failed."""
    if not result.passed:
        failed = ", ".join(c.name for c in result.failed_checks()) or "<no checks>"
        raise AuditError(f"audit '{result.name}' failed: {failed}")


def write_audit_json(path: str, result: AuditResult) -> str:
    """Write ``result`` to ``path`` as pretty JSON (deterministic, sorted keys)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)
    return path
