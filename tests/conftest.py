"""Shared pytest fixtures for the reviewer-added revision-experiment tests.

Synthesizes the PROTECT-90 / hv_double_line_90kv feature-name layout so unit
tests need no access to the real (multi-GB) dataset. Ported from the private
development repository's EPSR reviewer-response branch -- see
docs/PROVENANCE.md.
"""

from __future__ import annotations

import pytest

# Canonical measurement points of the 90 kV double-line topology, in the real
# on-disk order. Each contributes 6 channels: cur L1/L2/L3 then vol L1/L2/L3.
_POINTS = [
    (1, 1, 2, "A"),
    (1, 1, 2, "B"),
    (2, 1, 2, "A"),
    (2, 1, 2, "B"),
    (2, 2, 3, "A"),
    (2, 2, 3, "B"),
    (3, 2, 3, "A"),
    (3, 2, 3, "B"),
]

# Faulted-line classes (label form) present in the real labels.
FAULT_LINES = ["Line_1_2_a", "Line_1_2_b", "Line_2_3_a", "Line_2_3_b"]


def _build_feature_names() -> list[str]:
    names: list[str] = []
    for bus, frm, to, circ in _POINTS:
        for qty, unit in (("cur", "A"), ("vol", "V")):
            for phase in (1, 2, 3):
                names.append(
                    f"Bus_{bus}_Line_{frm:02d}_{to:02d}{circ}_{qty}_L{phase}_{unit}"
                )
    return names


@pytest.fixture
def feature_names_48() -> list[str]:
    """The 48 feature names of the 90 kV layout (8 points x 6 channels)."""
    names = _build_feature_names()
    assert len(names) == 48
    return names
