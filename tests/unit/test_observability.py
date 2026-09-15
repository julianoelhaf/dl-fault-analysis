"""Unit tests for dl_fault_analysis.data.observability.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md.
"""

from __future__ import annotations

import pytest

from dl_fault_analysis.data import observability as obs


# ---------------------------------------------------------------------------
# Shape / count tests
# ---------------------------------------------------------------------------
def test_full_observability_shape(feature_names_48):
    idx = obs.select_observability(feature_names_48, "full")
    assert idx == list(range(48))
    obs.validate_observability_selection(idx, feature_names_48, "full")


def test_full_line_conditioned_uses_all_relays(feature_names_48):
    # control: per-line model but all 8 relays (48 channels)
    idx = obs.select_observability(
        feature_names_48, "full_line_conditioned", "Line_1_2_a"
    )
    assert idx == list(range(48))
    obs.validate_observability_selection(idx, feature_names_48, "full_line_conditioned")
    d = obs.describe_selection(feature_names_48, "full_line_conditioned", "Line_1_2_a")
    assert d["n_relays"] == 8 and d["n_channels"] == 48


def test_terminal_pair_shape(feature_names_48):
    idx = obs.select_observability(feature_names_48, "terminal_pair", "Line_1_2_a")
    # Bus_1 Line_01_02A (0-5) + Bus_2 Line_01_02A (12-17)
    assert idx == [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
    obs.validate_observability_selection(idx, feature_names_48, "terminal_pair")


def test_single_ended_shape(feature_names_48):
    idx_a = obs.select_observability(
        feature_names_48, "single_ended", "Line_1_2_a", terminal_side="a"
    )
    idx_b = obs.select_observability(
        feature_names_48, "single_ended", "Line_1_2_a", terminal_side="b"
    )
    assert idx_a == [0, 1, 2, 3, 4, 5]  # from-bus end (Bus_1)
    assert idx_b == [12, 13, 14, 15, 16, 17]  # to-bus end (Bus_2)
    obs.validate_observability_selection(idx_a, feature_names_48, "single_ended")
    obs.validate_observability_selection(idx_b, feature_names_48, "single_ended")


# ---------------------------------------------------------------------------
# Correctness: selection matches the faulted line
# ---------------------------------------------------------------------------
def test_selected_relays_match_faulted_line(feature_names_48):
    desc = obs.describe_selection(feature_names_48, "terminal_pair", "Line_2_3_b")
    assert desc["n_relays"] == 2
    assert set(desc["selected_relays"]) == {
        "Bus_2_Line_02_03B",
        "Bus_3_Line_02_03B",
    }
    assert desc["feature_indices"] == [30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47]
    # every selected channel name must belong to the faulted line circuit
    for i in desc["feature_indices"]:
        assert "Line_02_03B" in feature_names_48[i]


def test_normalize_line_token_label_and_feature_forms_agree():
    assert obs.normalize_line_token("Line_1_2_a") == obs.normalize_line_token(
        "Line_01_02A"
    )
    lc = obs.normalize_line_token("Line_2_3_b")
    assert (lc.line_from, lc.line_to, lc.circuit) == (2, 3, "B")


# ---------------------------------------------------------------------------
# Failure modes (fail loudly)
# ---------------------------------------------------------------------------
def test_invalid_faulted_line_raises(feature_names_48):
    # Parseable token but not in the layout
    with pytest.raises(ValueError):
        obs.select_observability(feature_names_48, "terminal_pair", "Line_7_8_a")
    # Unparseable token
    with pytest.raises(ValueError):
        obs.normalize_line_token("not_a_line")


def test_single_ended_requires_side(feature_names_48):
    with pytest.raises(ValueError):
        obs.select_observability(feature_names_48, "single_ended", "Line_1_2_a")


def test_terminal_pair_requires_line(feature_names_48):
    with pytest.raises(ValueError):
        obs.select_observability(feature_names_48, "terminal_pair")


def test_missing_metadata_raises():
    with pytest.raises(ValueError):
        obs.parse_feature_layout([])
    with pytest.raises(ValueError):
        obs.select_observability([], "full")


def test_unparseable_feature_name_raises():
    with pytest.raises(ValueError):
        obs.parse_feature_layout(["totally_wrong_name"])


def test_unknown_mode_raises(feature_names_48):
    with pytest.raises(ValueError):
        obs.select_observability(feature_names_48, "partial")  # type: ignore[arg-type]
