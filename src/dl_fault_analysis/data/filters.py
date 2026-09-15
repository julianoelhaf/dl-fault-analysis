# filters.py
from __future__ import annotations

import re
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import dl_fault_analysis.data.labels as L

LINE_ID_PATTERN = re.compile(r"^Ln\d+-\d+[A-Z]?$")
LINE_OR_MAINBUS_PATTERN = re.compile(r"^(Ln\d+-\d+[A-Z]?|MainBus\d+)$")


Condition = Union[object, Callable[[object], bool]]

# =========================
# Filtering rules (row-level validity)
# =========================

FILTER_RULES: dict[str, list[tuple[str, Condition]]] = {
    L.Y_FAULT_LINE: [  # keep Lines and MainBuses only
        (L.Y_FAULT_PRESENT, 1),
        (
            L.Y_FAULT_LINE,
            lambda v: isinstance(v, str) and bool(LINE_OR_MAINBUS_PATTERN.match(v)),
        ),
    ],
    L.Y_FAULT_LOCATION: [  # TODo:filter for Lines only
        (L.Y_FAULT_PRESENT, 1),
        (
            L.Y_FAULT_LINE,
            lambda v: isinstance(v, str) and bool(LINE_ID_PATTERN.match(v)),
        ),
    ],
    L.Y_FAULT_CATEGORY: [
        (L.Y_FAULT_PRESENT, 1),
    ],
    L.Y_SWITCH_CATEGORY: [
        (L.Y_SWITCH_PRESENT, 1),
    ],
    L.Y_SWITCH_TARGET_BUS: [
        (L.Y_SWITCH_PRESENT, 1),
    ],
    L.Y_FAULT_PRESENT: [  # remove filter after new version of dataset
        (
            L.Y_FAULT_LINE,
            lambda v: isinstance(v, str) and bool(LINE_OR_MAINBUS_PATTERN.match(v)),
        ),
    ],
    L.EVENT_TYPE: [  # remove filter after new version of dataset
        (
            L.Y_FAULT_LINE,
            lambda v: isinstance(v, str) and bool(LINE_OR_MAINBUS_PATTERN.match(v)),
        ),
    ],
}


def build_valid_row_indices(labels_df: pd.DataFrame, target_label: str) -> np.ndarray:
    """
    Return row indices of samples valid for the given target_label.
    Pure index logic -- memmap-safe.
    Supports equality and callable predicates in FILTER_RULES.
    """
    mask = np.ones(len(labels_df), dtype=bool)

    for col, condition in FILTER_RULES.get(target_label, []):
        if col not in labels_df.columns:
            raise ValueError(
                f"Target '{target_label}' requires column '{col}', "
                f"but it is missing from labels_df"
            )

        values = labels_df[col].to_numpy()

        if callable(condition):
            mask &= np.fromiter(
                (bool(condition(v)) for v in values), dtype=bool, count=len(values)
            )
        else:
            mask &= values == condition

    return np.flatnonzero(mask).astype(np.int64)


# =========================
# Stratification rules (episode-level)
# =========================
#
# IMPORTANT:
# - Stratification happens at EPISODE level (grouped by sample_id).
# - For window-varying targets (fault/event detection), stratifying (if used)
#   balances *scenarios* (e.g., event_type), not the 0/1 window label ratio.

# Each target_label -> None | column name | tuple of column names (composite)
STRAT_RULES: dict[str, Optional[Union[str, Tuple[str, ...]]]] = {
    # ---- classification (episode-constant) ----
    L.EVENT_TYPE: L.EVENT_TYPE,
    L.Y_FAULT_CATEGORY: L.EVENT_TYPE,
    L.Y_FAULT_LINE: L.Y_FAULT_LINE,
    # ---- regression (optional: balance by line) ----
    L.Y_FAULT_LOCATION: L.Y_FAULT_LINE,
    # ---- detection (window-varying; optional scenario balancing) ----
    L.Y_FAULT_PRESENT: L.EVENT_TYPE,
    # If you later include switching + normal episodes, consider (EVENT_FAMILY, EVENT_TYPE)
    # for detection tasks:
    # L.Y_EVENT_PRESENT: (L.EVENT_FAMILY, L.EVENT_TYPE),
    L.Y_EVENT_PRESENT: L.EVENT_TYPE,
}


def get_strat_key(target_label: str) -> Optional[Union[str, Tuple[str, ...]]]:
    """Return the episode-level stratification key for a target_label, or None."""
    return STRAT_RULES.get(target_label, None)


def build_group_strat_labels(
    labels_df: pd.DataFrame,
    groups: pd.Series,  # sample_id per row/window
    target_label: str,
) -> Optional[np.ndarray]:
    """
    Build episode-level stratification labels aligned with unique groups.

    Returns:
        None if STRAT_RULES[target_label] is None, else a 1D array of length n_groups.

    Notes:
        - Asserts the strat columns are constant within each group (episode).
        - For composite strat keys, returns a string key like "family|type".
    """
    strat_key = get_strat_key(target_label)
    if strat_key is None:
        return None

    group_ids = groups.to_numpy()
    unique_groups = np.unique(group_ids)
    assert unique_groups.size >= 2, "need at least two groups for stratification"

    # Normalize strat_key to tuple[str, ...]
    if isinstance(strat_key, str):
        strat_cols: Tuple[str, ...] = (strat_key,)
    else:
        strat_cols = strat_key

    for col in strat_cols:
        if col not in labels_df.columns:
            raise ValueError(f"Stratification column '{col}' missing from labels_df")

    # Build per-row composite key if needed
    if len(strat_cols) == 1:
        per_row = labels_df[strat_cols[0]].to_numpy()
        out_dtype = per_row.dtype
    else:
        # string composite (safe + stable for sklearn stratify)
        per_row = (
            labels_df.loc[:, list(strat_cols)]
            .astype(str)
            .agg("|".join, axis=1)
            .to_numpy()
        )
        out_dtype = per_row.dtype

    strat = np.empty(unique_groups.size, dtype=out_dtype)

    for i, g in enumerate(unique_groups):
        vg = per_row[group_ids == g]
        if vg.size == 0:
            raise ValueError(f"Group '{g}' has no rows in labels_df")

        first = vg[0]
        if not np.all(vg == first):
            raise ValueError(
                f"Strat key {strat_cols} is not constant within group '{g}'"
            )
        strat[i] = first

    return strat


# =========================
# LEGACY CODE for hv_double_line_90kv topology (custom filtering logic)
# ========================
FILTER_RULES_LEGACY: dict[str, list[tuple[str, Condition]]] = {
    L.Y_FAULT_LINE: [
        (L.STATUS, "fault_start"),
    ],
    L.Y_FAULT_LOCATION: [
        (L.STATUS, "fault_start"),
    ],
    L.Y_FAULT_PRESENT: [],  # no additional filter (keep all windows)
    L.Y_FAULT_CLASS: [],  # no additional filter (keep all windows)
}


def build_valid_row_indices_hv_double_line_90kv(
    labels_df: pd.DataFrame, target_label: str
) -> np.ndarray:
    """
    Custom filtering for hv_double_line_90kv topology.
    """
    valid_target_labels = {
        L.Y_FAULT_LINE,  # Fault line identification
        L.Y_FAULT_LOCATION,  # Fault localization
        L.Y_FAULT_PRESENT,  # Fault detection
        L.Y_FAULT_CLASS,  # Fault classification
    }
    if target_label not in valid_target_labels:
        raise ValueError(
            f"Target '{target_label}' is not supported for hv_double_line_90kv topology. "
            f"Supported targets: {valid_target_labels}"
        )

    mask = np.ones(len(labels_df), dtype=bool)

    for col, condition in FILTER_RULES_LEGACY.get(target_label, []):
        if col not in labels_df.columns:
            raise ValueError(
                f"Target '{target_label}' requires column '{col}', "
                f"but it is missing from labels_df"
            )

        values = labels_df[col].to_numpy()

        if callable(condition):
            mask &= np.fromiter(
                (bool(condition(v)) for v in values), dtype=bool, count=len(values)
            )
        else:
            mask &= values == condition

    return np.flatnonzero(mask).astype(np.int64)
