# targets.py
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import dl_fault_analysis.data.labels as L
from dl_fault_analysis.constants import FAULT_LABEL_TO_ID
from dl_fault_analysis.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Target extraction + task inference
# =============================================================================
def encode_multiclass(y: np.ndarray) -> Tuple[np.ndarray, Dict[Any, int]]:
    """
    Encode multiclass string labels to integer indices.

    Args:
        y: Array of labels (strings or other).

    Returns:
        Tuple of (encoded_y, class_to_idx_dict).
    """
    try:
        classes = np.unique(y)
    except TypeError as exc:
        logger.error("Failed to encode multiclass targets. Raw y:")
        logger.error("%s", y)
        raise TypeError(
            "Failed to encode multiclass targets: invalid / NA values present."
        ) from exc

    class_to_idx: Dict[Any, int] = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[v] for v in y], dtype=np.int64)
    return y_idx, class_to_idx


def build_fault_label(
    event_type: str,
    a: bool,
    b: bool,
    c: bool,
    is_grounded: bool,
    status: str,
) -> str:
    # Any window not containing the fault start is treated as no_fault
    if status != "fault_start":
        return "no_fault"

    phases = ("A" if a else "") + ("B" if b else "") + ("C" if c else "")
    if not phases:
        raise ValueError(
            "status='fault_start' but no phase selected: "
            f"event_type={event_type}, A={a}, B={b}, C={c}"
        )

    ground = "G" if is_grounded else ""
    return f"{event_type}_{phases}{ground}"


def create_fault_classes(labels: pd.DataFrame) -> List[int]:
    required_cols = {
        L.EVENT_TYPE,
        L.Y_PHASE_A,
        L.Y_PHASE_B,
        L.Y_PHASE_C,
        L.Y_IS_GROUNDED,
        L.STATUS,
    }
    missing = required_cols - set(labels.columns)
    if missing:
        raise ValueError(
            f"Missing columns required for 'fault_class': {sorted(missing)}. "
            f"Found columns: {labels.columns.tolist()}"
        )

    cols = [
        L.EVENT_TYPE,
        L.Y_PHASE_A,
        L.Y_PHASE_B,
        L.Y_PHASE_C,
        L.Y_IS_GROUNDED,
        L.STATUS,
    ]

    fault_classes: List[int] = []
    for event_type, a, b, c, grounded, status in labels[cols].itertuples(
        index=False, name=None
    ):
        fault_label = build_fault_label(
            event_type=event_type,
            a=bool(a),
            b=bool(b),
            c=bool(c),
            is_grounded=bool(grounded),
            status=status,
        )

        try:
            fault_classes.append(FAULT_LABEL_TO_ID[fault_label])

        except KeyError as e:
            raise ValueError(
                f"Unknown fault label '{fault_label}'. "
                "Check FAULT_LABEL_TO_ID consistency."
            ) from e

    return fault_classes


def extract_target(
    labels_df: pd.DataFrame,
    target_label: str,
) -> Tuple[np.ndarray, Optional[Dict[Any, int]]]:
    """
    Extract and preprocess target labels from dataframe.

    - Multiclass string labels are encoded.
    - Integer labels returned as int64.
    - Floating labels returned as float32.
    - Special case: y_fault_location (given in percent 0..100) scaled to [0,1].

    Returns:
        (processed_y, class_to_idx_dict_or_none)
    """
    # If the target is 'y_fault_class', we need to build it from other columns first
    if target_label == L.Y_FAULT_CLASS:
        y_fault_class = create_fault_classes(labels_df)
        return np.array(y_fault_class, dtype=np.int64), None

    if target_label not in labels_df.columns:
        raise ValueError(
            f"target_label='{target_label}' not found. Available columns: {list(labels_df.columns)}"
        )

    y_col = labels_df[target_label].to_numpy()
    if y_col.ndim != 1:
        y_col = y_col.reshape(-1)

    # ---- Special case: fault location regression target ----
    if target_label == L.Y_FAULT_LOCATION:
        # convert to float
        y = y_col.astype(np.float32, copy=False)

        # keep NaNs as-is (filtering should remove invalid rows; subgroup dropna covers groups)
        finite = np.isfinite(y)
        if np.any(finite):
            y_min = float(np.min(y[finite]))
            y_max = float(np.max(y[finite]))

            # If values look like 0..100 (%), scale to 0..1.
            # If already in 0..1, keep as-is.
            # Otherwise, still scale by 100 if it clearly matches percent semantics.
            if y_max > 1.5:  # heuristic: not already normalized
                y = y / 100.0

        return y, None

    # ---- Existing logic ----
    if y_col.dtype.kind in {"O", "U", "S"}:
        y_idx, mapping = encode_multiclass(y_col)
        return y_idx, mapping

    if np.issubdtype(y_col.dtype, np.integer):
        return y_col.astype(np.int64, copy=False), None

    if np.issubdtype(y_col.dtype, np.floating):
        return y_col.astype(np.float32, copy=False), None

    y_idx, mapping = encode_multiclass(y_col.astype(str))
    return y_idx, mapping
