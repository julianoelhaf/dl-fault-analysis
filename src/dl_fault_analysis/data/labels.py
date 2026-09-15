# labels.py
from __future__ import annotations

# ---- Sample identifier ----
SAMPLE_ID = "sample_id"


# ---- Fault-related targets ----
Y_FAULT_CATEGORY = "y_fault_category"
Y_FAULT_CLASS = "y_fault_class"  # corresponds to the task: fault classification (window-level multi-class classification) | created from "event_type" and phase info
Y_FAULT_LINE = "y_fault_line"  # corresponds to the task: fault line identification (window-level multi-class classification; for fault windows only)
Y_FAULT_LOCATION = "y_fault_location"  # corresponds to the task: fault localization (window-level regression; for fault windows only)
Y_FAULT_PRESENT = "y_fault_present"  # corresponds to the task: fault detection (window-level binary classification)
Y_IS_GROUNDED = "y_is_grounded"

# ---- Switching-related targets ----
Y_SWITCH_ACTION = "y_switch_action"
Y_SWITCH_CATEGORY = "y_switch_category"
Y_SWITCH_PRESENT = "y_switch_present"
Y_SWITCH_TARGET_BUS = "y_switch_target_bus"

# ---- Event / temporal metadata ----
DT_START = "dt_start"
EVENT_FAMILY = "event_family"
EVENT_TYPE = "event_type"
OVERLAP_RATIO = "overlap_ratio"
STATUS = "status"
Y_EVENT_PRESENT = "y_event_present"

# ---- Phases -----
Y_PHASE_A = "y_phase_A"
Y_PHASE_B = "y_phase_B"
Y_PHASE_C = "y_phase_C"

ALL_TARGETS = {
    DT_START,
    EVENT_FAMILY,
    EVENT_TYPE,
    OVERLAP_RATIO,
    STATUS,
    Y_EVENT_PRESENT,
    Y_FAULT_CATEGORY,
    Y_FAULT_CLASS,
    Y_FAULT_LINE,
    Y_FAULT_LOCATION,
    Y_FAULT_PRESENT,
    Y_IS_GROUNDED,
    Y_SWITCH_ACTION,
    Y_SWITCH_CATEGORY,
    Y_SWITCH_PRESENT,
    Y_SWITCH_TARGET_BUS,
}


# Optional: public export list (helps static analysis)
__all__ = [
    "DT_START",
    "EVENT_FAMILY",
    "EVENT_TYPE",
    "OVERLAP_RATIO",
    "SAMPLE_ID",
    "STATUS",
    "Y_EVENT_PRESENT",
    "Y_FAULT_CATEGORY",
    "Y_FAULT_CLASS",
    "Y_FAULT_LINE",
    "Y_FAULT_LOCATION",
    "Y_FAULT_PRESENT",
    "Y_IS_GROUNDED",
    "Y_SWITCH_ACTION",
    "Y_SWITCH_CATEGORY",
    "Y_SWITCH_PRESENT",
    "Y_SWITCH_TARGET_BUS",
]
