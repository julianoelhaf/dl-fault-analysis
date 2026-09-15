"""Fault-label vocabulary and output-dimension map.

Vendored (and trimmed) from the private `psp_helper` package's
`constants.py` -- see docs/PROVENANCE.md for the full rationale.

The upstream `FAULT_LABEL_TO_ID` dict has 20 entries (ids 0-19): the 11
short-circuit fault types the PROTECT-90 dataset actually contains (ids
0-10, matching the paper's fault-classification vocabulary {c0, ..., c10}),
plus 9 arc-fault/incipient-fault ids (11-19) that PROTECT-90's PowerFactory
pipeline never emits (only `sc_type` in {0,1,2,3} -> flt_3ph_shc /
flt_2ph_shc / flt_1phg_shc / flt_2phg_shc are produced; arc and incipient
event types are a different dataset family). Only the reachable ids are
vendored here, so the label vocabulary in this repository cannot silently
grow beyond what PROTECT-90 can produce.
"""

from __future__ import annotations

FAULT_LABEL_TO_ID = {
    "no_fault": 0,
    # ---- 1-phase-to-ground ----
    "flt_1phg_shc_AG": 1,
    "flt_1phg_shc_BG": 2,
    "flt_1phg_shc_CG": 3,
    # ---- 2-phase (no ground) ----
    "flt_2ph_shc_AB": 4,
    "flt_2ph_shc_AC": 5,
    "flt_2ph_shc_BC": 6,
    # ---- 2-phase-to-ground ----
    "flt_2phg_shc_ABG": 7,
    "flt_2phg_shc_ACG": 8,
    "flt_2phg_shc_BCG": 9,
    # ---- 3-phase ----
    "flt_3ph_shc_ABC": 10,
}

FAULT_ID_TO_LABEL = {v: k for k, v in FAULT_LABEL_TO_ID.items()}

# Fixed output dimensions for targets whose class count doesn't need to be
# derived from the data at runtime. Deliberately has NO entry for
# "y_fault_class" (FC): its output dimension is always derived dynamically
# from the realized label set (see models/model_utils.create_model_from_name
# and data/targets.py), which is why FC's output dimension is 11 -- matching
# len(FAULT_LABEL_TO_ID) above -- rather than a value hardcoded here.
FAULT_TARGET_TO_OUTPUT_DIM = {
    "y_fault_present": 1,  # Binary classification: fault / no fault
    "y_event_present": 1,  # Binary classification: event / no event
    "y_fault_location": 1,  # Regression: fault location (% of line length)
    "y_fault_line": 4,  # Multi-class classification: fault on which line
}
