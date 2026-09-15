# task_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Type

import torch.nn as nn

import dl_fault_analysis.data.labels as L


@dataclass(frozen=True)
class TaskSpec:
    """
    Single source of truth for training semantics of a target label.

    Notes:
    - For CrossEntropyLoss, targets must be integer class indices (LongTensor).
      If the raw label column is a string (target_dtype=str), you must encode it
      before passing it to the criterion (e.g., via a label encoder).
    """

    criterion: nn.Module
    primary_metric: str
    higher_is_better: bool
    log_msg: str
    target_dtype: Type


# Canonical target label -> training spec
TASK_SPEC: Dict[str, TaskSpec] = {
    # ---- Binary detection ----
    L.Y_FAULT_PRESENT: TaskSpec(
        criterion=nn.BCEWithLogitsLoss(),
        primary_metric="f1",
        higher_is_better=True,
        log_msg="Binary classification task",
        target_dtype=int,
    ),
    L.Y_SWITCH_PRESENT: TaskSpec(
        criterion=nn.BCEWithLogitsLoss(),
        primary_metric="f1",
        higher_is_better=True,
        log_msg="Binary classification task",
        target_dtype=int,
    ),
    # ---- Multi-class classification (raw labels may be strings) ----
    L.Y_FAULT_LINE: TaskSpec(
        criterion=nn.CrossEntropyLoss(),
        primary_metric="f1_macro",
        higher_is_better=True,
        log_msg="Multi-class classification task",
        target_dtype=str,
    ),
    L.Y_SWITCH_CATEGORY: TaskSpec(
        criterion=nn.CrossEntropyLoss(),
        primary_metric="f1_macro",
        higher_is_better=True,
        log_msg="Multi-class classification task",
        target_dtype=str,
    ),
    L.Y_SWITCH_TARGET_BUS: TaskSpec(
        criterion=nn.CrossEntropyLoss(),
        primary_metric="f1_macro",
        higher_is_better=True,
        log_msg="Multi-class classification task",
        target_dtype=str,
    ),
    L.Y_SWITCH_ACTION: TaskSpec(
        criterion=nn.CrossEntropyLoss(),
        primary_metric="f1_macro",
        higher_is_better=True,
        log_msg="Multi-class classification task",
        target_dtype=str,
    ),
    L.Y_FAULT_CLASS: TaskSpec(
        criterion=nn.CrossEntropyLoss(),
        primary_metric="f1_macro",
        higher_is_better=True,
        log_msg="Multi-class classification task",
        target_dtype=str,
    ),
    L.EVENT_TYPE: TaskSpec(
        criterion=nn.CrossEntropyLoss(),
        primary_metric="f1_macro",
        higher_is_better=True,
        log_msg="Multi-class classification task",
        target_dtype=str,
    ),
    # ---- Regression ----
    L.Y_FAULT_LOCATION: TaskSpec(
        criterion=nn.MSELoss(),
        primary_metric="mae",
        higher_is_better=False,
        log_msg="Regression task",
        target_dtype=float,
    ),
}


def get_task_spec(target_label: str) -> TaskSpec:
    """Return the TaskSpec for a given target label (raises a clear error if missing)."""
    try:
        return TASK_SPEC[target_label]
    except KeyError as exc:
        raise ValueError(
            f"No TaskSpec defined for target_label '{target_label}'. "
            f"Expected one of: {sorted(TASK_SPEC.keys())}."
        ) from exc


def infer_task_type_from_spec(spec: TaskSpec) -> str:
    """
    Infer task type from the criterion in TaskSpec.
    Keeps logic centralized so scripts don't duplicate it.
    """
    if isinstance(spec.criterion, nn.BCEWithLogitsLoss):
        return "binary"
    if isinstance(spec.criterion, nn.CrossEntropyLoss):
        return "multiclass"
    return "regression"


def get_training_setup(target_label: str) -> Tuple[nn.Module, str, bool, str]:
    """
    Convenience helper: criterion_fn, primary_metric, higher_is_better, log_msg.
    """
    spec = get_task_spec(target_label)
    return spec.criterion, spec.primary_metric, spec.higher_is_better, spec.log_msg


def get_target_dtype(target_label: str) -> Type:
    """Convenience helper: expected dtype for the raw label column."""
    return get_task_spec(target_label).target_dtype
