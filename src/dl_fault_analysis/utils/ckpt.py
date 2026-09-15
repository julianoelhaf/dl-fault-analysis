"""Lean model checkpointing for the revision runners.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md.

Saves a per-fold ``state_dict`` plus a small metadata dict so a trained model
can be reloaded later for evaluation (e.g. the perturbation study reuses the
full-observability models instead of retraining). Files are written under a
``checkpoints/`` directory, which the repo's .gitignore already excludes.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Union

import torch
from torch import nn


def save_checkpoint(path: str, model: nn.Module, meta: Dict[str, Any]) -> str:
    """Save ``model.state_dict()`` + ``meta`` (cell tag, n_features, metrics, ...)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": dict(meta)}, path)
    return path


def load_into(
    model: nn.Module, path: str, device: Union[torch.device, str] = "cpu"
) -> Dict[str, Any]:
    """Load weights from ``path`` into ``model`` (in place); return saved meta.

    ``weights_only=False`` is passed explicitly rather than left to the
    PyTorch default, which changed to ``True`` in torch 2.6. These checkpoints
    are not weights-only: ``save_checkpoint`` stores a ``meta`` dict alongside
    the state dict, and its ``test_metrics`` values are typically numpy
    scalars, which the weights-only unpickler rejects. Relying on the default
    therefore works on torch < 2.6 and raises ``UnpicklingError`` on >= 2.6,
    while this repository declares only ``torch>=2.1``.

    This loads a pickle, so only use it on checkpoints this repository wrote
    (``checkpoints/``, which .gitignore excludes) -- never on a file from an
    untrusted source.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return payload.get("meta", {})
