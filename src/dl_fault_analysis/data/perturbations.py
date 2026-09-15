"""Deterministic test-time input perturbations for the minimal
communication-perturbation experiment.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md. Manuscript Section 3.6 / Table 9.

Each perturbation maps a single raw window ``x`` of shape ``(T, F)`` to a
perturbed window of the **same shape**, applied **before** per-sample scaling
and **only at test time** (no retraining). Scope is exactly the three modes
below.

* ``none``                 -- identity (baseline).
* ``time_shift_2_samples`` -- emulate a small synchronisation error: shift the
                              waveform forward by 2 samples along time, edge-pad
                              the start, drop the last 2 samples.
* ``block_loss_10ms``      -- emulate a brief communication dropout: zero a
                              contiguous 10 ms block of timesteps across all
                              channels (fixed offset for determinism).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

PERTURBATIONS: Tuple[str, ...] = ("none", "time_shift_2_samples", "block_loss_10ms")

_TIME_SHIFT_SAMPLES = 2
_BLOCK_MS = 10.0  # milliseconds
_BLOCK_START_FRAC = 0.1  # block begins 10% into the window (deterministic)


def block_loss_n_samples(sampling_frequency: float) -> int:
    """Number of timesteps in a 10 ms block at the given sampling rate."""
    return max(1, int(round((_BLOCK_MS / 1000.0) * float(sampling_frequency))))


def apply_perturbation(
    x: np.ndarray,
    kind: str,
    sampling_frequency: Optional[float] = None,
) -> np.ndarray:
    """Return a perturbed copy of window ``x`` (shape ``(T, F)``) for ``kind``.

    Raises ``ValueError`` on an unknown kind, a non-2D window, or a missing
    ``sampling_frequency`` when one is required (fail loudly).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (T, F), got shape {x.shape}")
    t = x.shape[0]

    if kind == "none":
        return x.copy()

    if kind == "time_shift_2_samples":
        s = min(_TIME_SHIFT_SAMPLES, t)
        out = np.empty_like(x)
        out[:s] = x[0]  # edge-pad the start with the first sample
        out[s:] = x[: t - s]
        return out

    if kind == "block_loss_10ms":
        if sampling_frequency is None:
            raise ValueError("block_loss_10ms requires sampling_frequency")
        n = min(block_loss_n_samples(sampling_frequency), t)
        start = min(int(round(_BLOCK_START_FRAC * t)), t - n)
        out = x.copy()
        out[start : start + n, :] = 0.0
        return out

    raise ValueError(f"unknown perturbation {kind!r}; expected one of {PERTURBATIONS}")
