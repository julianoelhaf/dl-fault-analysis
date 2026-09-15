"""Classical impedance-based fault location (physical baseline for FL).

Ported from the private development repository's EPSR reviewer-response
branch (Reviewer #2.1) -- see docs/PROVENANCE.md for exact source commit and
integration notes. Manuscript Section 3.7 / Table 10.

Implements the two textbook impedance-based fault locators used as a
*physically grounded* comparison point for the data-driven models:

* ``double_ended_location`` -- synchronized two-terminal positive-sequence
  algorithm. Fault-resistance and infeed independent; the strong baseline.
* ``single_ended_reactance`` -- one-terminal reactance method with fault-loop
  selection (SLG with zero-sequence compensation; phase-phase otherwise).

Both require the line's series sequence impedance and length (the per-episode
"true" parameters from the dataset metadata) -- information the DL models never
receive. Phasors are extracted by a full-cycle DFT at the fundamental.

All distances are returned as a fraction of line length measured from the
``line_from`` (sending, S) terminal; multiply by 100 for "% of line length".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Symmetrical-component row vectors (a-operator). V1 = A1 . [Va,Vb,Vc], etc.
_A = np.exp(1j * 2 * np.pi / 3)
_A1 = np.array([1.0, _A, _A**2]) / 3.0
_A0 = np.array([1.0, 1.0, 1.0]) / 3.0


def fundamental_phasors(window: np.ndarray, samples_per_cycle: int) -> np.ndarray:
    """Per-phase fundamental phasors from the LAST full cycle of ``window``.

    ``window`` is ``(T, 3)`` real samples (one measurement quantity, 3 phases).
    Returns a length-3 complex array of peak-amplitude phasors. Using the last
    full cycle assumes the fault is developed there (see the runner's window
    selection).
    """
    w = np.asarray(window, dtype=float)
    n = int(samples_per_cycle)
    if w.shape[0] < n:
        raise ValueError(f"window has {w.shape[0]} samples < one cycle ({n})")
    seg = w[-n:]
    k = np.exp(-1j * 2 * np.pi * np.arange(n) / n)  # fundamental bin
    return (2.0 / n) * (seg.T @ k)


def positive_sequence(phasors_abc: np.ndarray) -> complex:
    """Positive-sequence component of a 3-phase phasor triple."""
    return complex(_A1 @ np.asarray(phasors_abc))


def zero_sequence(phasors_abc: np.ndarray) -> complex:
    """Zero-sequence component of a 3-phase phasor triple."""
    return complex(_A0 @ np.asarray(phasors_abc))


def double_ended_location(
    v_s: complex,
    i_s: complex,
    v_r: complex,
    i_r: complex,
    z1_line: complex,
    *,
    min_current_ratio: float = 1e-2,
) -> float:
    """Synchronized two-ended positive-sequence fault distance (fraction from S).

    Solves ``V_S - m Z1 I_S = V_R - (1-m) Z1 I_R`` (both currents directed INTO
    the line), giving ``m = (V_S - V_R + Z1 I_R) / (Z1 (I_S + I_R))``. Inputs are
    positive-sequence phasors and ``z1_line`` is the *total* positive-sequence
    series impedance (Ohm) of the line. Returns ``nan`` if the denominator is
    degenerate (``|I_S + I_R|`` negligible vs the terminal currents), which would
    otherwise blow the estimate up.
    """
    denom_current = i_s + i_r
    scale = max(abs(i_s), abs(i_r), 1e-30)
    if abs(denom_current) < min_current_ratio * scale:
        return float("nan")
    m = (v_s - v_r + z1_line * i_r) / (z1_line * denom_current)
    return float(m.real)


def single_ended_reactance(
    v_abc: np.ndarray,
    i_abc: np.ndarray,
    z1_line: complex,
    z0_line: complex,
    faulted_phases: Sequence[int],
    grounded: bool,
) -> float:
    """One-terminal reactance-method fault distance (fraction from this terminal).

    Fault-loop selection:
      * single grounded phase  -> phase-ground loop ``V_p / (I_p + k0 I_res)``
        with ``k0 = (Z0 - Z1) / (3 Z1)`` and ``I_res = I_a + I_b + I_c``;
      * otherwise (LL / LLG / 3ph) -> phase-phase loop ``(V_p - V_q)/(I_p - I_q)``.
    The reactance method takes ``m = Im(Z_apparent) / Im(Z1_line)``, which cancels
    a purely resistive fault term but (unlike the two-ended method) remains
    sensitive to remote infeed and load.
    """
    v = np.asarray(v_abc)
    i = np.asarray(i_abc)
    fp = [int(p) for p in faulted_phases]
    if len(fp) == 1 and grounded:
        p = fp[0]
        i_res = complex(i.sum())
        k0 = (z0_line - z1_line) / (3.0 * z1_line)
        denom = i[p] + k0 * i_res
        # Same guard as the phase-phase branch below: a vanishing fault-loop
        # current gives no usable estimate. Returning nan (rather than letting
        # the division produce inf) matters downstream -- the aggregation in
        # run_impedance_baseline drops nan but not inf, so a single degenerate
        # window would otherwise push mae/rmse/p90/p99 to inf for its whole
        # protocol x method group.
        if abs(denom) < 1e-30:
            return float("nan")
        z_app = v[p] / denom
    else:
        p, q = (fp[0], fp[1]) if len(fp) >= 2 else (0, 1)
        denom = i[p] - i[q]
        if abs(denom) < 1e-30:
            return float("nan")
        z_app = (v[p] - v[q]) / denom
    if abs(z1_line.imag) < 1e-30:
        return float("nan")
    return float(z_app.imag / z1_line.imag)
