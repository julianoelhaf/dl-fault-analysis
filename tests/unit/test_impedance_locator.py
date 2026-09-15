"""Unit tests for the classical impedance-based fault locator.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from dl_fault_analysis.models.impedance_locator import (
    double_ended_location,
    fundamental_phasors,
    positive_sequence,
    single_ended_reactance,
)


def test_fundamental_phasor_recovers_amplitude_and_phase():
    n = 128
    t = np.arange(n)
    amp, phase = 100.0, 0.7
    sig = amp * np.cos(2 * np.pi * t / n + phase)
    ph = fundamental_phasors(np.stack([sig, sig, sig], axis=1), n)
    assert np.allclose(np.abs(ph), amp, rtol=1e-6)
    assert np.allclose(np.angle(ph), phase, atol=1e-6)


def test_double_ended_recovers_distance_independent_of_rf_and_infeed():
    # Construct two-ended positive-seq phasors consistent with a fault at m;
    # the algorithm must recover m EXACTLY for any fault-point voltage / currents
    # (i.e. independent of fault resistance and remote infeed).
    z1 = complex(0.1, 0.4) * 30.0  # R'/X' Ohm/km * length km
    for m in (0.05, 0.5, 0.92):
        i_s, i_r = complex(800, -120), complex(500, 60)
        v_f = complex(40000, -3000)  # arbitrary (encodes R_f * I_f)
        v_s = v_f + m * z1 * i_s
        v_r = v_f + (1.0 - m) * z1 * i_r
        est = double_ended_location(v_s, i_s, v_r, i_r, z1)
        assert est == pytest.approx(m, abs=1e-9)


def test_double_ended_guards_degenerate_denominator():
    z1 = complex(0.1, 0.4) * 30.0
    i_s = complex(700, 50)
    est = double_ended_location(complex(1, 1), i_s, complex(1, 1), -i_s, z1)
    assert np.isnan(est)  # I_S + I_R == 0 -> undefined


def test_single_ended_phase_phase_exact_without_fault_resistance():
    # Phase-phase loop with no fault resistance -> reactance method is exact.
    z1 = complex(0.08, 0.42) * 25.0
    z0 = complex(0.13, 0.6) * 25.0
    m = 0.4
    i_p, i_q = complex(1200, -200), complex(-1150, 150)
    # V_p - V_q = m * Z1 * (I_p - I_q) (no R_f term)
    base = m * z1 * (i_p - i_q)
    v = np.array([base, 0.0, 0.0], dtype=complex)
    i = np.array([i_p, i_q, 0.0], dtype=complex)
    est = single_ended_reactance(v, i, z1, z0, faulted_phases=[0, 1], grounded=False)
    assert est == pytest.approx(m, abs=1e-9)


def test_positive_sequence_of_balanced_set():
    a = np.exp(1j * 2 * np.pi / 3)
    balanced = np.array([1.0, a**2, a])  # positive-sequence set
    assert positive_sequence(balanced) == pytest.approx(1.0, abs=1e-9)


def test_single_ended_slg_returns_nan_on_a_vanishing_fault_loop_current():
    """A degenerate phase-ground loop must yield nan, not inf.

    run_impedance_baseline aggregates with `err.dropna()`, which drops nan but
    keeps inf, so an inf here would propagate into mae/rmse/p90/p99 for a whole
    protocol x method group rather than being excluded as one unusable window.
    The phase-phase branch already guarded this; the SLG branch did not.
    """
    z1 = complex(1.0, 10.0)
    z0 = complex(3.0, 30.0)
    k0 = (z0 - z1) / (3.0 * z1)

    # Choose i_a so that the SLG loop current i[p] + k0 * (ia+ib+ic) is exactly 0.
    # With ib = ic = 0 the loop denominator is i_a * (1 + k0).
    i_abc = np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
    v_abc = np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
    assert abs(1.0 + k0) > 1e-12  # sanity: the zero comes from the current, not k0

    out = single_ended_reactance(v_abc, i_abc, z1, z0, faulted_phases=[0], grounded=True)
    assert np.isnan(out), f"expected nan for a vanishing SLG loop current, got {out!r}"
    assert not np.isinf(out)


def test_single_ended_slg_still_returns_a_finite_estimate_when_well_conditioned():
    """The guard must not swallow ordinary, well-conditioned SLG windows."""
    z1 = complex(1.0, 10.0)
    z0 = complex(3.0, 30.0)
    i_abc = np.array([100.0 - 20.0j, 0.0 + 0.0j, 0.0 + 0.0j])
    v_abc = np.array([500.0 + 300.0j, 0.0 + 0.0j, 0.0 + 0.0j])

    out = single_ended_reactance(v_abc, i_abc, z1, z0, faulted_phases=[0], grounded=True)
    assert np.isfinite(out)
