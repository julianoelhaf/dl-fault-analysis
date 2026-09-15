"""Unit tests for dl_fault_analysis.data.perturbations.

Ported from the private development repository's EPSR reviewer-response
branch -- see docs/PROVENANCE.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from dl_fault_analysis.data import perturbations as pert


def _window(seed: int = 0, t: int = 50, f: int = 6) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((t, f)).astype(np.float32)


def test_none_is_identity_but_copy():
    x = _window()
    y = pert.apply_perturbation(x, "none")
    assert np.array_equal(x, y)
    assert y is not x  # must not alias the input


def test_time_shift_shifts_by_two_and_preserves_shape():
    x = _window()
    y = pert.apply_perturbation(x, "time_shift_2_samples")
    assert y.shape == x.shape
    assert np.array_equal(y[2:], x[:-2])  # content shifted forward by 2
    assert np.array_equal(y[0], x[0]) and np.array_equal(y[1], x[0])  # edge-pad


def test_block_loss_zeros_expected_block():
    fs, t = 6400.0, 320
    x = _window(t=t, f=4) + 1.0  # nonzero everywhere
    y = pert.apply_perturbation(x, "block_loss_10ms", sampling_frequency=fs)
    n = pert.block_loss_n_samples(fs)  # round(0.010*6400) = 64
    start = int(round(0.1 * t))  # 32
    assert n == 64 and y.shape == x.shape
    assert np.all(y[start : start + n] == 0.0)  # the block is zeroed
    assert np.all(y[:start] == x[:start])  # nothing else touched
    assert np.all(y[start + n :] == x[start + n :])


def test_block_loss_requires_sampling_frequency():
    with pytest.raises(ValueError):
        pert.apply_perturbation(_window(), "block_loss_10ms")


def test_unknown_perturbation_raises():
    with pytest.raises(ValueError):
        pert.apply_perturbation(_window(), "gaussian_noise")


def test_non_2d_raises():
    with pytest.raises(ValueError):
        pert.apply_perturbation(np.zeros((3, 4, 5), dtype=np.float32), "none")


def test_registry_is_exactly_three():
    assert pert.PERTURBATIONS == ("none", "time_shift_2_samples", "block_loss_10ms")
