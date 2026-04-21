"""Decay function tests. Invariants:

- ``decay(0) == 1.0``
- Monotonic non-increasing in age
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kindling.lifecycle.decay import (
    CustomDecay,
    ExponentialDecay,
    LinearDecay,
    NoDecay,
)


def test_exponential_half_life() -> None:
    decay = ExponentialDecay(half_life_days=180.0)
    assert decay(0) == pytest.approx(1.0)
    assert decay(180) == pytest.approx(0.5, rel=1e-6)
    assert decay(360) == pytest.approx(0.25, rel=1e-6)


def test_exponential_rejects_nonpositive_half_life() -> None:
    with pytest.raises(ValueError, match="half_life_days must be positive"):
        ExponentialDecay(half_life_days=0.0)


def test_exponential_monotonic() -> None:
    decay = ExponentialDecay(half_life_days=30.0)
    values = np.array([decay(age) for age in range(0, 200, 5)])
    assert np.all(np.diff(values) <= 0)


def test_linear_decays_to_zero() -> None:
    decay = LinearDecay(zero_at_days=365.0)
    assert decay(0) == 1.0
    assert decay(365) == 0.0
    assert decay(1000) == 0.0  # saturated


def test_linear_midpoint() -> None:
    decay = LinearDecay(zero_at_days=100.0)
    assert decay(50.0) == pytest.approx(0.5, rel=1e-9)


def test_no_decay_is_unit() -> None:
    decay = NoDecay()
    for age in [0, 1, 100, 10_000]:
        assert decay(age) == 1.0


def test_custom_decay_passes_through() -> None:
    decay = CustomDecay(fn=lambda age: math.exp(-0.01 * age))
    assert decay(0) == pytest.approx(1.0)
    assert decay(100) == pytest.approx(math.exp(-1), rel=1e-6)


def test_decay_vectorized() -> None:
    decay = ExponentialDecay(half_life_days=30.0)
    ages = np.array([0, 30, 60, 90], dtype=np.float64)
    values = decay(ages)
    assert isinstance(values, np.ndarray)
    np.testing.assert_allclose(values, [1.0, 0.5, 0.25, 0.125], rtol=1e-6)
