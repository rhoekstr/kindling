"""Per-query signal-column normalization.

Confirms z-score, min-max, softmax, and none behaviors produce the
expected output shapes + preserve the "dead signal = zero column"
invariant (no NaNs, no division-by-zero).
"""

from __future__ import annotations

import numpy as np
import pytest

from kindling.blend.normalize import normalize_columns


def test_none_is_identity() -> None:
    m = np.random.default_rng(0).standard_normal((5, 3))
    out = normalize_columns(m, mode="none")
    assert out is m or np.array_equal(out, m)


def test_zscore_centers_and_scales() -> None:
    m = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]])
    out = normalize_columns(m, mode="zscore")
    for c in range(2):
        assert out[:, c].mean() == pytest.approx(0.0, abs=1e-9)
        assert out[:, c].std() == pytest.approx(1.0, abs=1e-9)


def test_zscore_dead_column_stays_zero() -> None:
    m = np.array([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]])
    out = normalize_columns(m, mode="zscore")
    # All-same column becomes zeros (not NaN).
    assert np.isfinite(out).all()
    assert (out[:, 0] == 0.0).all()
    assert (out[:, 1] == 0.0).all()


def test_minmax_maps_to_unit_interval() -> None:
    m = np.array([[1000.0, 0.1], [2000.0, 0.5], [3000.0, 0.9]])
    out = normalize_columns(m, mode="minmax")
    assert out[:, 0].min() == pytest.approx(0.0)
    assert out[:, 0].max() == pytest.approx(1.0)
    assert out[:, 1].min() == pytest.approx(0.0)
    assert out[:, 1].max() == pytest.approx(1.0)


def test_softmax_sums_to_one_per_column() -> None:
    m = np.array([[2.0, 0.1], [1.0, 0.5], [0.5, 0.2]])
    out = normalize_columns(m, mode="softmax", softmax_temperature=1.0)
    for c in range(2):
        assert out[:, c].sum() == pytest.approx(1.0, abs=1e-9)


def test_scale_mismatch_collapsed_by_zscore() -> None:
    """The core motivation: a large-magnitude signal shouldn't drown
    out a [0, 1] signal after z-scoring. Both columns should contribute
    comparably to a downstream linear blend."""
    # Simulate 5 candidates: cooc column in the thousands, cosine in [0, 1].
    m = np.array([
        [25000.0, 0.91],
        [22000.0, 0.82],
        [19000.0, 0.75],
        [15000.0, 0.60],
        [10000.0, 0.40],
    ])
    out = normalize_columns(m, mode="zscore")
    # After z-scoring, both columns have std 1.0, so weighted sum
    # isn't dominated by magnitude.
    assert out[:, 0].std() == pytest.approx(1.0, abs=1e-6)
    assert out[:, 1].std() == pytest.approx(1.0, abs=1e-6)
    # Ordering preserved within each column (not inverted).
    assert np.argsort(-out[:, 0]).tolist() == [0, 1, 2, 3, 4]
    assert np.argsort(-out[:, 1]).tolist() == [0, 1, 2, 3, 4]


def test_empty_matrix_passes_through() -> None:
    m = np.zeros((0, 3))
    out = normalize_columns(m, mode="zscore")
    assert out.shape == (0, 3)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        normalize_columns(np.ones((2, 2)), mode="invalid")  # type: ignore[arg-type]
