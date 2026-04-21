"""Tests for the Gram-Schmidt signal decorrelation basis.

The basis standardizes signals first, then applies classical Gram-Schmidt.
The first column is preserved (modulo standardization). Subsequent columns
have their projections onto previous columns removed.
"""

from __future__ import annotations

import numpy as np
import pytest

from kindling.blend.decorrelate import DecorrelationBasis, fit_decorrelation


def test_correlated_signals_produce_orthogonal_transform() -> None:
    """Post-transform columns should have near-zero pairwise dot."""
    rng = np.random.default_rng(42)
    n = 500
    base = rng.normal(size=n)
    s = np.column_stack(
        [
            base,
            0.7 * base + 0.3 * rng.normal(size=n),
            0.4 * base + 0.6 * rng.normal(size=n),
        ]
    )
    basis = fit_decorrelation(s, signal_names=("s1", "s2", "s3"))
    transformed = basis.apply(s)
    for i in range(3):
        for j in range(i + 1, 3):
            dot = float((transformed[:, i] * transformed[:, j]).sum())
            assert abs(dot) < 1e-6, f"Columns {i}, {j} inner product {dot}"


def test_basis_preserves_first_column_shape() -> None:
    """After standardization + GS, the first column equals the z-scored input
    (no projections to remove)."""
    rng = np.random.default_rng(7)
    s = rng.normal(size=(100, 4))
    basis = fit_decorrelation(s, signal_names=("a", "b", "c", "d"))
    transformed = basis.apply(s)
    expected_col0 = (s[:, 0] - basis.means[0]) / basis.stds[0]
    np.testing.assert_allclose(transformed[:, 0], expected_col0, atol=1e-9)


def test_basis_rejects_wrong_shape() -> None:
    basis = DecorrelationBasis(
        signal_names=("a", "b"),
        means=(0.0, 0.0),
        stds=(1.0, 1.0),
        coefficients=((), (0.5,)),
    )
    with pytest.raises(ValueError, match="shape"):
        basis.apply(np.zeros((5, 3)))


def test_basis_handles_constant_column() -> None:
    """A constant column has std=0 - applying the basis shouldn't blow up."""
    s = np.column_stack(
        [
            np.ones(50),  # constant
            np.random.default_rng(0).normal(size=50),
        ]
    )
    basis = fit_decorrelation(s, signal_names=("const", "noise"))
    out = basis.apply(s)
    assert np.all(np.isfinite(out))


def test_basis_stores_fit_statistics() -> None:
    """Means and stds should match the fit data (up to numerical precision)."""
    rng = np.random.default_rng(3)
    s = rng.normal(loc=5.0, scale=2.0, size=(500, 2))
    basis = fit_decorrelation(s, signal_names=("a", "b"))
    assert abs(basis.means[0] - 5.0) < 0.3
    assert abs(basis.stds[0] - 2.0) < 0.3
