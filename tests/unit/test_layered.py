"""Tests for the layered scoring architecture.

Covers:
- Boost magnitude is a multiple of primary top-K adjacent gap median.
- Z is one-tailed: only positive z above threshold fires.
- Z is computed on the non-zero subset (sparse signals).
- Cumulative stack: multiple layers add independent boosts.
- Empty / degenerate inputs return primary unchanged.
- Diagnostic report exposes fire rates per layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from kindling.blend.layered import (
    LayeredConfig,
    _calibrate_boost,
    diagnostic_report,
    is_layer_meaningful,
    layered_score,
)


def test_no_refinements_returns_primary_unchanged() -> None:
    primary = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    out = layered_score(primary, refinement_scores=[])
    np.testing.assert_array_equal(out, primary)


def test_single_layer_boosts_high_z_candidate() -> None:
    """Candidate at z>2 fires; candidates at z<2 don't.

    Use a realistic-shaped non-zero distribution: 50 candidates around
    a baseline, plus one strong outlier at index 5. With this many
    samples, the outlier's z is well above 2 sigma.
    """
    primary = np.linspace(100, 1, 100)
    rng = np.random.default_rng(0)
    refine = np.zeros(100)
    # 50 baseline non-zero candidates around mean=2 with sigma=1.
    baseline_idx = rng.choice(np.arange(10, 100), size=50, replace=False)
    refine[baseline_idx] = rng.normal(loc=2.0, scale=1.0, size=50).clip(min=0.1)
    # One strong outlier at index 5.
    refine[5] = 20.0
    cfg = LayeredConfig(z_threshold=2.0, boost_multiplier=3.0)
    out = layered_score(primary, refinement_scores=[refine], config=cfg)
    # The standout should be boosted.
    assert out[5] > primary[5]
    # Most baseline-near-mean items should NOT be boosted.
    n_boosted = int((out > primary).sum())
    assert 1 <= n_boosted <= 5  # only the outliers fire


def test_negative_z_does_not_demote() -> None:
    """One-tailed: a candidate with low score gets no penalty."""
    primary = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
    # Strong outlier on idx 0; low score on idx 4.
    refine = np.array([10.0, 1.0, 1.0, 1.0, 0.5])
    out = layered_score(primary, refinement_scores=[refine])
    # idx 4 has lowest non-zero refinement score (negative z) - never demoted.
    assert out[4] >= primary[4]


def test_zero_signal_never_fires() -> None:
    """Items with score=0 should not fire even if 'far below' the
    non-zero population mean."""
    primary = np.array([5.0, 4.0, 3.0])
    refine = np.array([10.0, 0.0, 1.0])  # zero on idx 1
    out = layered_score(primary, refinement_scores=[refine])
    # idx 1 is zero in refinement → its z is -inf → no boost (matches primary).
    assert out[1] == primary[1]


def test_cumulative_stack_adds_independent_boosts() -> None:
    """A candidate that fires on two refinement layers gets ~2× boost."""
    # Primary with clear top-K spacing for boost calibration.
    primary = np.linspace(20, 1, 20)

    # Layer A: idx 0 stands out among nonzero {0, 5, 10, 15}.
    layer_a = np.zeros(20)
    layer_a[0] = 100.0
    layer_a[5] = 1.0
    layer_a[10] = 1.0
    layer_a[15] = 1.0

    # Layer B: idx 0 stands out again (independently).
    layer_b = np.zeros(20)
    layer_b[0] = 200.0
    layer_b[3] = 2.0
    layer_b[7] = 2.0
    layer_b[12] = 2.0

    out_one = layered_score(primary, [layer_a])
    out_two = layered_score(primary, [layer_a, layer_b])
    boost_one = out_one[0] - primary[0]
    boost_two = out_two[0] - primary[0]
    # Two layers fire, so the cumulative boost is roughly 2x the single-layer boost.
    assert boost_two == pytest.approx(2.0 * boost_one, rel=1e-9)


def test_boost_magnitude_calibrated_to_top_k_gap() -> None:
    """boost = multiplier × median(adjacent gaps in top-K of primary)."""
    primary = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.1])
    cfg = LayeredConfig(boost_multiplier=3.0, top_k_for_boost_calibration=5)
    # Top-5 sorted: 10, 8, 6, 4, 2. Adjacent gaps: 2, 2, 2, 2. Median=2. Boost=6.
    boost = _calibrate_boost(primary, cfg)
    assert boost == pytest.approx(6.0)


def test_zero_boost_when_no_variance_in_primary() -> None:
    """If primary has all-equal scores, boost is zero (no rank-spacing
    to calibrate against)."""
    primary = np.full(10, 1.0)
    refine = np.array([0.0] * 9 + [100.0])
    out = layered_score(primary, refinement_scores=[refine])
    np.testing.assert_array_equal(out, primary)


def test_min_nonzero_skip() -> None:
    """When fewer than min_nonzero candidates have signal, the layer
    is skipped (z is unstable)."""
    primary = np.linspace(10, 1, 10)
    refine = np.zeros(10)
    refine[3] = 100.0  # only 1 non-zero
    cfg = LayeredConfig(min_nonzero_for_zscore=3)
    out = layered_score(primary, refinement_scores=[refine], config=cfg)
    # Layer is skipped; primary unchanged.
    np.testing.assert_array_equal(out, primary)


def test_is_layer_meaningful_passes_balanced_layer() -> None:
    """A layer with ~5% fire rate (typical of well-calibrated sparse
    signals) is in the sensible band."""
    primary = np.linspace(100, 1, 100)
    rng = np.random.default_rng(0)
    refine = np.zeros(100)
    baseline = rng.choice(np.arange(10, 100), size=50, replace=False)
    refine[baseline] = rng.normal(loc=2.0, scale=1.0, size=50).clip(min=0.1)
    refine[0] = 30.0
    refine[5] = 25.0
    ok, reason = is_layer_meaningful(refine, primary)
    assert ok
    assert reason == "ok"


def test_is_layer_meaningful_rejects_all_zeros() -> None:
    primary = np.linspace(100, 1, 100)
    refine = np.zeros(100)
    ok, reason = is_layer_meaningful(refine, primary)
    assert not ok
    assert reason == "too_few_nonzero"


def test_is_layer_meaningful_rejects_low_fire_rate() -> None:
    """Layer with non-zero population but no candidate fires above
    z>2 should be rejected as 'low_fire_rate'."""
    primary = np.linspace(100, 1, 100)
    # 50 non-zero candidates all at the same value → sigma=0 → no fires.
    refine = np.zeros(100); refine[:50] = 1.0
    ok, reason = is_layer_meaningful(refine, primary)
    assert not ok
    assert reason in ("low_fire_rate", "degenerate_layer")


def test_is_layer_meaningful_rejects_high_fire_rate() -> None:
    """Layer where almost everyone fires isn't being selective."""
    primary = np.linspace(100, 1, 100)
    rng = np.random.default_rng(0)
    # Make most non-zero candidates outliers (large positive z).
    refine = np.zeros(100)
    refine[:80] = rng.normal(loc=10.0, scale=0.1, size=80)  # tight cluster
    refine[0:10] = 100.0  # 10 strong outliers - should fire on >30% if narrow distribution
    ok, reason = is_layer_meaningful(refine, primary, max_fire_rate=0.05)
    # With a strict max_fire_rate of 5%, this layer (which fires more) is rejected.
    assert not ok or reason == "high_fire_rate"


def test_is_layer_meaningful_rejects_degenerate_primary() -> None:
    """All-equal primary scores have zero rank spacing → no boost
    magnitude → no point in adding a layer."""
    primary = np.full(100, 1.0)
    refine = np.array([0.0] * 90 + [1.0] * 10)
    ok, reason = is_layer_meaningful(refine, primary)
    assert not ok
    assert reason == "degenerate_primary"


def test_diagnostic_report_fire_rates() -> None:
    """Realistic shape: 100-cand pool with 50-cand non-zero baseline."""
    primary = np.linspace(100, 1, 100)
    rng = np.random.default_rng(1)
    layer_a = np.zeros(100)
    baseline = rng.choice(np.arange(10, 100), size=50, replace=False)
    layer_a[baseline] = rng.normal(loc=2.0, scale=1.0, size=50).clip(min=0.1)
    layer_a[0] = 30.0  # strong outlier
    layer_b = np.zeros(100)  # all zeros
    layer_c = np.full(100, 1.0); layer_c[50:] = 0.0  # half non-zero, no outliers

    cfg = LayeredConfig(z_threshold=2.0)
    rep = diagnostic_report(
        primary,
        refinement_scores_by_name={"a": layer_a, "b": layer_b, "c": layer_c},
        config=cfg,
    )
    assert rep["boost_magnitude"] > 0
    assert rep["layers"]["a"]["n_fired"] >= 1
    assert rep["layers"]["b"]["would_skip"] is True  # all zeros
    # Layer c: 50 non-zero values all equal 1.0 → std=0, z=0 → no fires.
    assert rep["layers"]["c"]["n_fired"] == 0
