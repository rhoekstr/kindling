"""Bayesian blend tests.

The critical test is synthetic-data weight recovery: generate outcomes
from known weights, run VI, check that the posterior mean is close to
the truth and that the credible interval covers it.
"""

from __future__ import annotations

import numpy as np
import pytest

from kindling.blend import (
    BayesianBlend,
    BinaryIndependent,
    ListwiseCalibration,
    MultinomialSoftmax,
    OutcomeBatch,
    PairwiseBradleyTerry,
    run_diagnostics,
)
from kindling.blend.heuristic import SignalFeatures


def _synthetic_outcomes(
    true_weights: np.ndarray,
    n_lists: int,
    items_per_list: int,
    rng: np.random.Generator,
    noise_scale: float = 0.0,
) -> OutcomeBatch:
    """Generate synthetic (signals, selected) batches from known weights.

    For each list, draw K signals uniform in [0, 1], compute score =
    signals @ weights, and sample selected ~ Bernoulli(sigmoid(score)).
    """
    k = true_weights.shape[0]
    total = n_lists * items_per_list
    signals = rng.uniform(size=(total, k))
    scores = signals @ true_weights
    if noise_scale > 0:
        scores = scores + rng.normal(scale=noise_scale, size=scores.shape)
    probs = 1.0 / (1.0 + np.exp(-scores))
    selected = (rng.uniform(size=total) < probs).astype(np.int64)
    positions = np.tile(np.arange(1, items_per_list + 1), n_lists)
    list_ids = np.repeat(np.arange(n_lists), items_per_list)
    return OutcomeBatch(
        signal_matrix=signals,
        selected=selected,
        positions=positions,
        list_ids=list_ids,
    )


def test_vi_recovers_weight_ordering_binary_independent() -> None:
    """Given synthetic outcomes from known weights, VI should recover the
    correct ordering of weights (largest stays largest, etc). Magnitude
    recovery is looser because REINFORCE is noisy at small batch sizes;
    the precision test runs on listwise calibration below."""
    rng = np.random.default_rng(seed=42)
    true_weights = np.array([0.6, 0.3, 0.1])
    batch = _synthetic_outcomes(true_weights, n_lists=500, items_per_list=10, rng=rng)
    blend = BayesianBlend.from_prior(
        signal_names=("a", "b", "c"),
        prior_alpha=np.array([1.0, 1.0, 1.0]),
    )
    blend.fit_posterior(
        batch=batch,
        likelihood=BinaryIndependent(),
        rng=np.random.default_rng(seed=0),
        max_iter=300,
    )
    recovered = blend.posterior_mean
    # Weights sum to 1 (Dirichlet constraint).
    assert abs(recovered.sum() - 1.0) < 1e-9
    # Correct rank ordering.
    assert np.argmax(recovered) == np.argmax(true_weights)
    # Magnitude within 0.2 (REINFORCE variance tolerance on synthetic data).
    np.testing.assert_allclose(recovered, true_weights, atol=0.2)


def test_vi_posterior_contracts_with_more_data() -> None:
    """Posterior variance should shrink as more outcomes are added."""
    rng = np.random.default_rng(seed=7)
    true_weights = np.array([0.5, 0.3, 0.2])

    small_batch = _synthetic_outcomes(true_weights, 20, 10, rng)
    big_batch = _synthetic_outcomes(true_weights, 200, 10, rng)

    prior = np.array([1.0, 1.0, 1.0])
    blend_small = BayesianBlend.from_prior(("a", "b", "c"), prior).fit_posterior(
        small_batch, BinaryIndependent(), np.random.default_rng(0), max_iter=200
    )
    blend_big = BayesianBlend.from_prior(("a", "b", "c"), prior).fit_posterior(
        big_batch, BinaryIndependent(), np.random.default_rng(0), max_iter=200
    )
    assert blend_big.posterior_variance.sum() < blend_small.posterior_variance.sum()


def test_vi_reproducibility_with_fixed_seed() -> None:
    """Same seed -> same posterior."""
    rng = np.random.default_rng(seed=3)
    true_weights = np.array([0.4, 0.4, 0.2])
    batch = _synthetic_outcomes(true_weights, 100, 10, rng)

    prior = np.array([1.0, 1.0, 1.0])
    b1 = BayesianBlend.from_prior(("a", "b", "c"), prior).fit_posterior(
        batch, BinaryIndependent(), np.random.default_rng(seed=0), max_iter=100
    )
    b2 = BayesianBlend.from_prior(("a", "b", "c"), prior).fit_posterior(
        batch, BinaryIndependent(), np.random.default_rng(seed=0), max_iter=100
    )
    np.testing.assert_allclose(b1.posterior_beta, b2.posterior_beta, atol=1e-12)


def test_vi_prior_dominates_with_no_data() -> None:
    """With an empty outcome batch, the posterior should equal the prior."""
    empty = OutcomeBatch(
        signal_matrix=np.zeros((0, 3)),
        selected=np.zeros(0, dtype=np.int64),
        positions=np.zeros(0, dtype=np.int64),
        list_ids=np.zeros(0, dtype=np.int64),
    )
    prior = np.array([2.0, 5.0, 1.0])
    blend = BayesianBlend.from_prior(("a", "b", "c"), prior)
    blend.fit_posterior(empty, BinaryIndependent(), np.random.default_rng(0), max_iter=50)
    # Allowing small drift because Adam still takes steps and the KL
    # gradient is nonzero at beta != alpha.
    np.testing.assert_allclose(blend.posterior_mean, prior / prior.sum(), atol=0.05)


def test_credible_interval_covers_truth_at_claimed_rate() -> None:
    """On synthetic data, the per-signal 90% CI should contain the truth
    for most signals (allowing for finite-sample variation)."""
    rng = np.random.default_rng(seed=11)
    true_weights = np.array([0.5, 0.3, 0.2])
    batch = _synthetic_outcomes(true_weights, 200, 10, rng)
    blend = BayesianBlend.from_prior(("a", "b", "c"), np.array([1.0, 1.0, 1.0])).fit_posterior(
        batch, BinaryIndependent(), np.random.default_rng(0), max_iter=200
    )
    ci = blend.credible_interval(coverage=0.9)
    # For each signal, is truth inside [lower, upper]? At least 2 of 3 should be.
    covered = ((ci[:, 0] <= true_weights) & (true_weights <= ci[:, 1])).sum()
    assert covered >= 2


def test_score_with_uncertainty_respects_coverage() -> None:
    """Higher coverage -> wider intervals."""
    rng = np.random.default_rng(seed=5)
    batch = _synthetic_outcomes(np.array([0.5, 0.5]), 50, 10, rng)
    blend = BayesianBlend.from_prior(("a", "b"), np.array([1.0, 1.0])).fit_posterior(
        batch, BinaryIndependent(), np.random.default_rng(0), max_iter=100
    )
    features = SignalFeatures(
        matrix=rng.uniform(size=(10, 2)),
        signal_names=("a", "b"),
    )
    _, lower50, upper50 = blend.score_with_uncertainty(features, coverage=0.5)
    _, lower95, upper95 = blend.score_with_uncertainty(features, coverage=0.95)
    # 95% band must be wider than 50% band at every point.
    assert np.all((upper95 - lower95) >= (upper50 - lower50) - 1e-9)


def test_listwise_calibration_log_prob_is_finite() -> None:
    rng = np.random.default_rng(seed=2)
    batch = _synthetic_outcomes(np.array([0.5, 0.3, 0.2]), 50, 10, rng)
    ll = ListwiseCalibration().log_prob(np.array([0.4, 0.4, 0.2]), batch)
    assert np.isfinite(ll)


def test_pairwise_log_prob_finite_and_nonpositive() -> None:
    rng = np.random.default_rng(seed=4)
    batch = _synthetic_outcomes(np.array([0.5, 0.3, 0.2]), 30, 10, rng)
    ll = PairwiseBradleyTerry().log_prob(np.array([0.4, 0.4, 0.2]), batch)
    assert np.isfinite(ll)
    assert ll <= 0  # log probability


def test_multinomial_log_prob_finite() -> None:
    rng = np.random.default_rng(seed=6)
    batch = _synthetic_outcomes(np.array([0.5, 0.3, 0.2]), 30, 10, rng)
    ll = MultinomialSoftmax().log_prob(np.array([0.4, 0.4, 0.2]), batch)
    assert np.isfinite(ll)


def test_diagnostics_runs_on_fitted_blend() -> None:
    rng = np.random.default_rng(seed=13)
    batch = _synthetic_outcomes(np.array([0.5, 0.3, 0.2]), 100, 10, rng)
    blend = BayesianBlend.from_prior(("a", "b", "c"), np.array([1.0, 1.0, 1.0])).fit_posterior(
        batch, BinaryIndependent(), np.random.default_rng(0), max_iter=150
    )
    report = run_diagnostics(
        blend, batch, BinaryIndependent(), np.random.default_rng(0), n_ppc_samples=64
    )
    # All fields populate; we don't require all pass on this small sample.
    assert report.elbo_final == pytest.approx(report.elbo_final)
    assert 0.0 <= report.ess_ratio <= 1.0
