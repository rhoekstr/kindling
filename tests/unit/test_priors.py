"""Data-adaptive prior construction tests."""

from __future__ import annotations

import numpy as np
import pytest

from kindling.blend.priors import (
    MAX_ALPHA,
    MIN_ALPHA,
    DataFeatures,
    construct_prior,
    load_prior_coefficients,
)


def _default_features(**overrides: float) -> DataFeatures:
    defaults = {
        "graph_density": 0.0,
        "clustering_coefficient": 0.0,
        "session_density": 0.0,
        "catalog_to_entity_ratio": 1.0,
        "n_interactions": 10_000,
        # Default-on so the session_stiffness rule doesn't fire and the
        # "all-zero features == baseline prior" test stays meaningful.
        # Tests exercising stiffness set this to False explicitly.
        "has_explicit_sessions": True,
    }
    defaults.update(overrides)
    return DataFeatures(**defaults)  # type: ignore[arg-type]


def test_all_zero_features_give_baseline_prior() -> None:
    features = _default_features()
    signals = ("cooccurrence", "community", "path_full", "path_tail", "path_basket")
    alpha = construct_prior(signal_names=signals, features=features)
    np.testing.assert_allclose(alpha, 1.0)


def test_graph_density_boosts_cooccurrence() -> None:
    features = _default_features(graph_density=0.5)
    alpha = construct_prior(signal_names=("cooccurrence", "path_full"), features=features)
    assert alpha[0] > alpha[1]
    assert alpha[0] == pytest.approx(1.0 * (1 + 5.0 * 0.5), rel=1e-6)


def test_clustering_coefficient_boosts_community_and_topology() -> None:
    features = _default_features(clustering_coefficient=0.4)
    alpha = construct_prior(
        signal_names=("community", "graph_topology", "other"), features=features
    )
    assert alpha[0] > alpha[2]
    assert alpha[1] > alpha[2]


def test_high_catalog_ratio_broadens_prior() -> None:
    features_normal = _default_features(graph_density=0.2, catalog_to_entity_ratio=1.0)
    features_coldstart = _default_features(graph_density=0.2, catalog_to_entity_ratio=50.0)
    alpha_normal = construct_prior(("cooccurrence",), features=features_normal)
    alpha_cold = construct_prior(("cooccurrence",), features=features_coldstart)
    # Cold start divides alpha by 2 (shrink_factor) when ratio > 10.
    assert alpha_cold[0] == pytest.approx(alpha_normal[0] / 2.0, rel=1e-6)


def test_alpha_clipped_to_valid_range() -> None:
    features = _default_features(
        graph_density=1.0, clustering_coefficient=1.0, session_density=10.0
    )
    alpha = construct_prior(
        signal_names=("cooccurrence", "community", "path_full"), features=features
    )
    assert np.all(alpha >= MIN_ALPHA)
    assert np.all(alpha <= MAX_ALPHA)


def test_unknown_signal_gets_baseline() -> None:
    features = _default_features(graph_density=0.5)
    alpha = construct_prior(signal_names=("unknown_signal",), features=features)
    assert alpha[0] == pytest.approx(1.0)


def test_priors_toml_loads() -> None:
    coefs = load_prior_coefficients()
    assert "baseline" in coefs
    assert "graph_density" in coefs


def test_session_stiffness_shrinks_path_priors_when_no_explicit_sessions() -> None:
    """Ratings-like input (no session_id column) should shrink the path
    priors so the blend leans on cooccurrence."""
    features = _default_features(
        graph_density=0.01,
        session_density=20.0,  # would normally boost path priors
        has_explicit_sessions=False,
    )
    signals = ("path_full", "path_tail", "path_basket", "cooccurrence")
    alpha = construct_prior(signal_names=signals, features=features)
    # All three path priors clipped to MIN_ALPHA; cooc above baseline
    # because graph_density still applies.
    assert alpha[0] == alpha[1] == alpha[2] == 0.5  # MIN_ALPHA
    assert alpha[3] > 1.0


def test_session_stiffness_skipped_when_explicit_sessions() -> None:
    """Explicit sessions means path priors keep their session_density boost."""
    features = _default_features(
        graph_density=0.01,
        session_density=20.0,
        has_explicit_sessions=True,
    )
    signals = ("path_full", "path_tail", "path_basket", "cooccurrence")
    alpha = construct_prior(signal_names=signals, features=features)
    # Path priors boosted far above baseline.
    assert alpha[0] > 10.0
    assert alpha[1] > 5.0
    assert alpha[2] > 3.0
