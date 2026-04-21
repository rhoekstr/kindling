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
