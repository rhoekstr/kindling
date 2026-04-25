"""Tests for the fit-time auto-calibrator for layered scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling import Engine
from kindling.blend.layered_calibrator import (
    DEFAULT_BOOST_GRID,
    DEFAULT_Z_GRID,
    CalibrationResult,
    calibrate,
)
from kindling.loaders import synthetic


@pytest.fixture(scope="module")
def fitted_grocery_engine() -> Engine:
    split = synthetic.make_grocery(
        n_entities=200, n_items_per_category=10, n_categories=4,
        n_sessions_per_entity=8, items_per_session=4, seed=0,
    )
    return Engine().fit(split.train)


def test_calibrate_returns_a_layered_config(fitted_grocery_engine) -> None:
    result = calibrate(fitted_grocery_engine, n_users=20, retrieval_budget=50)
    assert isinstance(result, CalibrationResult)
    assert result.best_config.z_threshold in set(DEFAULT_Z_GRID) | {2.5}  # default fallback covers 2.5
    assert result.best_config.boost_multiplier in set(DEFAULT_BOOST_GRID) | {3.0}
    assert result.n_users_evaluated > 0
    assert result.elapsed_seconds >= 0
    # Grid should have full Z×B cells.
    assert len(result.grid_results) == len(DEFAULT_Z_GRID) * len(DEFAULT_BOOST_GRID)


def test_calibrate_unfitted_engine_returns_default() -> None:
    engine = Engine()
    result = calibrate(engine, n_users=10)
    assert result.fallback_to_default
    assert result.n_users_evaluated == 0


def test_calibrate_grid_results_include_metric_fields(fitted_grocery_engine) -> None:
    result = calibrate(fitted_grocery_engine, n_users=20, retrieval_budget=50)
    for cell in result.grid_results:
        assert "z" in cell
        assert "boost" in cell
        assert "ndcg" in cell
        assert "mrr" in cell


def test_calibrate_with_custom_grid(fitted_grocery_engine) -> None:
    z_grid = (1.5, 2.0)
    boost_grid = (1.0, 5.0)
    result = calibrate(
        fitted_grocery_engine, n_users=20, retrieval_budget=50,
        z_grid=z_grid, boost_grid=boost_grid,
    )
    assert len(result.grid_results) == 4
    for cell in result.grid_results:
        assert cell["z"] in z_grid
        assert cell["boost"] in boost_grid


def test_calibrate_seed_is_deterministic(fitted_grocery_engine) -> None:
    r1 = calibrate(fitted_grocery_engine, n_users=20, retrieval_budget=50, rng_seed=42)
    r2 = calibrate(fitted_grocery_engine, n_users=20, retrieval_budget=50, rng_seed=42)
    assert r1.n_users_evaluated == r2.n_users_evaluated
    # Best config should match exactly with same seed + state.
    assert r1.best_config.z_threshold == r2.best_config.z_threshold
    assert r1.best_config.boost_multiplier == r2.best_config.boost_multiplier
    # Grid NDCG values match.
    g1 = sorted(r1.grid_results, key=lambda r: (r["z"], r["boost"]))
    g2 = sorted(r2.grid_results, key=lambda r: (r["z"], r["boost"]))
    for c1, c2 in zip(g1, g2, strict=True):
        assert c1["ndcg"] == pytest.approx(c2["ndcg"])


def test_calibrate_sparse_data_caps_boost(fitted_grocery_engine) -> None:
    """When avg events/user is below sparse_data_threshold, the
    calibrator caps boost_multiplier to avoid amplifying noise."""
    # Force sparse-data branch by setting a very high threshold so
    # any dataset triggers the cap.
    result = calibrate(
        fitted_grocery_engine,
        n_users=20, retrieval_budget=50,
        sparse_data_threshold=10_000,
        sparse_data_boost_ceiling=3.0,
    )
    # The chosen boost should never exceed the ceiling.
    assert result.best_config.boost_multiplier <= 3.0
    # And no grid cell with boost > 3 should appear.
    for cell in result.grid_results:
        assert cell["boost"] <= 3.0


def test_calibrate_lift_check_returns_disabled_when_no_lift(
    fitted_grocery_engine,
) -> None:
    """When no grid cell beats cooc-only by min_lift, the calibrator
    returns boost_multiplier=0 (effectively disables boosting)."""
    # Set a min_lift that no cell can beat (huge floor).
    result = calibrate(
        fitted_grocery_engine,
        n_users=20, retrieval_budget=50,
        min_lift_over_cooc_only=1.0,  # impossible
    )
    assert result.fallback_to_default is True
    assert result.best_config.boost_multiplier == 0.0


def test_calibrate_min_user_interactions_filter(fitted_grocery_engine) -> None:
    """min_user_interactions filter should be respected; if no users
    have enough history, fallback_to_default fires."""
    result = calibrate(
        fitted_grocery_engine, n_users=20,
        min_user_interactions=10_000,  # impossible threshold
    )
    assert result.fallback_to_default
    assert result.n_users_evaluated == 0
