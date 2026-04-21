"""Phase 4 re-rank tests: DPP, temperature, lift, calibration.

Plan invariants verified here:
- DPP greedy MAP is deterministic given fixed inputs; diversity_weight=0
  returns argsort of quality.
- temperature=0 everywhere = pure argmax list.
- temperature=1 everywhere = maximally novel subject to pool.
- Per-position temperature `[0,0,0.5,1,1]` differs from uniform 0.6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling import Engine
from kindling.graph.item_graph import build_item_graph
from kindling.rerank.calibration import (
    CategoryIndex,
    apply_calibration,
    build_category_index,
)
from kindling.rerank.dpp import CooccurrenceCosineKernel, DPPGreedy
from kindling.rerank.lift import apply_lift, compute_population_baselines
from kindling.rerank.temperature import (
    TemperatureObjective,
    resolve_temperature,
    solve_beam,
    solve_greedy,
)

# ---- DPP -----------------------------------------------------------------


def _demo_graph() -> object:
    df = pd.DataFrame(
        {
            "entity_id": ["a"] * 3 + ["b"] * 3 + ["c"] * 2,
            "item_id": [1, 2, 3, 1, 2, 4, 5, 6],
        }
    )
    return build_item_graph(df)


def test_dpp_diversity_zero_returns_quality_argsort() -> None:
    graph = _demo_graph()
    kernel = CooccurrenceCosineKernel(graph)
    dpp = DPPGreedy(kernel=kernel, diversity_weight=0.0)
    qualities = np.array([0.9, 0.3, 0.7, 0.5], dtype=np.float64)
    order = dpp.rerank(item_ids=[1, 2, 3, 4], qualities=qualities, k=4)
    assert order == [0, 2, 3, 1]


def test_dpp_deterministic_with_same_inputs() -> None:
    graph = _demo_graph()
    kernel = CooccurrenceCosineKernel(graph)
    dpp = DPPGreedy(kernel=kernel, diversity_weight=0.5)
    qualities = np.array([0.9, 0.3, 0.7, 0.5], dtype=np.float64)
    a = dpp.rerank(item_ids=[1, 2, 3, 4], qualities=qualities, k=3)
    b = dpp.rerank(item_ids=[1, 2, 3, 4], qualities=qualities, k=3)
    assert a == b


def test_dpp_respects_k_bound() -> None:
    graph = _demo_graph()
    kernel = CooccurrenceCosineKernel(graph)
    dpp = DPPGreedy(kernel=kernel, diversity_weight=0.5)
    qualities = np.ones(4, dtype=np.float64)
    order = dpp.rerank(item_ids=[1, 2, 3, 4], qualities=qualities, k=2)
    assert len(order) == 2


# ---- Temperature ---------------------------------------------------------


def _build_objective() -> TemperatureObjective:
    # 5 candidates: first 2 have high score + low novelty, last 3 have low
    # score + high novelty. A non-trivial temperature profile should
    # produce different picks than a flat profile.
    scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1], dtype=np.float64)
    novelty = np.array([0.1, 0.2, 0.8, 0.85, 0.95], dtype=np.float64)
    return TemperatureObjective(scores=scores, novelty=novelty)


def test_temperature_zero_returns_argmax() -> None:
    obj = _build_objective()
    temps = np.zeros(3, dtype=np.float64)
    picks = solve_greedy(obj, temps, n_positions=3)
    # Top-3 scores are at indices 0, 1, 2.
    assert picks == [0, 1, 2]


def test_temperature_one_favors_novelty() -> None:
    obj = _build_objective()
    temps = np.ones(3, dtype=np.float64)
    picks = solve_greedy(obj, temps, n_positions=3)
    # Top-3 novelty values are at indices 4, 3, 2.
    assert picks == [4, 3, 2]


def test_per_position_differs_from_uniform() -> None:
    """Plan's critical-path test: temperature=[0,0,0.5,1,1] on a 5-item
    list produces demonstrably different outputs than temperature=0.6 (the
    average)."""
    obj = _build_objective()
    staged = solve_beam(obj, np.array([0.0, 0.0, 0.5, 1.0, 1.0]), n_positions=5)
    uniform = solve_beam(obj, np.full(5, 0.6), n_positions=5)
    assert staged != uniform


def test_resolve_temperature_named_profile() -> None:
    temps = resolve_temperature("balanced", n=5)
    np.testing.assert_allclose(temps, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_resolve_temperature_sparse_dict() -> None:
    temps = resolve_temperature({0: 0.0, 4: 1.0}, n=5)
    np.testing.assert_allclose(temps, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_resolve_temperature_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown temperature profile"):
        resolve_temperature("nonsense", n=5)


def test_temperature_solver_reproducibility() -> None:
    obj = _build_objective()
    temps = np.array([0.0, 0.3, 0.6, 0.9, 1.0])
    a = solve_beam(obj, temps, n_positions=5)
    b = solve_beam(obj, temps, n_positions=5)
    assert a == b


# ---- Lift ---------------------------------------------------------------


def test_population_baselines_computed_correctly() -> None:
    df = pd.DataFrame(
        {
            "entity_id": ["a", "a", "b", "b", "c"],
            "item_id": [1, 2, 1, 3, 4],
        }
    )
    baselines = compute_population_baselines(df)
    # item 1 owned by 2 entities of 3 total -> 2/3
    assert baselines.item_to_baseline[1] == pytest.approx(2 / 3)
    assert baselines.item_to_baseline[2] == pytest.approx(1 / 3)
    assert baselines.item_to_baseline[3] == pytest.approx(1 / 3)
    assert baselines.item_to_baseline[4] == pytest.approx(1 / 3)


def test_lift_zero_weight_preserves_scores() -> None:
    df = pd.DataFrame({"entity_id": ["a", "b"], "item_id": [1, 2]})
    baselines = compute_population_baselines(df)
    scores = np.array([1.0, 2.0], dtype=np.float64)
    result = apply_lift(scores, [1, 2], baselines, weight=0.0)
    np.testing.assert_allclose(result, scores)


def test_lift_boosts_rare_items() -> None:
    df = pd.DataFrame(
        {
            "entity_id": ["a", "a", "a", "a", "b", "c"],
            "item_id": [1, 2, 3, 4, 1, 1],  # item 1 is popular, others rare
        }
    )
    baselines = compute_population_baselines(df)
    scores = np.array([1.0, 1.0], dtype=np.float64)  # tied
    result = apply_lift(scores, [1, 2], baselines, weight=1.0)
    # item 2 is rarer so its lifted score should be higher.
    assert result[1] > result[0]


# ---- Calibration --------------------------------------------------------


def _demo_category_index() -> CategoryIndex:
    interactions = pd.DataFrame(
        {
            "entity_id": ["a"] * 4 + ["b"] * 2,
            "item_id": [1, 2, 3, 4, 1, 5],
        }
    )
    metadata = pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4, 5, 6, 7],
            "category": ["x", "x", "x", "y", "y", "z", "z"],
        }
    )
    idx = build_category_index(interactions, metadata, "category")
    assert idx is not None
    return idx


def test_category_index_builds() -> None:
    idx = _demo_category_index()
    assert "x" in idx.item_to_category.values()
    dist_a = idx.entity_distribution("a")
    assert dist_a["x"] == pytest.approx(0.75)
    assert dist_a["y"] == pytest.approx(0.25)


def test_calibration_weight_zero_is_identity() -> None:
    idx = _demo_category_index()
    scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6], dtype=np.float64)
    ordered = [0, 1, 2, 3, 4]
    item_ids = [1, 2, 3, 4, 5]
    result = apply_calibration(
        ordered_indices=ordered,
        item_ids=item_ids,
        scores=scores,
        entity_id="a",
        index=idx,
        weight=0.0,
        k=5,
    )
    assert result == ordered


def test_calibration_biases_toward_entity_distribution() -> None:
    """When the entity strongly prefers x (3/4 of their owned items) the
    calibrated list should include more x-category items than if
    calibration were off."""
    idx = _demo_category_index()
    # Scores rank y-category items highest, but entity 'a' has mostly x.
    # With calibration on, we expect x items to surface despite lower scores.
    scores = np.array([0.5, 0.4, 0.3, 0.9, 0.8], dtype=np.float64)
    ordered = list(np.argsort(-scores))
    item_ids = [1, 2, 3, 4, 5]  # items 1/2/3 are x, 4/5 are y
    plain = ordered[:3]
    calibrated = apply_calibration(
        ordered_indices=ordered,
        item_ids=item_ids,
        scores=scores,
        entity_id="a",
        index=idx,
        weight=0.8,
        k=3,
    )
    x_in_plain = sum(1 for i in plain if idx.item_to_category[item_ids[i]] == "x")
    x_in_calibrated = sum(1 for i in calibrated if idx.item_to_category[item_ids[i]] == "x")
    assert x_in_calibrated >= x_in_plain


# ---- Engine integration -------------------------------------------------


def test_engine_recommend_accepts_rerank_params() -> None:
    df = pd.DataFrame(
        {
            "entity_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
            "item_id": [1, 2, 3, 4, 5, 6, 1, 2, 4, 7, 8, 9, 2, 3, 5, 8, 10, 11],
            "timestamp": pd.to_datetime([f"2026-01-{i:02d}" for i in range(1, 7)] * 3),
        }
    )
    engine = Engine(vi_max_iter=30).fit(df)
    recs = engine.recommend(
        entity_id="a",
        n=3,
        diversity=0.5,
        temperature=[0.0, 0.3, 0.9],
        emphasis="distinctive",
    )
    assert len(recs) <= 3
    assert all(r.credible_interval is not None for r in recs)


def test_engine_temperature_profile_by_name() -> None:
    df = pd.DataFrame(
        {
            "entity_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
            "item_id": [1, 2, 3, 4, 5, 6, 1, 2, 4, 7, 8, 9, 2, 3, 5, 8, 10, 11],
            "timestamp": pd.to_datetime([f"2026-01-{i:02d}" for i in range(1, 7)] * 3),
        }
    )
    engine = Engine(vi_max_iter=30).fit(df)
    recs_uniform = engine.recommend(entity_id="a", n=5, temperature=0.6)
    recs_staged = engine.recommend(entity_id="a", n=5, temperature="explore_tail")
    # Different profile -> different list (or at least different head item).
    uniform_items = [r.item_id for r in recs_uniform]
    staged_items = [r.item_id for r in recs_staged]
    assert uniform_items != staged_items or len(uniform_items) == 0
