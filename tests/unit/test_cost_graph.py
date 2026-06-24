"""Cost graph tests (PRD §3.6, plan Phase 5).

Invariants:
- positive_only mode + `remove` actions: costs are not populated.
- explicit mode + `remove` action: entity_cost increases for (entity, item).
- adding more `remove` events for the same pair monotonically increases
  effective_cost for that pair.
- population_cost is bounded in [0, 1] (fraction of entities rejecting).
"""

from __future__ import annotations

import pandas as pd
import pytest

from kindling.engine import Engine
from kindling.graph.cost_graph import (
    DEFAULT_ALPHA_POP,
    CostGraph,
    _context_key,
    build_cost_graph,
)


def _explicit_interactions() -> pd.DataFrame:
    """Entities a, b interact positively with items 1, 2; a rejects item 3."""
    return pd.DataFrame(
        {
            "entity_id": ["a", "a", "a", "b", "b"],
            "item_id": [1, 2, 3, 1, 2],
            "action_type": ["add", "add", "remove", "add", "add"],
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
            ),
        }
    )


def test_no_action_type_yields_empty_cost_graph() -> None:
    df = pd.DataFrame({"entity_id": ["a", "b"], "item_id": [1, 2]})
    graph = build_cost_graph(df)
    assert graph.n_rejected_items == 0
    assert graph.n_entity_rejections == 0


def test_remove_action_populates_entity_layer() -> None:
    df = _explicit_interactions()
    graph = build_cost_graph(df)
    assert graph.entity_cost[("a", 3)] == 1.0
    assert ("b", 3) not in graph.entity_cost


def test_population_cost_is_fraction_of_entities() -> None:
    df = _explicit_interactions()
    graph = build_cost_graph(df)
    # Item 3: 1 of 2 entities rejected it -> 0.5.
    assert graph.population_cost[3] == pytest.approx(0.5)


def test_effective_cost_combines_layers() -> None:
    df = _explicit_interactions()
    graph = build_cost_graph(df)
    eff = graph.effective_cost(item=3, entity="a")
    # alpha_pop * pop + entity = 0.3 * 0.5 + 1.0 = 1.15 (plus zero context)
    assert eff == pytest.approx(DEFAULT_ALPHA_POP * 0.5 + 1.0)


def test_monotonic_increase_with_more_removes() -> None:
    """Property: adding another remove event for (entity, item) strictly
    increases effective_cost for that pair."""
    df = _explicit_interactions()
    graph_once = build_cost_graph(df)
    df_twice = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "entity_id": ["a"],
                    "item_id": [3],
                    "action_type": ["remove"],
                    "timestamp": pd.to_datetime(["2026-01-10"]),
                }
            ),
        ],
        ignore_index=True,
    )
    graph_twice = build_cost_graph(df_twice)
    assert graph_twice.effective_cost(3, "a") > graph_once.effective_cost(3, "a")


def test_negative_rating_also_populates_cost() -> None:
    df = pd.DataFrame(
        {
            "entity_id": ["a", "a"],
            "item_id": [1, 2],
            "action_type": ["rate", "rate"],
            "rating": [5.0, 1.0],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        }
    )
    graph = build_cost_graph(df)
    # Low-rated item 2 goes into cost graph; high-rated item 1 does not.
    assert graph.entity_cost.get(("a", 2), 0.0) > 0
    assert graph.entity_cost.get(("a", 1), 0.0) == 0


def test_context_key_is_deterministic() -> None:
    a = _context_key({1, 2, 3})
    b = _context_key({3, 2, 1})
    assert a == b


def test_context_key_bounded_size() -> None:
    key = _context_key(set(range(20)))
    assert len(key) == 8


def test_vectorized_lookups_match_scalar() -> None:
    df = _explicit_interactions()
    graph = build_cost_graph(df)
    items = [1, 2, 3, 4]
    scalar = [graph.effective_cost(i, "a") for i in items]
    vec = graph.effective_cost_many(items, "a")
    for s, v in zip(scalar, vec, strict=True):
        assert v == pytest.approx(s)


def test_positive_only_mode_ignores_remove_actions() -> None:
    """Engine integration: when negative_signal_mode='positive_only', the
    cost graph stays empty even when remove actions are present."""

    df = _explicit_interactions()
    engine = Engine(negative_signal_mode="positive_only", vi_max_iter=30).fit(df)
    # Unreachable: engine's cost graph is empty.
    assert engine._cost_graph is not None
    assert engine._cost_graph.n_rejected_items == 0


def test_explicit_mode_builds_cost_graph() -> None:

    df = _explicit_interactions()
    engine = Engine(negative_signal_mode="explicit", vi_max_iter=30).fit(df)
    assert engine._cost_graph is not None
    assert engine._cost_graph.n_rejected_items > 0


def test_explicit_mode_is_default_when_action_type_present() -> None:
    """With action_type column and no explicit mode, engine picks 'explicit'."""

    df = _explicit_interactions()
    engine = Engine(vi_max_iter=30).fit(df)
    assert engine.negative_signal_mode == "explicit"


def test_positive_only_is_default_without_action_type() -> None:

    df = pd.DataFrame(
        {
            "entity_id": ["a", "a", "b", "b"],
            "item_id": [1, 2, 1, 3],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
        }
    )
    engine = Engine(vi_max_iter=30).fit(df)
    assert engine.negative_signal_mode == "positive_only"


def test_invalid_mode_rejected() -> None:

    with pytest.raises(ValueError, match="negative_signal_mode"):
        Engine(negative_signal_mode="not_a_mode")


def test_empty_cost_graph_effective_cost_is_zero() -> None:
    graph = CostGraph()
    assert graph.effective_cost(42, "a") == 0.0
