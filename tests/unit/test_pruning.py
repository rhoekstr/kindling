"""Pruning tests (PRD §3.5, plan Phase 6).

Invariants:
- Pruning is idempotent: running twice is a no-op on the second pass.
- Pruning below threshold drops only entries with weight < threshold.
- Engine.prune collects preserved aggregates per structure.
- Engine.prune at fit time keeps posterior structure shape sane.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kindling.engine import Engine
from kindling.graph.cost_graph import build_cost_graph
from kindling.graph.item_graph import build_item_graph
from kindling.lifecycle.pruning import PruningConfig
from kindling.path._sessions import SessionSequence
from kindling.path.basket_index import build_basket_index
from kindling.path.path_tree import build_path_tree
from kindling.path.tail_index import build_tail_index


def _sessions() -> list[SessionSequence]:
    return [
        SessionSequence(session_id=0, entity_id="a", items=(1, 2, 3, 4), end_timestamp=None),
        SessionSequence(session_id=1, entity_id="b", items=(1, 2, 5, 6), end_timestamp=None),
        SessionSequence(session_id=2, entity_id="c", items=(7, 8, 9), end_timestamp=None),
    ]


def test_tail_prune_drops_low_weight() -> None:
    idx = build_tail_index(_sessions())
    before = idx.n_pairs
    n, weight = idx.prune_below(support_threshold=1.5)
    assert n > 0
    assert weight > 0
    assert idx.n_pairs < before


def test_tail_prune_idempotent() -> None:
    idx = build_tail_index(_sessions())
    idx.prune_below(0.5)
    second = idx.prune_below(0.5)
    assert second == (0, 0.0)


def test_path_tree_prune_drops_empty_rows() -> None:
    tree = build_path_tree(_sessions(), max_prefix=2)
    max_weight = max(max(row.values()) for row in tree.counts.values())
    before = tree.n_prefixes
    n, _ = tree.prune_below(max_weight + 1.0)
    assert n > 0
    assert tree.n_prefixes < before


def test_basket_prune_rebuilds_postings() -> None:
    idx = build_basket_index(_sessions())
    before = idx.n_observations
    n, _ = idx.prune_below(2.0)
    assert n == before
    assert idx.n_observations == 0
    assert idx.n_items_indexed == 0


def test_item_graph_prune_drops_low_edges() -> None:
    df = pd.DataFrame(
        {
            "entity_id": ["a", "a", "b", "b", "c"],
            "item_id": [1, 2, 1, 3, 4],
        }
    )
    graph = build_item_graph(df)
    before = graph.n_edges
    n, _ = graph.prune_below(2.0)
    assert n > 0
    assert graph.n_edges < before


def test_cost_graph_prune_across_layers() -> None:
    df = pd.DataFrame(
        {
            "entity_id": ["a", "b", "c"],
            "item_id": [1, 1, 1],
            "action_type": ["remove", "remove", "remove"],
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        }
    )
    cost = build_cost_graph(df)
    before_entity = len(cost.entity_cost)
    n, _ = cost.prune_below(1.5)
    assert n > 0
    assert len(cost.entity_cost) < before_entity


# ---- Engine integration -------------------------------------------------


def _phase6_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
            "item_id": [1, 2, 3, 4, 5, 6, 1, 2, 4, 7, 8, 9, 2, 3, 5, 8, 10, 11],
            "timestamp": pd.to_datetime([f"2026-01-{i:02d}" for i in range(1, 7)] * 3),
        }
    )


def test_engine_prune_returns_aggregates() -> None:
    engine = Engine(
        vi_max_iter=20,
        pruning_config=PruningConfig(enabled=True, support_threshold=1.5),
    ).fit(_phase6_df())
    aggregates = engine.prune()
    assert isinstance(aggregates, list)
    for agg in aggregates:
        assert agg.n_pruned_entries == 0
        assert agg.total_pruned_weight == 0.0


def test_engine_prune_records_preserved_aggregates_at_fit() -> None:
    engine = Engine(
        vi_max_iter=20,
        pruning_config=PruningConfig(enabled=True, support_threshold=1.5),
    ).fit(_phase6_df())
    aggregates = engine.preserved_aggregates
    assert len(aggregates) > 0


def test_engine_prune_disabled_is_noop() -> None:
    engine = Engine(
        vi_max_iter=20,
        pruning_config=PruningConfig(enabled=False),
    ).fit(_phase6_df())
    aggregates = engine.prune()
    assert aggregates == []
    assert engine.preserved_aggregates == []


def test_engine_recommend_still_works_after_prune() -> None:
    engine = Engine(
        vi_max_iter=20,
        pruning_config=PruningConfig(enabled=True, support_threshold=0.5),
    ).fit(_phase6_df())
    recs = engine.recommend(entity_id="a", n=3)
    assert isinstance(recs, list)


def test_empty_graphs_and_indexes_prune_cleanly() -> None:
    idx = build_tail_index([])
    assert idx.prune_below(1.0) == (0, 0.0)
    tree = build_path_tree([], max_prefix=2)
    assert tree.prune_below(1.0) == (0, 0.0)
    basket = build_basket_index([])
    assert basket.prune_below(1.0) == (0, 0.0)


def test_negative_threshold_is_noop() -> None:
    idx = build_tail_index(_sessions())
    before = idx.n_pairs
    assert idx.prune_below(-1.0) == (0, 0.0)
    assert idx.n_pairs == before
    assert idx.prune_below(0.0) == (0, 0.0)


@pytest.mark.parametrize("threshold", [0.1, 0.5, 1.5])
def test_pruning_monotonic_in_threshold(threshold: float) -> None:
    """Higher threshold => more pruning."""
    idx1 = build_tail_index(_sessions())
    n1, _ = idx1.prune_below(threshold)
    idx2 = build_tail_index(_sessions())
    n2, _ = idx2.prune_below(threshold + 1.0)
    assert n2 >= n1
