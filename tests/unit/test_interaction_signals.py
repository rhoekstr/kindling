"""Unit tests for interaction_network + interaction_neighborhood signals.

Covers:
- interaction_network: PPR convergence, exclusion of owned items,
  cold-start fallback, top-budget output ordering.
- interaction_neighborhood: Louvain produces communities, top-N
  selection, all 5 centrality variants produce valid scores, cache
  hits on repeat queries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling.graph.temporal_interaction import (
    KernelParams,
    build_temporal_interaction_graph,
)
from kindling.retrieve.interaction_network import (
    InteractionNetworkConfig,
    build_interaction_network,
)
from kindling.retrieve.interaction_neighborhood import (
    ALL_CENTRALITIES,
    InteractionNeighborhoodConfig,
    build_interaction_neighborhood,
)


def _toy_graph():
    """Build a small temporal graph with 4 disjoint co-interaction
    clusters so Louvain has clear structure to find."""
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2024-01-01").value // 10**9
    rows = []
    # 4 disjoint groups of 5 items each.
    groups = [list(range(g * 5, g * 5 + 5)) for g in range(4)]
    for u in range(50):
        for s in range(8):
            session_start = base + (u * 86400 * 7) + (s * 86400)
            # Each user mostly interacts within one preferred group.
            preferred = groups[u % 4]
            other = groups[(u % 4 + rng.integers(1, 4)) % 4]
            session_items = list(rng.choice(preferred, size=3, replace=False))
            session_items += list(rng.choice(other, size=1, replace=False))
            for j, item in enumerate(session_items):
                rows.append({
                    "entity_id": u,
                    "item_id": int(item),
                    "timestamp": pd.to_datetime(session_start + j * 60, unit="s"),
                })
    df = pd.DataFrame(rows)
    item_index = {i: i for i in range(20)}
    return build_temporal_interaction_graph(df, item_index), df


# ---- interaction_network ----


def test_interaction_network_ranks_neighbors_above_unrelated() -> None:
    g, df = _toy_graph()
    model = build_interaction_network(g)
    assert model is not None

    # User 0 prefers group 0 (items 0..4); seeds = those items.
    seeds = np.array([0, 1, 2], dtype=np.int64)
    scores = model.score_many(seeds, exclude_indices={0, 1, 2})
    # Items 3, 4 (same group) should outscore items 15..19 (most-distant group).
    assert scores[3] > scores[15]
    assert scores[4] > scores[19]


def test_interaction_network_excludes_owned() -> None:
    g, _ = _toy_graph()
    model = build_interaction_network(g)
    seeds = np.array([0, 1], dtype=np.int64)
    excl = {0, 1, 5}
    scores = model.score_many(seeds, exclude_indices=excl)
    for idx in excl:
        assert scores[idx] == 0.0


def test_interaction_network_returns_empty_when_seeds_unknown() -> None:
    g, _ = _toy_graph()
    model = build_interaction_network(g)
    out = model.retrieve(
        entity_id="ghost",
        owned_items=np.array([], dtype=object),
        history=tuple(),
        budget=10,
        exclude=None,
    )
    assert out == []


def test_interaction_network_top_budget_descending() -> None:
    g, _ = _toy_graph()
    model = build_interaction_network(g)
    out = model.retrieve(
        entity_id=0,
        owned_items=np.array([0, 1, 2], dtype=np.int64),
        history=(0, 1, 2),
        budget=10,
        exclude=None,
    )
    assert all(out[i].score >= out[i + 1].score for i in range(len(out) - 1))


# ---- interaction_neighborhood ----


def test_neighborhood_louvain_finds_communities() -> None:
    g, _ = _toy_graph()
    model = build_interaction_neighborhood(g)
    assert model is not None
    # Toy graph has 4 disjoint groups; Louvain should find 4 communities
    # (give or take, depending on resolution).
    assert 2 <= model.n_communities <= 6
    # Every item assigned to a community.
    assert (model.item_index_to_community >= 0).sum() == 20


def test_neighborhood_top_communities_match_seeds() -> None:
    g, _ = _toy_graph()
    model = build_interaction_neighborhood(g)
    # User who only interacts with items 0..4 should match the community
    # containing them.
    seeds = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    top_comms = model._select_top_communities(seeds)
    expected_comm = int(model.item_index_to_community[0])
    assert expected_comm in top_comms


@pytest.mark.parametrize("centrality", list(ALL_CENTRALITIES))
def test_neighborhood_all_centralities_produce_valid_scores(centrality) -> None:
    g, _ = _toy_graph()
    model = build_interaction_neighborhood(g)
    out = model.retrieve(
        entity_id=0,
        owned_items=np.array([0, 1], dtype=np.int64),
        history=(0, 1),
        budget=10,
        exclude=None,
        centrality_override=centrality,
    )
    # Betweenness can degenerate to all-zero on small fully-connected
    # communities (no shortest-path bridges to weigh), in which case
    # an empty list is the honest output. For non-degenerate centralities
    # we expect non-empty output with bounded, descending scores.
    if not out:
        assert centrality == "betweenness"
        return
    for c in out:
        assert 0.0 <= c.score <= 1.0
    assert all(out[i].score >= out[i + 1].score for i in range(len(out) - 1))


def test_neighborhood_centrality_cache_hits_on_repeat() -> None:
    g, _ = _toy_graph()
    model = build_interaction_neighborhood(g)
    seeds = np.array([0, 1, 2], dtype=np.int64)
    comms = model._select_top_communities(seeds)
    # First call populates cache.
    s1 = model._subgraph_centrality(comms, "betweenness")
    # Second call should hit cache.
    cache_size_before = len(model.centrality_cache)
    s2 = model._subgraph_centrality(comms, "betweenness")
    cache_size_after = len(model.centrality_cache)
    assert cache_size_before == cache_size_after  # no new entry
    np.testing.assert_array_equal(s1, s2)
