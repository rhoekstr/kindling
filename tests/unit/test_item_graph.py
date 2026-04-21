"""Item graph and co-occurrence tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kindling.graph.item_graph import build_item_graph


def test_empty_interactions() -> None:
    graph = build_item_graph(pd.DataFrame({"entity_id": [], "item_id": []}))
    assert graph.n_items == 0
    assert graph.n_edges == 0


def test_single_entity_single_item_has_no_edges() -> None:
    df = pd.DataFrame({"entity_id": ["a"], "item_id": [1]})
    graph = build_item_graph(df)
    assert graph.n_items == 1
    assert graph.n_edges == 0


def test_co_occurrence_count(tiny_interactions: pd.DataFrame) -> None:
    graph = build_item_graph(tiny_interactions)
    # entity a owns {1, 2, 3}; b owns {1, 2, 4}; c owns {5}; d owns {3, 4}
    # items 1 and 2 co-occur in entities a and b -> weight 2
    # items 1 and 3 co-occur in entity a only     -> weight 1
    # items 3 and 4 co-occur in entity d only     -> weight 1
    # items 1 and 4 co-occur in entity b only     -> weight 1
    adj = graph.adjacency
    idx = graph.item_index
    assert adj[idx[1], idx[2]] == 2
    assert adj[idx[1], idx[3]] == 1
    assert adj[idx[3], idx[4]] == 1
    # item 5 is a singleton; no edges touching it
    assert adj.getrow(idx[5]).sum() == 0


def test_adjacency_is_symmetric(tiny_interactions: pd.DataFrame) -> None:
    graph = build_item_graph(tiny_interactions)
    diff = (graph.adjacency - graph.adjacency.T).nnz
    assert diff == 0


def test_diagonal_is_zero(tiny_interactions: pd.DataFrame) -> None:
    graph = build_item_graph(tiny_interactions)
    assert graph.adjacency.diagonal().sum() == 0


def test_neighbors_ordered_by_weight(tiny_interactions: pd.DataFrame) -> None:
    graph = build_item_graph(tiny_interactions)
    neighbors = graph.neighbors(1, top_k=5)
    # item 2 is the strongest co-occurrence partner of item 1 (weight 2)
    assert neighbors.iloc[0]["item_id"] == 2
    assert neighbors.iloc[0]["weight"] == 2
    # weights should be non-increasing
    weights = neighbors["weight"].to_numpy()
    assert np.all(np.diff(weights) <= 0)


def test_duplicate_entity_item_doesnt_inflate_count() -> None:
    """A user who interacts with the same item twice should not count twice
    toward co-occurrence — we drop_duplicates in build_item_graph."""
    df = pd.DataFrame(
        {
            "entity_id": ["a", "a", "a", "a", "b", "b"],
            "item_id": [1, 1, 2, 3, 1, 2],
        }
    )
    graph = build_item_graph(df)
    idx = graph.item_index
    # (1, 2) should co-occur in a and b -> weight 2, not 3 or 4
    assert graph.adjacency[idx[1], idx[2]] == 2


def test_unknown_item_neighbors() -> None:
    graph = build_item_graph(pd.DataFrame({"entity_id": ["a", "a"], "item_id": [1, 2]}))
    neighbors = graph.neighbors("nonexistent", top_k=5)
    assert neighbors.empty
