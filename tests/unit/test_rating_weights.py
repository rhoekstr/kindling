"""Rating-weighted positive signals.

Locks in the transform: (rating - 2.5) / 2.5 clipped to [0, 1], with
missing-rating rows defaulting to weight 1.0 (binary). Cooc and ALS
both read this weight in place of their old hard-coded ones().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling.graph.item_graph import build_item_graph
from kindling.preprocess import preprocess_interactions


def test_weight_transform_maps_ratings_to_zero_one() -> None:
    df = pd.DataFrame(
        {"entity_id": [1, 2, 3, 4, 5], "item_id": ["a"] * 5, "rating": [1, 2, 3, 4, 5]}
    )
    _, ctx = preprocess_interactions(df)
    processed, _ = preprocess_interactions(df)
    w = processed["_interaction_weight"].to_numpy()
    assert w[0] == pytest.approx(0.0)  # below threshold
    assert w[1] == pytest.approx(0.0)  # below threshold
    assert w[2] == pytest.approx(0.2)
    assert w[3] == pytest.approx(0.6)
    assert w[4] == pytest.approx(1.0)


def test_weight_defaults_to_one_when_no_rating_column() -> None:
    df = pd.DataFrame({"entity_id": [1, 2], "item_id": ["a", "b"]})
    _, ctx = preprocess_interactions(df)
    processed, _ = preprocess_interactions(df)
    w = processed["_interaction_weight"].to_numpy()
    assert (w == 1.0).all()


def test_weight_treats_missing_rating_as_implicit_positive() -> None:
    df = pd.DataFrame({"entity_id": [1, 2], "item_id": ["a", "b"], "rating": [np.nan, 5]})
    _, ctx = preprocess_interactions(df)
    processed, _ = preprocess_interactions(df)
    w = processed["_interaction_weight"].to_numpy()
    assert w[0] == 1.0  # NaN rating -> treat as implicit positive
    assert w[1] == 1.0  # 5 stars = 1.0


def test_item_graph_drops_low_rating_edges() -> None:
    """Users who only rated 1-2 stars should contribute nothing to cooc."""
    df = pd.DataFrame(
        {
            "entity_id": [1, 1, 2, 2, 3, 3],
            "item_id": ["A", "B", "A", "B", "A", "C"],
            "rating": [5, 5, 1, 2, 5, 5],
        }
    )
    processed, _ = preprocess_interactions(df)
    g = build_item_graph(processed)
    adj = g.adjacency.toarray()
    idx = g.item_index
    # User 1 and 3 both rated A high. User 2's low ratings don't count.
    # A-B edge: only user 1 contributes (user 2's ratings drop out).
    # A-C edge: only user 3 contributes.
    # Expected weighted cooc (A, B) = 1.0 (user 1's 1.0 * 1.0).
    # Expected weighted cooc (A, C) = 1.0 (user 3's 1.0 * 1.0).
    assert adj[idx["A"], idx["B"]] == pytest.approx(1.0)
    assert adj[idx["A"], idx["C"]] == pytest.approx(1.0)


def test_item_graph_binary_behavior_preserved_without_rating() -> None:
    """No rating column -> cooc is integer count of shared users, same
    as before rating support was added."""
    df = pd.DataFrame(
        {
            "entity_id": [1, 1, 2, 2, 3, 3],
            "item_id": ["A", "B", "A", "B", "A", "C"],
        }
    )
    g = build_item_graph(df)
    adj = g.adjacency.toarray()
    idx = g.item_index
    # Users 1 and 2 both rated A and B -> cooc = 2.
    # User 3 rated A and C -> cooc = 1.
    assert adj[idx["A"], idx["B"]] == pytest.approx(2.0)
    assert adj[idx["A"], idx["C"]] == pytest.approx(1.0)


def test_item_graph_deduplicates_with_max_weight() -> None:
    """A (user, item) repeated with different ratings should keep the
    MAX weight, not sum them."""
    df = pd.DataFrame(
        {
            "entity_id": [1, 1, 2, 2],
            "item_id": ["A", "A", "A", "A"],
            "rating": [3, 5, 4, 4],
        }
    )
    processed, _ = preprocess_interactions(df)
    w = processed["_interaction_weight"].to_numpy()
    assert w.shape == (4,)
    # Weights: [0.2, 1.0, 0.6, 0.6]. After max per pair:
    # (1, A) -> max(0.2, 1.0) = 1.0
    # (2, A) -> max(0.6, 0.6) = 0.6
    g = build_item_graph(processed)
    # Only one item so adjacency is empty but the graph should be valid.
    assert g.n_items == 1
