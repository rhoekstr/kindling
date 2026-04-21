"""Property-based invariants on the item graph.

Invariants from the plan's testing strategy:
- In positive_only mode (Phase 1), the adjacency is symmetric.
- Diagonal is always zero.
- For any input, neighbors() returns weights in non-increasing order.
- Adjacency entry (i, j) equals the number of entities owning both items
  (for the binary implicit-feedback mode we ship today).

The PRD's "symmetry of co-occurrence" example is only valid because Phase 1
is positive_only with no action_type differentiation. Phase 5 introduces the
cost graph and these properties split: the item graph stays symmetric,
the cost graph is directional. Property tests are versioned accordingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from kindling.graph.item_graph import build_item_graph


def _interactions(entity_count: int, item_count: int, rows: list[tuple[int, int]]) -> pd.DataFrame:
    entities = [f"e{e}" for e, _ in rows]
    items = [f"i{i}" for _, i in rows]
    if not entities:
        return pd.DataFrame({"entity_id": [], "item_id": []})
    return pd.DataFrame({"entity_id": entities, "item_id": items})


interaction_rows = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=8),
        st.integers(min_value=0, max_value=8),
    ),
    min_size=0,
    max_size=50,
)


@given(rows=interaction_rows)
@settings(max_examples=100, deadline=None)
def test_adjacency_is_symmetric(rows: list[tuple[int, int]]) -> None:
    df = _interactions(9, 9, rows)
    graph = build_item_graph(df)
    if graph.n_items == 0:
        return
    diff = (graph.adjacency - graph.adjacency.T).nnz
    assert diff == 0


@given(rows=interaction_rows)
@settings(max_examples=100, deadline=None)
def test_diagonal_is_zero(rows: list[tuple[int, int]]) -> None:
    df = _interactions(9, 9, rows)
    graph = build_item_graph(df)
    if graph.n_items == 0:
        return
    assert graph.adjacency.diagonal().sum() == 0


@given(rows=interaction_rows)
@settings(max_examples=100, deadline=None)
def test_adjacency_nonnegative(rows: list[tuple[int, int]]) -> None:
    df = _interactions(9, 9, rows)
    graph = build_item_graph(df)
    if graph.n_items == 0:
        return
    assert (graph.adjacency.data >= 0).all()


@given(rows=interaction_rows)
@settings(max_examples=50, deadline=None)
def test_neighbors_are_sorted(rows: list[tuple[int, int]]) -> None:
    df = _interactions(9, 9, rows)
    graph = build_item_graph(df)
    if graph.n_items == 0:
        return
    item = graph.item_ids[0]
    neighbors = graph.neighbors(item, top_k=10)
    if len(neighbors) < 2:
        return
    weights = neighbors["weight"].to_numpy()
    assert np.all(np.diff(weights) <= 0)


@given(rows=interaction_rows)
@settings(max_examples=50, deadline=None)
def test_cooccurrence_equals_entity_overlap_count(
    rows: list[tuple[int, int]],
) -> None:
    """Golden property: adjacency[i, j] = |{e : e owns both i and j}|.

    This is the definition. We compute both ways and compare.
    """
    df = _interactions(9, 9, rows)
    graph = build_item_graph(df)
    if graph.n_items < 2:
        return

    # Brute force: for each pair (i, j), count entities that own both.
    pairs = df.drop_duplicates()
    owned_by: dict[str, set[str]] = {}
    for _, row in pairs.iterrows():
        owned_by.setdefault(row["entity_id"], set()).add(row["item_id"])

    for idx_i in range(min(graph.n_items, 4)):  # sample for speed
        for idx_j in range(idx_i + 1, min(graph.n_items, 4)):
            item_i = graph.item_ids[idx_i]
            item_j = graph.item_ids[idx_j]
            expected = sum(1 for owned in owned_by.values() if item_i in owned and item_j in owned)
            actual = float(graph.adjacency[idx_i, idx_j])
            assert actual == expected, (item_i, item_j, actual, expected)
