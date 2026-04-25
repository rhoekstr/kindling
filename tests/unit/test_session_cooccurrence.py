"""Tests for the session-row item-cooccurrence graph."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling.graph.session_cooccurrence import (
    SessionCooccurrenceGraph,
    build_session_cooccurrence_graph,
)


def _df(rows):
    return pd.DataFrame(rows)


def test_session_cooc_counts_session_co_membership() -> None:
    """adjacency[i,j] = number of sessions containing both items."""
    df = _df([
        # session A: items 0, 1
        {"entity_id": 1, "item_id": 0},
        {"entity_id": 1, "item_id": 1},
        # session B: items 0, 2
        {"entity_id": 1, "item_id": 0},
        {"entity_id": 1, "item_id": 2},
        # session C: items 0, 1, 2 (all three)
        {"entity_id": 2, "item_id": 0},
        {"entity_id": 2, "item_id": 1},
        {"entity_id": 2, "item_id": 2},
    ])
    session_ids = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    g = build_session_cooccurrence_graph(
        df, item_index={0: 0, 1: 1, 2: 2}, session_ids=session_ids,
    )
    assert g is not None
    A = g.adjacency.toarray()
    # (0, 1) co-occur in sessions A and C → adjacency[0, 1] == 2
    # (0, 2) co-occur in sessions B and C → adjacency[0, 2] == 2
    # (1, 2) co-occur in session C only   → adjacency[1, 2] == 1
    assert A[0, 1] == 2.0
    assert A[0, 2] == 2.0
    assert A[1, 2] == 1.0
    np.testing.assert_array_equal(A, A.T)
    assert A[0, 0] == 0.0  # diagonal zeroed


def test_deep_session_gate_skips_singleton_heavy_data() -> None:
    """When most sessions are 1-item, the graph builder returns None."""
    # 10 sessions all with single items - 0% deep.
    df = _df([{"entity_id": i, "item_id": i % 5} for i in range(10)])
    session_ids = np.arange(10, dtype=np.int64)
    g = build_session_cooccurrence_graph(
        df, item_index={i: i for i in range(5)}, session_ids=session_ids,
    )
    assert g is None  # gate triggered


def test_deep_session_gate_threshold_configurable() -> None:
    """Lower the gate to 0 and force a build even on shallow data."""
    df = _df([{"entity_id": i, "item_id": i % 5} for i in range(10)])
    session_ids = np.arange(10, dtype=np.int64)
    g = build_session_cooccurrence_graph(
        df, item_index={i: i for i in range(5)}, session_ids=session_ids,
        min_deep_session_fraction=0.0,
    )
    # All sessions are singletons → no co-occurrences → graph is empty.
    assert g is not None
    assert g.deep_session_fraction == 0.0
    assert g.n_edges == 0


def test_session_cooc_drops_items_outside_index() -> None:
    df = _df([
        {"entity_id": 1, "item_id": 0},
        {"entity_id": 1, "item_id": 99},  # not in index
        {"entity_id": 1, "item_id": 1},
    ])
    session_ids = np.array([0, 0, 0])
    g = build_session_cooccurrence_graph(
        df, item_index={0: 0, 1: 1, 2: 2}, session_ids=session_ids,
    )
    assert g is not None
    A = g.adjacency.toarray()
    assert A[0, 1] == 1.0  # 0,1 co-occur in session 0
    assert A[0, 2] == 0.0  # item 2 not in this session


def test_score_against_owned() -> None:
    """Direct lookup mirrors cooccurrence's scoring shape."""
    df = _df([
        {"entity_id": 1, "item_id": 0},
        {"entity_id": 1, "item_id": 1},  # session 0: items 0, 1
        {"entity_id": 2, "item_id": 0},
        {"entity_id": 2, "item_id": 2},  # session 1: items 0, 2
    ])
    session_ids = np.array([0, 0, 1, 1])
    g = build_session_cooccurrence_graph(
        df, item_index={0: 0, 1: 1, 2: 2}, session_ids=session_ids,
    )
    # Owned = {0}: score for item 1 = 1, score for item 2 = 1, item 0 excluded.
    scores = g.score_against_owned(np.array([0]), exclude_indices={0})
    assert scores[0] == 0.0
    assert scores[1] == scores[2]  # both co-occur with 0 once → tie after norm
    assert scores[1] > 0.0


def test_rating_burst_skipped() -> None:
    """When the session inference strategy is GMM with a sub-300s gap
    (rating-burst signature, like ml1m), the builder skips the build
    so we don't amplify UI-click-burst noise."""
    df = _df([
        {"entity_id": 1, "item_id": 0},
        {"entity_id": 1, "item_id": 1},
        {"entity_id": 1, "item_id": 2},
    ])
    session_ids = np.array([0, 0, 0])
    g = build_session_cooccurrence_graph(
        df,
        item_index={0: 0, 1: 1, 2: 2},
        session_ids=session_ids,
        session_strategy="gmm",
        session_gap_seconds=87.0,  # ml1m's actual rating-burst midpoint
    )
    assert g is None


def test_explicit_sessions_pass_burst_check() -> None:
    """Explicit session_id always builds (no rating-burst suspicion)."""
    df = _df([
        {"entity_id": 1, "item_id": 0},
        {"entity_id": 1, "item_id": 1},
    ])
    g = build_session_cooccurrence_graph(
        df,
        item_index={0: 0, 1: 1},
        session_ids=np.array([0, 0]),
        session_strategy="explicit",
        session_gap_seconds=0.0,  # explicit doesn't infer a gap
    )
    assert g is not None


def test_diagnostics_populated() -> None:
    """n_sessions, median_session_size, deep_session_fraction populated."""
    df = _df([
        {"entity_id": 1, "item_id": 0},
        {"entity_id": 1, "item_id": 1},
        {"entity_id": 1, "item_id": 2},
        {"entity_id": 2, "item_id": 0},
        {"entity_id": 2, "item_id": 1},
    ])
    session_ids = np.array([0, 0, 0, 1, 1])
    g = build_session_cooccurrence_graph(
        df, item_index={0: 0, 1: 1, 2: 2}, session_ids=session_ids,
    )
    assert g is not None
    assert g.n_sessions == 2
    assert g.median_session_size == 2.5  # sizes [3, 2]
    assert g.deep_session_fraction == 1.0  # both sessions have ≥2 items
