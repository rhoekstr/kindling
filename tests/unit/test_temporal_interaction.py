"""Tests for the temporal interaction graph substrate.

Covers:
- Pure-count fallback when timestamps are missing (kernel = 1).
- GMM-based kernel calibration when timestamps + bimodal sessions exist.
- Symmetric adjacency, zero diagonal.
- Per-user history cap.
- Two-pointer walk truncates at the kernel cutoff.
- Items not in the catalog index are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling.graph.temporal_interaction import (
    KernelParams,
    build_temporal_interaction_graph,
    calibrate_kernel,
)


def _df(rows):
    return pd.DataFrame(rows)


def test_pure_count_fallback_no_timestamp() -> None:
    df = _df([
        {"entity_id": 1, "item_id": 10},
        {"entity_id": 1, "item_id": 11},
        {"entity_id": 2, "item_id": 10},
        {"entity_id": 2, "item_id": 12},
    ])
    kp = calibrate_kernel(df)
    assert kp.pure_count is True
    assert kp.strategy == "pure_count"
    # kernel is identically 1.
    assert float(kp.kernel(np.float64(0))) == 1.0
    assert float(kp.kernel(np.float64(86400 * 365))) == 1.0


def test_gmm_calibration_with_bimodal_sessions() -> None:
    # 100 users, ~3 sessions of ~5 items, sessions hours apart.
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2024-01-01").value // 10**9
    rows = []
    for u in range(100):
        for s in range(3):
            session_start = base + (u * 86400 * 30) + (s * 86400 * 2) + rng.integers(0, 600)
            session_items = rng.integers(0, 50, size=5)
            for j, item in enumerate(session_items):
                rows.append({
                    "entity_id": u,
                    "item_id": int(item),
                    "timestamp": pd.to_datetime(session_start + j * 60, unit="s"),
                })
    df = _df(rows)
    kp = calibrate_kernel(df)
    # GMM should detect bimodality; midpoint should land between session
    # interior (~minutes) and across-session (~days).
    assert kp.strategy == "gmm"
    assert not kp.pure_count
    assert 60 < kp.midpoint_seconds < 86400
    # kernel should be ~1 within sessions and ~0 across sessions.
    assert float(kp.kernel(np.float64(60))) > 0.9
    assert float(kp.kernel(np.float64(86400 * 7))) < 0.01


def test_symmetric_adjacency_zero_diagonal() -> None:
    df = _df([
        {"entity_id": 1, "item_id": 0},
        {"entity_id": 1, "item_id": 1},
        {"entity_id": 1, "item_id": 2},
    ])
    g = build_temporal_interaction_graph(df, item_index={0: 0, 1: 1, 2: 2})
    A = g.adjacency.toarray()
    np.testing.assert_array_equal(A, A.T)
    np.testing.assert_array_equal(np.diag(A), np.zeros(3))


def test_per_user_history_cap() -> None:
    # User has 50 interactions; cap to 10. Only the most recent 10 should
    # contribute pairs.
    base = pd.Timestamp("2024-01-01").value // 10**9
    rows = [
        {
            "entity_id": 1,
            "item_id": i,
            "timestamp": pd.to_datetime(base + i * 60, unit="s"),
        }
        for i in range(50)
    ]
    item_index = {i: i for i in range(50)}
    g_full = build_temporal_interaction_graph(_df(rows), item_index, max_history_per_user=50)
    g_capped = build_temporal_interaction_graph(_df(rows), item_index, max_history_per_user=10)
    # Capped version emits fewer pairs.
    assert g_capped.n_pairs_generated < g_full.n_pairs_generated


def test_drops_items_outside_index() -> None:
    df = _df([
        {"entity_id": 1, "item_id": 0},
        {"entity_id": 1, "item_id": 99},  # not in index
        {"entity_id": 1, "item_id": 1},
    ])
    g = build_temporal_interaction_graph(df, item_index={0: 0, 1: 1, 2: 2})
    # Item 99 silently dropped; only (0, 1) pair remains.
    A = g.adjacency.toarray()
    assert A[0, 1] > 0
    assert A[0, 2] == 0
    assert A[1, 2] == 0


def test_kernel_cutoff_truncates_distant_pairs() -> None:
    # Two events 1 hour apart, then a third 10 days later. Cutoff should
    # exclude the (event1, event3) and (event2, event3) pairs.
    base = pd.Timestamp("2024-01-01").value // 10**9
    df = _df([
        {"entity_id": 1, "item_id": 0, "timestamp": pd.to_datetime(base, unit="s")},
        {"entity_id": 1, "item_id": 1, "timestamp": pd.to_datetime(base + 3600, unit="s")},
        {"entity_id": 1, "item_id": 2, "timestamp": pd.to_datetime(base + 86400 * 10, unit="s")},
    ])
    # Manual narrow kernel: 1-hour midpoint, very steep.
    kp = KernelParams(midpoint_seconds=3600, steepness_seconds=600, pure_count=False, strategy="manual_fallback")
    g = build_temporal_interaction_graph(df, item_index={0: 0, 1: 1, 2: 2}, kernel_params=kp)
    A = g.adjacency.toarray()
    assert A[0, 1] > 0  # 1 hour - inside the kernel midpoint
    assert A[0, 2] == 0  # 10 days - past the cutoff
    assert A[1, 2] == 0


def test_empty_input_produces_empty_graph() -> None:
    df = _df([{"entity_id": 1, "item_id": 0}])  # single event, no pairs possible
    g = build_temporal_interaction_graph(df, item_index={0: 0, 1: 1})
    assert g.n_edges == 0
    assert g.n_users_contributed == 1
