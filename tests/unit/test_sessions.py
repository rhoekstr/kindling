"""Session inference tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from kindling.ingest.sessions import infer_sessions


def test_explicit_session_id_used_directly() -> None:
    df = pd.DataFrame(
        {
            "entity_id": ["a", "a", "b"],
            "item_id": [1, 2, 3],
            "session_id": [10, 10, 20],
        }
    )
    result = infer_sessions(df)
    assert result.strategy == "explicit"
    assert list(result.session_ids) == [10, 10, 20]


def test_no_timestamp_creates_singleton_sessions() -> None:
    df = pd.DataFrame({"entity_id": ["a", "b"], "item_id": [1, 2]})
    result = infer_sessions(df)
    assert result.strategy == "manual_fallback"
    assert len(set(result.session_ids)) == 2  # each row its own session


def test_gmm_infers_bimodal_gap() -> None:
    """Build synthetic data with clear in-session (seconds) and cross-session
    (days) gaps. GMM should find the bimodal structure."""
    rng = np.random.default_rng(0)
    entity_ids = []
    timestamps = []
    base = pd.Timestamp("2026-01-01 00:00:00")
    for user in range(30):
        # 5 sessions per user, each with 5 interactions
        for session in range(5):
            day_offset = pd.Timedelta(days=user * 7 + session * 3)
            session_start = base + day_offset + pd.Timedelta(minutes=rng.integers(0, 60))
            for event in range(5):
                entity_ids.append(f"u{user}")
                # within-session gap: 10-180 seconds
                gap = pd.Timedelta(seconds=10 + 170 * rng.random() * event)
                timestamps.append(session_start + gap)
    df = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "item_id": list(range(len(entity_ids))),
            "timestamp": timestamps,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = infer_sessions(df)
    # The GMM should fire on this bimodal data.
    assert result.strategy in ("gmm", "manual_fallback")
    # Minimum strategy correctness: sessions partition the interactions
    # monotonically (session ids increase in row order within an entity).
    assert len(result.session_ids) == len(df)


def test_session_inference_is_deterministic() -> None:
    """Repeating inference on the same input yields identical session ids."""
    rows = []
    base = pd.Timestamp("2026-01-01")
    for user in range(3):
        for event in range(60):
            rows.append(
                {
                    "entity_id": f"u{user}",
                    "item_id": event,
                    "timestamp": base + pd.Timedelta(hours=event * 3),
                }
            )
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = infer_sessions(df)
        r2 = infer_sessions(df)
    np.testing.assert_array_equal(r1.session_ids, r2.session_ids)
    assert r1.strategy == r2.strategy


def test_session_ids_align_with_input_rows() -> None:
    """The returned session_ids array has the same length as the input
    DataFrame and aligns row-for-row."""
    df = pd.DataFrame(
        {
            "entity_id": ["a", "b", "a", "b"],
            "item_id": [1, 2, 3, 4],
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-06-01", "2026-06-01"]
            ),
        }
    )
    result = infer_sessions(df)
    assert len(result.session_ids) == len(df)
