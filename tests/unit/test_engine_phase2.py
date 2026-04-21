"""Phase 2 Engine integration: path structures surface on the public API and
the debug payload carries per-signal information."""

from __future__ import annotations

import pandas as pd

from kindling import Engine


def test_engine_fits_with_timestamps() -> None:
    df = pd.DataFrame(
        {
            "entity_id": ["a", "a", "a", "b", "b"],
            "item_id": [1, 2, 3, 1, 2],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 10:02:00",
                    "2026-01-01 10:05:00",
                    "2026-01-02 12:00:00",
                    "2026-01-02 12:03:00",
                ]
            ),
        }
    )
    engine = Engine().fit(df)
    assert engine.tail_index.n_anchors > 0
    # Session inference ran
    assert engine.session_inference.strategy in {"gmm", "manual_fallback"}


def test_engine_surfaces_path_structures_even_without_timestamps() -> None:
    """Path tree and tail index are empty without timestamps; basket index
    still builds from item sets."""
    df = pd.DataFrame(
        {
            "entity_id": ["a", "a", "a", "b", "b"],
            "item_id": [1, 2, 3, 1, 2],
        }
    )
    engine = Engine().fit(df)
    assert engine.tail_index.n_anchors == 0
    assert engine.path_tree.n_prefixes == 0


def test_recommendations_carry_debug_signal_payload() -> None:
    df = pd.DataFrame(
        {
            "entity_id": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
            "item_id": [1, 2, 3, 1, 2, 4, 2, 3, 5],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 10:02:00",
                    "2026-01-01 10:05:00",
                    "2026-01-02 12:00:00",
                    "2026-01-02 12:03:00",
                    "2026-01-02 12:07:00",
                    "2026-01-03 09:00:00",
                    "2026-01-03 09:02:00",
                    "2026-01-03 09:05:00",
                ]
            ),
        }
    )
    engine = Engine().fit(df)
    recs = engine.recommend(entity_id="a", n=3)
    assert len(recs) > 0
    debug = recs[0].explanation.debug()
    # All path family + cooccurrence signals are reported
    assert "signals" in debug
    for name in ("path_full", "path_tail", "path_basket", "cooccurrence"):
        assert name in debug["signals"]
    # The dominant signal is recorded
    assert debug["dominant_signal"] in {
        "path_full",
        "path_tail",
        "path_basket",
        "cooccurrence",
    }


def test_recommend_never_returns_owned_items_phase2() -> None:
    """Regression test: with both retrievers active and the blend scoring,
    owned items must never appear in the output."""
    df = pd.DataFrame(
        {
            "entity_id": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            "item_id": [1, 2, 3, 4, 1, 2, 5, 6, 2, 3, 7, 8],
            "timestamp": pd.to_datetime(
                ["2026-01-01"] * 4 + ["2026-01-02"] * 4 + ["2026-01-03"] * 4
            ),
        }
    )
    engine = Engine().fit(df)
    owned_by_a = {1, 2, 3, 4}
    recs = engine.recommend(entity_id="a", n=5)
    for r in recs:
        assert r.item_id not in owned_by_a
