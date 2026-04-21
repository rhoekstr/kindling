"""Phase 2/3 Engine integration: path structures surface on the public API,
the debug payload carries per-signal information, and the Bayesian posterior
surfaces credible intervals and diagnostics."""

from __future__ import annotations

import pandas as pd

from kindling import Engine
from kindling.blend.likelihoods import (
    BinaryIndependent,
    MultinomialSoftmax,
    PairwiseBradleyTerry,
)


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


_PHASE3_DF = pd.DataFrame(
    {
        "entity_id": ["a"] * 6 + ["b"] * 6 + ["c"] * 6,
        "item_id": [1, 2, 3, 4, 5, 6, 1, 2, 4, 7, 8, 9, 2, 3, 5, 8, 10, 11],
        "timestamp": pd.to_datetime([f"2026-01-{i:02d}" for i in range(1, 7)] * 3),
    }
)


def test_engine_posterior_summary_populates() -> None:
    """After fit, posterior_summary() returns Bayesian blend stats and
    diagnostic results."""
    engine = Engine(vi_max_iter=50).fit(_PHASE3_DF)
    summary = engine.posterior_summary()
    assert summary["bayesian_blend_active"] is True
    # Phase 5: 4 positive signals + 3 cost signals = 7.
    assert len(summary["signal_names"]) == 7  # type: ignore[arg-type]
    assert len(summary["posterior_mean"]) == 7  # type: ignore[arg-type]
    ci = summary["credible_interval"]
    assert len(ci) == 7  # type: ignore[arg-type]
    assert "diagnostics" in summary


def test_recommendation_carries_credible_interval() -> None:
    engine = Engine(vi_max_iter=50).fit(_PHASE3_DF)
    recs = engine.recommend(entity_id="a", n=3)
    assert recs
    for r in recs:
        assert r.credible_interval is not None
        lower, upper = r.credible_interval
        assert lower <= upper
        assert r.credible_coverage == 0.9


def test_engine_accepts_alternative_likelihoods() -> None:
    """Engine should fit with any of the four likelihoods."""
    for likelihood in (
        BinaryIndependent(),
        PairwiseBradleyTerry(),
        MultinomialSoftmax(),
    ):
        engine = Engine(likelihood=likelihood, vi_max_iter=30).fit(_PHASE3_DF)
        recs = engine.recommend(entity_id="a", n=3)
        assert isinstance(recs, list)


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
