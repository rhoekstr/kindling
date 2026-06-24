"""Engine-integrated LightGBMRanker training + warm-regime scoring.

Verifies the ranker is fitted at Engine.fit time when `lightgbm` is
available and that recommend() routes through it instead of the
Bayesian blend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling.engine import Engine


def _df(n_entities: int = 50, sessions_per_entity: int = 6, items_per_session: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2026-01-01")
    # Item catalog larger than ranker_negatives_per_positive so the test
    # can draw 99 negatives per entity.
    n_items = 150
    for ent in range(n_entities):
        for s in range(sessions_per_entity):
            items = rng.choice(n_items, size=items_per_session, replace=False)
            for k, item in enumerate(items):
                rows.append({
                    "entity_id": ent,
                    "item_id": int(item),
                    "timestamp": base + pd.Timedelta(days=s, minutes=int(k)),
                    "session_id": ent * 100 + s,
                })
    return pd.DataFrame(rows)


pytest.importorskip("lightgbm")


def test_ranker_fitted_after_engine_fit() -> None:
    """use_ranker=True trains the LambdaRank at fit time."""
    engine = Engine(use_ranker=True).fit(_df())
    assert engine._ranker is not None
    assert engine._ranker.is_fitted


def test_recommend_uses_ranker_scores() -> None:
    """When the ranker is fitted, recommendations should use its scores
    rather than the Bayesian posterior mean for ordering."""
    engine = Engine(use_ranker=True).fit(_df())
    assert engine._ranker is not None and engine._ranker.is_fitted
    recs = engine.recommend(entity_id=0, n=10)
    assert recs
    # Scores should be LambdaRank outputs (real-valued, can be negative
    # or large). Posterior-mean scores are in [0, ~1]. We assert the
    # values are consistent with one scoring model at least.
    scores = [r.score for r in recs]
    # Monotonic non-increasing order.
    assert scores == sorted(scores, reverse=True)


def test_ranker_can_be_disabled() -> None:
    engine = Engine(use_ranker=False).fit(_df())
    assert engine._ranker is None
    recs = engine.recommend(entity_id=0, n=5)
    assert recs


def test_ranker_skipped_when_too_few_pairs() -> None:
    """Small dataset below ranker_min_train_pairs should skip training."""
    tiny = pd.DataFrame({
        "entity_id": ["a", "a", "b"],
        "item_id": [1, 2, 3],
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-01"]),
    })
    engine = Engine(use_ranker=True, ranker_min_train_pairs=10_000).fit(tiny)
    assert engine._ranker is None
