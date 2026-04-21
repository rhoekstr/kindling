"""Engine smoke tests — the three-stage pipeline runs and respects its contract."""

from __future__ import annotations

import pandas as pd
import pytest

from kindling import Engine
from kindling.engine import EngineNotFittedError


def test_recommend_before_fit_raises() -> None:
    engine = Engine()
    with pytest.raises(EngineNotFittedError):
        engine.recommend(entity_id="anyone")


def test_recommend_returns_items(tiny_interactions: pd.DataFrame) -> None:
    engine = Engine().fit(tiny_interactions)
    recs = engine.recommend(entity_id="a", n=5)
    assert len(recs) > 0
    # Entity a owns {1, 2, 3} — none of these should appear in recs
    for rec in recs:
        assert rec.item_id not in {1, 2, 3}
        assert rec.explanation.primary


def test_recommend_respects_n(tiny_interactions: pd.DataFrame) -> None:
    engine = Engine().fit(tiny_interactions)
    recs = engine.recommend(entity_id="a", n=2)
    assert len(recs) <= 2


def test_recommend_unknown_entity_returns_empty(
    tiny_interactions: pd.DataFrame,
) -> None:
    engine = Engine().fit(tiny_interactions)
    # No co-occurrence signal available, so the retriever returns nothing.
    assert engine.recommend(entity_id="unknown_entity", n=5) == []


def test_recommend_with_constraint(tiny_interactions: pd.DataFrame) -> None:
    engine = Engine().fit(tiny_interactions)
    # Rule out item 4 explicitly
    recs = engine.recommend(
        entity_id="a",
        n=5,
        constraints=[lambda item: item != 4],
    )
    assert all(r.item_id != 4 for r in recs)


def test_recommend_scores_sorted(tiny_interactions: pd.DataFrame) -> None:
    engine = Engine().fit(tiny_interactions)
    recs = engine.recommend(entity_id="a", n=10)
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_data_density_report(tiny_interactions: pd.DataFrame) -> None:
    engine = Engine().fit(tiny_interactions)
    density = engine.data_density()
    assert density["n_items"] == 5
    assert density["n_entities"] == 4
    assert density["n_interactions"] == 9
    assert 0.0 <= density["graph_density"] <= 1.0


def test_fit_is_chainable(tiny_interactions: pd.DataFrame) -> None:
    engine = Engine()
    returned = engine.fit(tiny_interactions)
    assert returned is engine
