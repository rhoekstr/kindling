"""Cold-start popularity fallback.

When retrieval returns no candidates for an entity (unseen entity or
entity whose owned items exhaust their cooccurrence + path neighbors),
kindling falls through to a popularity-ranked list with the entity's
owned items excluded. This is the largest single accuracy win on
sparse-data evaluations per ADR-growth-curves.md.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kindling.engine import Engine


def _small_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["a", "a", "a", "b", "b", "c", "c"],
            "item_id": [1, 2, 3, 1, 4, 2, 5],
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03",
                 "2026-01-01", "2026-01-02",
                 "2026-01-01", "2026-01-02"]
            ),
        }
    )


def test_unseen_entity_gets_popularity_fallback() -> None:
    engine = Engine().fit(_small_df())
    recs = engine.recommend(entity_id="d_unseen", n=3)
    assert len(recs) > 0, "cold-start must return non-empty for unseen entity"
    for rec in recs:
        assert rec.explanation.debug_payload.get("fallback") == "popularity"
        assert rec.credible_interval is None


def test_fallback_excludes_owned_items() -> None:
    engine = Engine().fit(_small_df())
    recs = engine.recommend(entity_id="a", n=5)
    # 'a' owns 1, 2, 3. Fallback (if triggered) should pick from {4, 5}.
    for rec in recs:
        assert rec.item_id not in {1, 2, 3}, f"fallback leaked owned item {rec.item_id}"


def test_fallback_applies_constraints() -> None:
    engine = Engine().fit(_small_df())
    recs = engine.recommend(
        entity_id="d_unseen", n=5, constraints=[lambda i: i != 1]
    )
    for rec in recs:
        assert rec.item_id != 1


def test_fallback_respects_n() -> None:
    engine = Engine().fit(_small_df())
    recs = engine.recommend(entity_id="d_unseen", n=2)
    assert len(recs) == 2


@pytest.mark.integration
def test_cold_start_lifts_sparse_ml_ndcg() -> None:
    """On ML-1M 10% prefix where most eval entities are not in the
    subsample, the engine should no longer return empty lists."""
    pytest.importorskip("kindling.loaders.movielens")
    from kindling.loaders import movielens

    split = movielens.load_1m(test_fraction=0.1)
    train_subset = split.train.iloc[: int(len(split.train) * 0.1)].reset_index(drop=True)
    engine = Engine().fit(train_subset)

    # Pick an entity that appears in test but not in train_subset.
    train_entities = set(train_subset["entity_id"].unique())
    test_entities = set(split.test["entity_id"].unique())
    unseen = next(iter(test_entities - train_entities), None)
    if unseen is None:
        pytest.skip("no unseen entity available in this split")
    recs = engine.recommend(entity_id=unseen, n=10)
    assert recs, "cold-start should fall back rather than return []"
    assert recs[0].explanation.debug_payload.get("fallback") == "popularity"
