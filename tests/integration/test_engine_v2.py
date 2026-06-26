"""End-to-end coverage of the production (v2) engine public API.

Replaces the deleted v1 ``test_engine`` / ``test_pipeline`` suites. Uses
synthetic data (offline, fast). Covers fit → recommend, the
no-training new-user / anonymous path, the popularity fallback, cold
slots, and the regime-activation profile.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kindling import Engine, Recommendation
from kindling.loaders import synthetic


@pytest.fixture
def ratings():
    return synthetic.make_ratings(n_entities=120, n_items=80, ratings_per_entity=25, seed=0)


def _fit(split, **kw):
    eng = Engine(random_state=0, **kw)
    eng.fit(split.train)
    return eng


def test_fit_then_recommend_returns_recommendations(ratings):
    eng = _fit(ratings)
    entity = ratings.train["entity_id"].iloc[0]
    recs = eng.recommend(entity_id=entity, n=10)
    assert 0 < len(recs) <= 10
    assert all(isinstance(r, Recommendation) for r in recs)
    # Scores are finite and sorted descending.
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)
    assert all(r.item_id is not None for r in recs)


def test_recommend_is_deterministic(ratings):
    a = _fit(ratings).recommend(entity_id=ratings.train["entity_id"].iloc[0], n=10)
    b = _fit(ratings).recommend(entity_id=ratings.train["entity_id"].iloc[0], n=10)
    assert [r.item_id for r in a] == [r.item_id for r in b]


def test_small_catalog_selects_ease_base(ratings):
    eng = _fit(ratings)
    # n_items well under the 20k EASE gate → closed-form EASE base.
    assert eng._state.profile.get("base_scorer_used") == "ease"


def test_recommend_batch_matches_per_user_recommend(ratings):
    # recommend_batch (native Rust fast path when the extension is built; a
    # per-user Python loop otherwise) must produce the same lists as recommend.
    eng = _fit(ratings)
    ents = list(pd.Index(ratings.train["entity_id"].unique()))[:20]
    batch = eng.recommend_batch(ents, n=10)
    assert len(batch) == len(ents)
    for ent, recs in zip(ents, batch):
        ref = eng.recommend(entity_id=ent, n=10)
        assert [r.item_id for r in recs] == [r.item_id for r in ref]
        assert all(isinstance(r, Recommendation) for r in recs)


def test_recommend_batch_unknown_entity_returns_empty(ratings):
    eng = _fit(ratings)
    assert eng.recommend_batch(["__not_a_real_entity__"], n=10) == [[]]


def test_native_single_recommend_matches_python(ratings):
    # Engine.recommend defaults to the native path; it must match the Python
    # reference (_use_native=False) for the supported (EASE) fit.
    eng = _fit(ratings)
    ents = list(pd.Index(ratings.train["entity_id"].unique()))[:20]
    eng._use_native = False
    py = {e: [r.item_id for r in eng.recommend(e, 10)] for e in ents}
    eng._use_native = True
    for e in ents:
        assert [r.item_id for r in eng.recommend(e, 10)] == py[e]


def test_recommend_for_items_warm_seeds_personalize(ratings):
    eng = _fit(ratings)
    seeds = ratings.train["item_id"].value_counts().index[:3].tolist()
    recs = eng.recommend_for_items(seed_item_ids=seeds, n=10)
    assert 0 < len(recs) <= 10
    assert all(isinstance(r, Recommendation) for r in recs)


def test_recommend_for_items_empty_falls_back_to_popularity(ratings):
    eng = _fit(ratings)
    recs = eng.recommend_for_items(seed_item_ids=[], n=10)
    # Zero-seed cold start must still return the popularity list, not error.
    assert len(recs) == 10
    pop = ratings.train["item_id"].value_counts().index.tolist()
    assert recs[0].item_id in pop[:20]


def test_cold_slots_surface_metadata_only_items():
    # Warm interactions over items 0..39; metadata adds cold items 40..49.
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for u in range(150):
        for it in rng.choice(40, size=int(rng.integers(4, 12)), replace=False):
            rows.append(
                {
                    "entity_id": u,
                    "item_id": int(it),
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(it)),
                }
            )
    train = pd.DataFrame(rows)
    meta = pd.DataFrame(
        {
            "item_id": list(range(50)),
            "title": [f"item {i}" for i in range(50)],
            "genre": ["a" if i % 2 else "b" for i in range(50)],
        }
    )
    eng = Engine(cold_slots=1, open_catalog=True, random_state=0)
    eng.fit(train, item_metadata=meta)
    recs = eng.recommend(entity_id=5, n=10)
    assert len(recs) == 10
    # The reserved cold slot can surface an unseen (>=40) metadata-only item.
    assert any(r.item_id >= 40 for r in recs)
