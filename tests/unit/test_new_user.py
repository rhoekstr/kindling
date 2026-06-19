"""Tests for new-user / anonymous serving — recommend_for_items + cold fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling.engine_v2 import EngineV2


def _genre_data(seed: int = 0, per_genre: int = 25, n_users: int = 800):
    """Two genres; users consume within one genre → genre-block co-occurrence,
    so a seed item's EASE/cooc neighbors are same-genre."""
    rng = np.random.default_rng(seed)
    warm = [f"w{g}_{i}" for g in (0, 1) for i in range(per_genre)]
    rows = []
    for u in range(n_users):
        g = u % 2
        pool = [it for it in warm if it.startswith(f"w{g}_")]
        for it in rng.choice(pool, size=int(rng.integers(6, 14)), replace=False):
            rows.append((u, it))
    return pd.DataFrame(rows, columns=["entity_id", "item_id"])


@pytest.fixture(scope="module")
def engine():
    return EngineV2(persona_min_users=10**9, random_state=0).fit(_genre_data())


def test_new_user_from_seeds_is_personalized(engine):
    # A brand-new user (never in train) seeded with genre-0 items gets
    # genre-0 recommendations — no per-user training needed.
    recs = engine.recommend_for_items(["w0_1", "w0_2", "w0_3"], n=5)
    assert len(recs) == 5
    assert all(r.item_id.startswith("w0_") for r in recs)
    assert all(r.base_kind == "ease" for r in recs)


def test_new_user_seeds_excluded_from_results(engine):
    seeds = ["w1_0", "w1_1", "w1_2"]
    recs = engine.recommend_for_items(seeds, n=10)
    assert not (set(seeds) & {r.item_id for r in recs})
    assert all(r.item_id.startswith("w1_") for r in recs)


def test_zero_seeds_fall_back_to_popularity(engine):
    recs = engine.recommend_for_items([], n=5)
    assert len(recs) == 5
    assert all(r.base_kind == "cold_popularity" for r in recs)
    # popularity is descending
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_all_unknown_seeds_fall_back(engine):
    recs = engine.recommend_for_items(["ghost1", "ghost2"], n=5)
    assert recs
    assert all(r.base_kind == "cold_popularity" for r in recs)


def test_mixed_known_unknown_uses_known(engine):
    recs = engine.recommend_for_items(["w0_4", "ghost", "w0_5"], n=5)
    assert all(r.item_id.startswith("w0_") for r in recs)
    assert all(r.base_kind == "ease" for r in recs)  # not the fallback


def test_known_entity_recommend_unchanged(engine):
    # The refactor (recommend -> _recommend_core) must not change known-user
    # behavior: entity 0 (genre 0) still gets genre-0 recs.
    recs = engine.recommend(0, n=5)
    assert len(recs) == 5
    assert all(r.item_id.startswith("w0_") for r in recs)


def test_recommend_for_items_requires_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        EngineV2().recommend_for_items(["x"], n=5)
