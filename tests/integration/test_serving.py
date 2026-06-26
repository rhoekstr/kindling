"""KindlingServer — serving artifact round-trip + parity with the engine."""

from __future__ import annotations

import pandas as pd
import pytest

from kindling import Engine
from kindling._native import CORE_AVAILABLE
from kindling.loaders import synthetic
from kindling.serving import KindlingServer

pytestmark = pytest.mark.skipif(not CORE_AVAILABLE, reason="native extension not built")


@pytest.fixture(scope="module")
def fitted():
    s = synthetic.make_ratings(n_entities=120, n_items=80, ratings_per_entity=25, seed=0)
    return Engine(random_state=0).fit(s.train), s


def test_server_recommend_matches_engine(fitted):
    eng, s = fitted
    server = KindlingServer.from_engine(eng)
    ents = list(pd.Index(s.train["entity_id"].unique()))[:25]
    for e in ents:
        got = [r.item_id for r in server.recommend(e, 10)]
        want = [r.item_id for r in eng.recommend(e, 10)]
        assert got == want


def test_server_save_load_roundtrip(fitted, tmp_path):
    eng, s = fitted
    KindlingServer.from_engine(eng).save(tmp_path / "artifact")
    server = KindlingServer.load(tmp_path / "artifact")
    ents = list(pd.Index(s.train["entity_id"].unique()))[:25]
    # batch == per-user, and both match the original engine
    batch = server.recommend_batch(ents, 10)
    for e, recs in zip(ents, batch):
        loaded = [r.item_id for r in server.recommend(e, 10)]
        assert [r.item_id for r in recs] == loaded
        assert loaded == [r.item_id for r in eng.recommend(e, 10)]


def test_server_cold_and_seed_paths(fitted, tmp_path):
    eng, s = fitted
    server = KindlingServer.from_engine(eng)
    # unknown entity -> popularity fallback
    cold = server.recommend("__nobody__", 10)
    assert 0 < len(cold) <= 10
    assert all(r.base_kind == "cold_popularity" for r in cold)
    # anonymous seed-based serving is stateless and personalized
    seeds = list(pd.Index(s.train["item_id"].unique()))[:3]
    recs = server.recommend_for_items(seeds, 10)
    assert 0 < len(recs) <= 10
    # empty seeds -> popularity
    assert all(r.base_kind == "cold_popularity" for r in server.recommend_for_items([], 5))
