"""Save / load a fitted v2 engine and recommend identically."""

from __future__ import annotations

import json

import pytest

from kindling import Engine
from kindling.loaders import synthetic
from kindling.persist import FORMAT_VERSION, load_engine


@pytest.fixture
def fitted(tmp_path):
    s = synthetic.make_ratings(n_entities=120, n_items=80, seed=0)
    e = Engine(random_state=0)
    e.fit(s.train)
    return e, s, tmp_path


def _recs(engine, ent):
    return [(r.item_id, round(r.score, 8)) for r in engine.recommend(entity_id=ent, n=10)]


def test_save_load_recommends_identically(fitted):
    e, s, tmp = fitted
    ent = s.train.entity_id.iloc[0]
    before = _recs(e, ent)
    path = tmp / "engine.kdl"
    e.save(path)
    loaded = Engine.load(path)
    assert _recs(loaded, ent) == before


def test_save_load_preserves_cold_user_and_activation(fitted):
    e, s, tmp = fitted
    path = tmp / "engine.kdl"
    e.save(path)
    loaded = Engine.load(path)
    seeds = list(s.train.item_id.iloc[:3])
    assert len(loaded.recommend_for_items(seed_item_ids=seeds, n=5)) == 5
    assert loaded.activation_plan.base_scorer == e.activation_plan.base_scorer


def test_header_records_format_and_version(fitted):
    e, _, tmp = fitted
    path = tmp / "engine.kdl"
    e.save(path)
    with open(path, "rb") as fh:
        header = json.loads(fh.readline())
    assert header["magic"] == "kindling-engine"
    assert header["format_version"] == FORMAT_VERSION


def test_unfitted_engine_refuses_to_save(tmp_path):
    with pytest.raises(RuntimeError, match="not fitted"):
        Engine().save(tmp_path / "x.kdl")


def test_non_engine_file_rejected(tmp_path):
    p = tmp_path / "bogus.kdl"
    p.write_bytes(b"not a kindling file\n")
    with pytest.raises(ValueError, match="not a kindling engine"):
        load_engine(p)


def test_incompatible_format_version_rejected(tmp_path):
    p = tmp_path / "future.kdl"
    p.write_bytes(json.dumps({"magic": "kindling-engine", "format_version": 999}).encode() + b"\n")
    with pytest.raises(ValueError, match="Unsupported engine format"):
        load_engine(p)
