"""Golden-output regression: exact recommendations on a frozen dataset.

The new anti-drift anchor (the v1<->v2 differential test was removed with
v1). A fixed synthetic dataset + fixed config must always produce the same
top-K item ids and scores. If this breaks, the scoring path changed — make
sure that was intentional and re-freeze.
"""

from __future__ import annotations

import pytest

from kindling import Engine
from kindling.loaders import synthetic

# Frozen on the v2 EASE stack (make_ratings seed=7, Engine random_state=0).
GOLDEN = {
    0: [(14, 2.8806), (16, 2.5699), (12, 1.9998), (6, 0.7571), (3, 0.4295)],
    1: [(8, 3.3067), (16, 1.5263), (37, 1.1754), (29, 1.0394), (14, 0.8303)],
}


@pytest.fixture(scope="module")
def engine():
    s = synthetic.make_ratings(n_entities=60, n_items=40, ratings_per_entity=15, seed=7)
    e = Engine(random_state=0)
    e.fit(s.train)
    return e


def test_base_is_ease(engine):
    assert engine._state.profile.get("base_scorer_used") == "ease"


@pytest.mark.parametrize("entity", list(GOLDEN))
def test_golden_recommendations(engine, entity):
    recs = engine.recommend(entity_id=entity, n=5)
    got_ids = [r.item_id for r in recs]
    got_scores = [round(float(r.score), 4) for r in recs]
    want_ids = [i for i, _ in GOLDEN[entity]]
    want_scores = [s for _, s in GOLDEN[entity]]
    assert got_ids == want_ids, "top-K item ids drifted — scoring changed"
    for g, w in zip(got_scores, want_scores):
        assert abs(g - w) < 1e-3, f"score drifted: {g} vs {w}"
