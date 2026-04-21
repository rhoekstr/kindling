"""End-to-end invariants that must hold for the Engine output.

- Retrieval dedup: union-max-score is commutative and associative — tested
  here by checking that the Engine's recommendations are stable regardless
  of interaction row order.
- Recommendations never include owned items.
- Recommendation scores are non-increasing.
- Constraint filter: filtered items never appear.
"""

from __future__ import annotations

import random

import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kindling import Engine

rows_strategy = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=5),
        st.integers(min_value=0, max_value=9),
    ),
    min_size=6,
    max_size=60,
).filter(lambda rows: len({e for e, _ in rows}) >= 2 and len({i for _, i in rows}) >= 3)


def _df(rows: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": [f"e{e}" for e, _ in rows],
            "item_id": [f"i{i}" for _, i in rows],
        }
    )


@given(rows=rows_strategy, seed=st.integers(min_value=0, max_value=1000))
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_row_order_doesnt_affect_recommendations(
    rows: list[tuple[int, int]],
    seed: int,
) -> None:
    """Row order must not change the recommendation list for a given entity
    (co-occurrence is a set-based operation)."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    engine_a = Engine().fit(_df(rows))
    engine_b = Engine().fit(_df(shuffled))

    entity = "e0"
    recs_a = [r.item_id for r in engine_a.recommend(entity_id=entity, n=10)]
    recs_b = [r.item_id for r in engine_b.recommend(entity_id=entity, n=10)]
    assert recs_a == recs_b


@given(rows=rows_strategy)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_never_recommends_owned_items(rows: list[tuple[int, int]]) -> None:
    df = _df(rows)
    engine = Engine().fit(df)
    owned_by_entity: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        owned_by_entity.setdefault(row["entity_id"], set()).add(row["item_id"])
    for entity, owned in owned_by_entity.items():
        for rec in engine.recommend(entity_id=entity, n=10):
            assert rec.item_id not in owned


@given(rows=rows_strategy)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_scores_non_increasing(rows: list[tuple[int, int]]) -> None:
    df = _df(rows)
    engine = Engine().fit(df)
    for entity in df["entity_id"].unique():
        recs = engine.recommend(entity_id=entity, n=10)
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)


@given(rows=rows_strategy)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_constraint_filter_is_honored(rows: list[tuple[int, int]]) -> None:
    df = _df(rows)
    engine = Engine().fit(df)
    forbidden = "i0"
    for entity in df["entity_id"].unique():
        recs = engine.recommend(
            entity_id=entity,
            n=10,
            constraints=[lambda item: item != forbidden],
        )
        assert all(r.item_id != forbidden for r in recs)
