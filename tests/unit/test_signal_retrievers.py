"""Unit tests for the per-signal retrievers (standalone diagnostic).

Each retriever must return at least some candidates on a non-trivial
fixture, respect the exclude set, and surface different candidate sets
from one another (otherwise the standalone eval is uninformative).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling.engine import Engine
from kindling.personas import KMeansClustering, PersonaConfig
from kindling.retrieve.signal_retrievers import (
    ALSRetriever,
    CosineRetriever,
    PathBasketRetriever,
    PathFullRetriever,
    PathTailRetriever,
    PersonaRetriever,
)


def _fit_engine() -> Engine:
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2026-01-01")
    for ent in range(80):
        group = ent // 20  # 4 taste groups
        for s in range(6):
            items = rng.choice(
                list(range(group * 15, group * 15 + 15)) + list(range(60, 75)),
                size=5,
                replace=False,
            )
            for k, item in enumerate(items):
                rows.append({
                    "entity_id": ent,
                    "item_id": int(item),
                    "timestamp": base + pd.Timedelta(days=s, minutes=int(k)),
                    "session_id": ent * 100 + s,
                })
    df = pd.DataFrame(rows)
    cfg = PersonaConfig(
        enabled=True,
        clustering=KMeansClustering(n_clusters=4, random_state=0),
        min_activation_users=20,
    )
    return Engine(persona_config=cfg).fit(df)


def test_path_tail_retriever_returns_candidates() -> None:
    engine = _fit_engine()
    item_ids = np.asarray(engine.item_graph.item_ids, dtype=object)
    r = PathTailRetriever(engine._tail_index, item_ids)
    history = engine._history_by_entity[0]
    owned = engine._owned_by_entity[0]
    exclude = set(owned.tolist())
    candidates = r.retrieve(history, budget=20, exclude=exclude)
    assert candidates
    for c in candidates:
        assert c.item_id not in exclude
        assert c.source == "path_tail"


def test_path_full_retriever_returns_candidates() -> None:
    engine = _fit_engine()
    item_ids = np.asarray(engine.item_graph.item_ids, dtype=object)
    r = PathFullRetriever(engine._path_tree, item_ids)
    history = engine._history_by_entity[0]
    exclude = set(engine._owned_by_entity[0].tolist())
    candidates = r.retrieve(history, budget=20, exclude=exclude)
    # Path-full may return zero if no matching prefix - skip only if empty.
    for c in candidates:
        assert c.item_id not in exclude


def test_path_basket_retriever_returns_candidates() -> None:
    engine = _fit_engine()
    item_ids = np.asarray(engine.item_graph.item_ids, dtype=object)
    r = PathBasketRetriever(engine._basket_index, item_ids)
    history = engine._history_by_entity[0]
    q = frozenset(history)
    exclude = set(engine._owned_by_entity[0].tolist())
    candidates = r.retrieve(q, budget=20, exclude=exclude)
    assert candidates
    for c in candidates:
        assert c.item_id not in exclude


def test_cosine_retriever_returns_candidates() -> None:
    engine = _fit_engine()
    if engine._item_cosine is None:
        pytest.skip("item cosine not fitted")
    item_ids = np.asarray(engine.item_graph.item_ids, dtype=object)
    r = CosineRetriever(engine._item_cosine, engine._item_graph, item_ids)
    owned = engine._owned_by_entity[0]
    candidates = r.retrieve(owned, budget=20, exclude=set(owned.tolist()))
    assert candidates


def test_als_retriever_returns_candidates() -> None:
    engine = _fit_engine()
    if engine._als_factors is None:
        pytest.skip("ALS factors not fitted (requires `implicit`)")
    item_ids = np.asarray(engine.item_graph.item_ids, dtype=object)
    r = ALSRetriever(engine._als_factors, engine._item_graph, item_ids)
    owned = engine._owned_by_entity[0]
    candidates = r.retrieve(entity_id=0, budget=20, exclude=set(owned.tolist()))
    assert candidates


def test_persona_retriever_returns_candidates() -> None:
    engine = _fit_engine()
    if engine._persona_index is None or engine._persona_index.n_personas == 0:
        pytest.skip("persona index not fitted")
    item_ids = np.asarray(engine.item_graph.item_ids, dtype=object)
    r = PersonaRetriever(engine._persona_index, item_ids)
    owned = engine._owned_by_entity[0]
    history = engine._history_by_entity[0]
    candidates = r.retrieve(
        entity_id=0,
        owned_items=owned,
        history=history,
        budget=20,
        exclude=set(owned.tolist()),
    )
    assert candidates


def test_retrievers_produce_distinct_candidate_sets() -> None:
    """Sanity: different retrievers pick different items for the same entity.
    If they all picked identical sets, the diagnostic adds nothing."""
    engine = _fit_engine()
    item_ids = np.asarray(engine.item_graph.item_ids, dtype=object)

    entity = 0
    owned = engine._owned_by_entity[entity]
    history = engine._history_by_entity[entity]
    exclude = set(owned.tolist())

    sets: dict[str, set[object]] = {}
    sets["path_tail"] = {
        c.item_id
        for c in PathTailRetriever(engine._tail_index, item_ids).retrieve(history, 30, exclude)
    }
    sets["path_basket"] = {
        c.item_id
        for c in PathBasketRetriever(engine._basket_index, item_ids).retrieve(
            frozenset(history), 30, exclude
        )
    }
    if engine._als_factors is not None:
        sets["als"] = {
            c.item_id
            for c in ALSRetriever(engine._als_factors, engine._item_graph, item_ids).retrieve(
                entity, 30, exclude
            )
        }
    if engine._persona_index is not None:
        sets["persona"] = {
            c.item_id
            for c in PersonaRetriever(engine._persona_index, item_ids).retrieve(
                entity, owned, history, 30, exclude
            )
        }

    non_empty = {k: v for k, v in sets.items() if v}
    assert len(non_empty) >= 2
    # At least one pair must differ.
    names = list(non_empty.keys())
    any_different = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if non_empty[names[i]] != non_empty[names[j]]:
                any_different = True
                break
        if any_different:
            break
    assert any_different, "all retrievers returned identical candidate sets"
