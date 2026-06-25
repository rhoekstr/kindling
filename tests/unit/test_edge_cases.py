"""Degenerate / malformed inputs degrade gracefully or fail clearly.

Locks in the engine's robustness so future changes can't silently regress
it: malformed input raises a clear contract error; degenerate-but-valid
input (single user/item, unknown entity, odd seeds, n=0) returns sensibly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling import Engine
from kindling.ingest.contract import InteractionContractError


def _engine():
    return Engine(random_state=0)


def _df():
    return pd.DataFrame({"entity_id": ["a", "a", "b", "b", "c"], "item_id": [1, 2, 1, 3, 2]})


def test_empty_interactions_raise_clear_error():
    with pytest.raises(InteractionContractError, match="empty"):
        _engine().fit(pd.DataFrame({"entity_id": [], "item_id": []}))


def test_null_item_raises_clear_error():
    with pytest.raises(InteractionContractError, match="null"):
        _engine().fit(pd.DataFrame({"entity_id": ["a", "b"], "item_id": [1, np.nan]}))


def test_single_user_and_single_item_fit():
    _engine().fit(pd.DataFrame({"entity_id": ["a", "a"], "item_id": [1, 2]}))
    _engine().fit(pd.DataFrame({"entity_id": ["a", "b"], "item_id": [1, 1]}))


def test_duplicate_interactions_fit():
    e = _engine()
    e.fit(pd.DataFrame({"entity_id": ["a", "a", "a"], "item_id": [1, 1, 1]}))
    assert e.recommend(entity_id="a", n=5) is not None


def test_recommend_unknown_entity_returns_list():
    e = _engine()
    e.fit(_df())
    recs = e.recommend(entity_id="does-not-exist", n=5)
    assert isinstance(recs, list)  # cold/popularity fallback, not a crash


def test_recommend_for_items_handles_unknown_and_duplicate_seeds():
    e = _engine()
    e.fit(_df())
    assert isinstance(e.recommend_for_items(seed_item_ids=[999, 888], n=5), list)
    assert isinstance(e.recommend_for_items(seed_item_ids=[1, 1, 1], n=5), list)
    assert e.recommend_for_items(seed_item_ids=[], n=5) is not None


def test_recommend_n_zero_returns_empty():
    e = _engine()
    e.fit(_df())
    assert e.recommend(entity_id="a", n=0) == []
