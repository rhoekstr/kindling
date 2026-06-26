"""Native engine hardening — adversarial inputs must not panic.

The native ``EngineState`` is a PyO3 object callable directly with arbitrary
indices, so ``recommend`` must defend against out-of-range / negative owned
items, degenerate ``n``, and malformed construction. A Rust panic surfaces as a
``pyo3_runtime.PanicException``; these tests assert clean behavior instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from kindling import Engine
from kindling._native import CORE_AVAILABLE, kindling_core
from kindling._native_engine import build_native_engine
from kindling.loaders import synthetic

pytestmark = pytest.mark.skipif(not CORE_AVAILABLE, reason="native extension not built")


@pytest.fixture(scope="module")
def native():
    s = synthetic.make_ratings(n_entities=120, n_items=80, ratings_per_entity=25, seed=0)
    eng = Engine(random_state=0).fit(s.train)
    ne = build_native_engine(eng)
    assert ne is not None
    return ne, eng._state.n_items


def _valid(items, n_items):
    return all(0 <= int(i) < n_items for i in items)


def test_out_of_range_owned_does_not_panic(native):
    ne, n_items = native
    for owned in ([n_items + 999], [n_items, n_items * 5], [-1, -42], [10**9]):
        items, scores, kinds = ne.recommend(owned, -1, 10, 0.0)
        assert _valid(items, n_items)
        assert len(items) == len(scores) == len(kinds)


def test_empty_and_degenerate_n(native):
    ne, n_items = native
    assert ne.recommend([], -1, 10, 0.0)[0] == []          # no owned
    assert ne.recommend([1, 2, 3], -1, 0, 0.0)[0] == []     # n = 0
    big = ne.recommend([1, 2, 3], -1, 10**6, 0.0)[0]        # n >> catalog
    assert _valid(big, n_items) and len(big) <= n_items


def test_duplicate_and_mixed_owned(native):
    ne, n_items = native
    items, _s, _k = ne.recommend([3, 3, 3, n_items + 1, -5, 7], -1, 10, 0.0)
    assert _valid(items, n_items)
    # owned items are never recommended back
    assert 3 not in items and 7 not in items


def test_out_of_range_user_row(native):
    ne, n_items = native
    # huge / negative user_row must not index out of the user table
    for ur in (-1, 10**9, -10**9):
        items, _s, _k = ne.recommend([1, 2, 5], ur, 10, 0.0)
        assert _valid(items, n_items)


def test_build_engine_rejects_bad_shapes():
    with pytest.raises(ValueError, match="n_items"):
        kindling_core.build_engine({"ease_b": np.zeros((4, 4), np.float32)}, {"n_items": 0})
    with pytest.raises(ValueError, match="inconsistent"):
        # ease dim 4 > n_items 2
        kindling_core.build_engine(
            {"ease_b": np.zeros((4, 4), np.float32)}, {"n_items": 2, "base_is_ease": True}
        )
    with pytest.raises(ValueError, match="base scorer"):
        # no base scorer at all
        kindling_core.build_engine({}, {"n_items": 10})
