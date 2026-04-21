"""Differential tests: Rust extension output must match pure-Python.

Skip the test module entirely when ``kindling_native`` isn't built.
The Engine is expected to fall back gracefully; we verify it here by
comparing the in-process Rust output against the pure-Python path for
every kernel we ported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from kindling._native import NATIVE_AVAILABLE, kindling_native
from kindling.engine import _cooccurrence_signal
from kindling.graph.item_graph import build_item_graph
from kindling.path._sessions import SessionSequence
from kindling.path.basket_index import BasketSimilarity, build_basket_index

pytestmark = pytest.mark.skipif(
    not NATIVE_AVAILABLE, reason="kindling_native not built in this environment"
)


def test_cooccurrence_signal_matches_python() -> None:
    rng = np.random.default_rng(42)
    interactions = pd.DataFrame(
        {
            "entity_id": rng.integers(0, 50, size=500),
            "item_id": rng.integers(0, 100, size=500),
        }
    )
    graph = build_item_graph(interactions)

    owned = np.array(list(graph.item_index.keys())[:20])
    cands = list(graph.item_index.keys())[20:60]

    # Python fallback path (bypass the native branch).
    owned_indices = [graph.item_index[i] for i in owned]
    summed = np.asarray(graph.adjacency[owned_indices].sum(axis=0)).ravel()
    py = np.array(
        [float(summed[graph.item_index[c]]) for c in cands], dtype=np.float64
    )

    # Rust path.
    rust = _cooccurrence_signal(cands, owned, graph)
    np.testing.assert_allclose(rust, py, rtol=1e-6, atol=1e-9)


def test_tail_score_many_matches_python() -> None:
    row = [(1, 3.0), (2, 1.5), (3, 0.5), (4, 2.0)]
    row_total = 7.0
    candidates = [1, 2, 5, 3, 4]

    # Python reference.
    py = np.array(
        [
            (3.0 / 7.0),
            (1.5 / 7.0),
            0.0,
            (0.5 / 7.0),
            (2.0 / 7.0),
        ],
        dtype=np.float64,
    )
    rust = np.asarray(
        kindling_native.tail_score_many(row, row_total, candidates)
    )
    np.testing.assert_allclose(rust, py, rtol=1e-9)


def test_dpp_cosine_matches_python_reference() -> None:
    """Verify the native cosine kernel matches numpy-computed reference
    on a small co-occurrence matrix."""
    rng = np.random.default_rng(0)
    n_items = 30
    n_features = 20
    # Build a small "CSR row per item" by sparsifying a dense matrix.
    dense = rng.normal(size=(n_items, n_features)).astype(np.float32)
    dense[np.abs(dense) < 0.5] = 0.0  # introduce sparsity
    csr = sparse.csr_matrix(dense)

    # Expected similarity: np cosine.
    norms = np.linalg.norm(dense, axis=1)
    expected = np.zeros((n_items, n_items), dtype=np.float64)
    for i in range(n_items):
        for j in range(n_items):
            if i == j:
                expected[i, j] = 1.0
                continue
            if norms[i] == 0 or norms[j] == 0:
                continue
            expected[i, j] = float(np.dot(dense[i], dense[j]) / (norms[i] * norms[j]))

    row_ptr = csr.indptr.astype(np.int64)
    row_ind = csr.indices.astype(np.int32)
    row_dat = csr.data.astype(np.float32)
    rust = np.asarray(
        kindling_native.cosine_similarity_matrix(row_ptr, row_ind, row_dat)
    )
    np.testing.assert_allclose(rust, expected, rtol=1e-5, atol=1e-6)


def test_dedup_max_score_matches_python() -> None:
    item_ids = [10, 20, 10, 30, 20, 40]
    scores = [1.0, 2.0, 3.0, 0.5, 1.5, 4.0]

    # Expected: winners [20 -> 2.0, 10 -> 3.0, 30 -> 0.5, 40 -> 4.0]
    # sorted desc: 40 (4.0), 10 (3.0), 20 (2.0), 30 (0.5)
    rust = kindling_native.dedup_max_score(item_ids, scores, 10)
    assert [item_ids[i] for i in rust] == [40, 10, 20, 30]


def test_basket_score_many_matches_python() -> None:
    """Compare Rust basket_score_many against the Python BasketIndex on
    the same set of observations."""
    sessions = [
        SessionSequence(0, "a", (1, 2, 3, 4), None),
        SessionSequence(1, "b", (1, 2, 5, 6), None),
        SessionSequence(2, "c", (3, 4, 5), None),
    ]
    idx = build_basket_index(sessions)
    query = frozenset({1, 2})
    candidates = [3, 4, 5, 6, 7]
    py_scores = idx.score_many(
        candidates, query_basket=query, similarity=BasketSimilarity.COVERAGE
    )

    # Rust call: assemble the parallel arrays from the Python index.
    basket_items_flat: list[int] = []
    basket_starts: list[int] = []
    basket_lens: list[int] = []
    next_items: list[int] = []
    weights: list[float] = []
    for obs in idx.observations:
        basket_starts.append(len(basket_items_flat))
        basket_lens.append(len(obs.basket))
        basket_items_flat.extend(sorted(obs.basket))
        next_items.append(obs.next_item)
        weights.append(obs.weight)
    overlap_ids = sorted({i for item in query for i in idx.postings.get(item, [])})

    rust = np.asarray(
        kindling_native.basket_score_many(
            basket_starts,
            basket_lens,
            basket_items_flat,
            next_items,
            weights,
            overlap_ids,
            sorted(query),
            candidates,
        )
    )
    np.testing.assert_allclose(rust, py_scores, rtol=1e-6, atol=1e-9)
