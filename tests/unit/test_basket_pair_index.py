"""Pair-index path for BasketIndex (post Phase-10 latency work).

The pair index trades a slight semantic (drops basket observations whose
overlap with the query is a single item) for a large latency win on
session-rich data. These tests lock in the parity + the stated tradeoff.
"""

from __future__ import annotations

import numpy as np

from kindling.path._sessions import SessionSequence
from kindling.path.basket_index import BasketSimilarity, build_basket_index


def _sessions(n: int = 200, n_items: int = 50, seed: int = 0) -> list[SessionSequence]:
    rng = np.random.default_rng(seed)
    out: list[SessionSequence] = []
    for i in range(n):
        size = int(rng.integers(4, 12))
        items = tuple(rng.choice(n_items, size=size, replace=False).tolist())
        out.append(SessionSequence(session_id=i, entity_id=i, items=items, end_timestamp=None))
    return out


def test_pair_index_built_when_ints() -> None:
    idx = build_basket_index(_sessions(), build_pair_index=True)
    assert idx.pair_postings, "pair_postings should be populated for int items"
    # Every pair posting list must reference a valid observation index.
    n_obs = len(idx.observations)
    for posting in idx.pair_postings.values():
        for obs_idx in posting:
            assert 0 <= obs_idx < n_obs


def test_pair_index_skipped_when_non_int() -> None:
    sessions = [
        SessionSequence(0, "a", ("bread", "butter", "jam"), None),
        SessionSequence(1, "b", ("bread", "milk"), None),
    ]
    idx = build_basket_index(sessions, build_pair_index=True)
    assert idx.pair_postings == {}
    # Item postings still work.
    assert idx.postings


def test_pair_index_scores_match_item_index_on_large_query() -> None:
    """On realistic query sizes (>=10 items), the pair index must score
    within a tight tolerance of the item index - the only observations
    it drops are single-overlap ones, which contribute <= 1/|Q| weight."""
    sessions = _sessions(n=300)
    idx_pair = build_basket_index(sessions, build_pair_index=True)
    idx_item = build_basket_index(sessions, build_pair_index=False)
    candidates = list(range(50))
    rng = np.random.default_rng(42)
    for _ in range(5):
        q = frozenset(rng.choice(50, size=15, replace=False).tolist())
        s_pair = idx_pair.score_many(candidates, q, BasketSimilarity.COVERAGE)
        s_item = idx_item.score_many(candidates, q, BasketSimilarity.COVERAGE)
        assert s_pair.std() > 0 and s_item.std() > 0
        corr = float(np.corrcoef(s_pair, s_item)[0, 1])
        assert corr >= 0.85, f"corr {corr:.3f} below threshold"


def test_pair_index_fallback_when_query_is_single_item() -> None:
    """|Q| < 2 falls back to item postings so the single-item signal
    still contributes."""
    idx = build_basket_index(_sessions(), build_pair_index=True)
    candidates = list(range(50))
    q = frozenset({3})
    scores = idx.score_many(candidates, q, BasketSimilarity.COVERAGE)
    # At least some candidates should score >0 because item posting for
    # item 3 is non-empty with high probability on this fixture.
    assert scores.sum() > 0


def test_distinctiveness_weighting_beats_popularity() -> None:
    """When a globally-popular item (milk) and a basket-specific item
    (refried beans) both appear after the query basket, distinctiveness
    weighting must pick the specific item; the raw signal picks the
    popular one because its global mass dominates."""
    import random

    rng = random.Random(0)
    sessions: list[SessionSequence] = []
    sid = 0
    # 100 Mexican baskets: 80% next-add is refried beans (5), 20% milk (99).
    for _ in range(100):
        nxt = 5 if rng.random() < 0.8 else 99
        sessions.append(SessionSequence(sid, sid, (0, 1, 2, nxt), None))
        sid += 1
    # 900 generic sessions, all next-add = milk. Sometimes share item 0 or 1
    # with the query so they show up in the overlap.
    for _ in range(900):
        b0 = 0 if rng.random() < 0.3 else 1
        sessions.append(SessionSequence(sid, sid, (b0, 10, 20, 30, 99), None))
        sid += 1

    candidates = [5, 99]
    q = frozenset({0, 1, 2})

    idx_raw = build_basket_index(sessions, build_pair_index=False, distinctiveness_weighting=False)
    s_raw = idx_raw.score_many(candidates, q, BasketSimilarity.COVERAGE)
    assert s_raw[1] > s_raw[0], "raw signal should be popularity-biased (milk wins)"

    idx_d = build_basket_index(sessions, build_pair_index=False, distinctiveness_weighting=True)
    s_d = idx_d.score_many(candidates, q, BasketSimilarity.COVERAGE)
    assert s_d[0] > s_d[1], "distinctiveness signal should surface the specific item (refried)"


def test_prune_below_rebuilds_pair_postings() -> None:
    idx = build_basket_index(_sessions(), build_pair_index=True)
    n_pairs_before = len(idx.pair_postings)
    # Find the median observation weight and prune at 1.5x it.
    weights = sorted(obs.weight for obs in idx.observations)
    threshold = weights[len(weights) // 2] * 1.5 if weights else 0.0
    pruned_count, _ = idx.prune_below(threshold)
    if pruned_count == 0:
        return
    n_obs_after = len(idx.observations)
    # Every pair posting list must still reference a valid (smaller) obs range.
    for posting in idx.pair_postings.values():
        for obs_idx in posting:
            assert 0 <= obs_idx < n_obs_after
    # Pair-posting count is non-increasing.
    assert len(idx.pair_postings) <= n_pairs_before
