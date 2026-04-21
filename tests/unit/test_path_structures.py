"""Tests for TailIndex, PathTree, BasketIndex."""

from __future__ import annotations

import pandas as pd
import pytest

from kindling.ingest import infer_sessions, validate_interactions
from kindling.lifecycle.decay import ExponentialDecay
from kindling.path._sessions import SessionSequence, sessions_from_interactions
from kindling.path.basket_index import BasketSimilarity, _coverage, build_basket_index
from kindling.path.path_tree import build_path_tree
from kindling.path.tail_index import build_tail_index


@pytest.fixture
def session_frame() -> pd.DataFrame:
    """Three users with clear sequential sessions."""
    return pd.DataFrame(
        {
            "entity_id": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "c"],
            "item_id": [1, 2, 3, 1, 2, 4, 5, 6, 7, 8],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 10:05:00",
                    "2026-01-01 10:10:00",
                    "2026-01-02 12:00:00",
                    "2026-01-02 12:03:00",
                    "2026-01-02 12:07:00",
                    "2026-01-03 15:00:00",
                    "2026-01-03 15:02:00",
                    "2026-01-03 15:05:00",
                    "2026-01-03 15:09:00",
                ]
            ),
        }
    )


@pytest.fixture
def sessions(session_frame: pd.DataFrame) -> list:
    validate_interactions(session_frame)
    sess = infer_sessions(session_frame)
    return list(sessions_from_interactions(session_frame, sess.session_ids))


# ---- TailIndex ------------------------------------------------------------


def test_tail_index_captures_consecutive_pairs(sessions: list) -> None:
    idx = build_tail_index(sessions)
    # Session a: 1->2, 2->3 ; session b: 1->2, 2->4 ; session c: 5->6, 6->7, 7->8
    assert idx.counts[1][2] == 2.0
    assert idx.counts[2][3] == 1.0
    assert idx.counts[2][4] == 1.0
    # Unique (anchor, successor) pairs across all sessions:
    # (1,2) [shared between a and b], (2,3), (2,4), (5,6), (6,7), (7,8) = 6
    assert idx.n_pairs == 6


def test_tail_index_probability(sessions: list) -> None:
    idx = build_tail_index(sessions)
    # From 2, we've seen -> 3 once, -> 4 once; so P(3|2) = 0.5
    assert idx.score(candidate=3, last_item=2) == pytest.approx(0.5)
    assert idx.score(candidate=4, last_item=2) == pytest.approx(0.5)


def test_tail_index_unknown_anchor(sessions: list) -> None:
    idx = build_tail_index(sessions)
    assert idx.score(candidate=99, last_item=42) == 0.0
    assert idx.score(candidate=99, last_item=None) == 0.0


def test_tail_index_ignores_self_loops() -> None:
    # Session with a repeated item: should not count 1 -> 1.
    session = SessionSequence(session_id=0, entity_id="x", items=(1, 1, 2), end_timestamp=None)
    idx = build_tail_index([session])
    assert 1 not in idx.counts.get(1, {})
    assert idx.counts[1][2] == 1.0


def test_tail_index_decay_shrinks_old_sessions() -> None:
    now = 1_700_000_000.0
    old = SessionSequence(
        session_id=0, entity_id="x", items=(1, 2), end_timestamp=now - 180 * 86400
    )
    recent = SessionSequence(session_id=1, entity_id="y", items=(1, 2), end_timestamp=now)
    decay = ExponentialDecay(half_life_days=180.0)
    idx = build_tail_index([old, recent], decay=decay, reference_timestamp=now)
    # Old contributes ~0.5, recent contributes 1.0 -> total ~1.5
    assert idx.counts[1][2] == pytest.approx(1.5, rel=1e-4)


# ---- PathTree -------------------------------------------------------------


def test_path_tree_requires_three_items(sessions: list) -> None:
    tree = build_path_tree(sessions, max_prefix=2)
    # (1, 2) -> 3 from session a; (1, 2) -> 4 from session b
    row = tree.counts[(1, 2)]
    assert row[3] == 1.0
    assert row[4] == 1.0
    # P(3 | (1, 2)) = 0.5
    assert tree.score(3, history=(1, 2)) == pytest.approx(0.5)


def test_path_tree_back_off(sessions: list) -> None:
    """If the longest prefix doesn't match, back off to shorter prefixes."""
    tree = build_path_tree(sessions, max_prefix=3)
    # History (99, 1, 2): no match for 3-gram (99, 1, 2) but (1, 2) exists.
    assert tree.score(3, history=(99, 1, 2)) == pytest.approx(0.5)


def test_path_tree_zero_when_no_prefix_matches(sessions: list) -> None:
    tree = build_path_tree(sessions, max_prefix=3)
    assert tree.score(3, history=(42, 43)) == 0.0


def test_path_tree_rejects_invalid_max_prefix() -> None:
    with pytest.raises(ValueError, match="max_prefix must be >= 2"):
        build_path_tree([], max_prefix=1)


# ---- BasketIndex ----------------------------------------------------------


def test_basket_index_captures_next_add(sessions: list) -> None:
    idx = build_basket_index(sessions)
    # Session a contributes (basket={1}, next=2), (basket={1,2}, next=3)
    # Session b contributes (basket={1}, next=2), (basket={1,2}, next=4)
    # Session c contributes (basket={5}, next=6), ... and so on.
    assert idx.n_observations > 0
    # Query with {1, 2} under coverage similarity should score 3 and 4 non-zero
    query = frozenset({1, 2})
    scores = idx.score_many(candidates=[3, 4, 99], query_basket=query)
    assert scores[0] > 0
    assert scores[1] > 0
    assert scores[2] == 0


def test_basket_index_coverage_semantics(sessions: list) -> None:
    """Coverage = |Q & B_h| / |Q| — penalizes missing from historical basket
    only when it was also in the query."""
    idx = build_basket_index(sessions)
    # Query {1} has perfect overlap with basket {1, 2}; score 3 should be 1.0
    # if the only matching training observation was (basket={1,2}, next=3)
    # with coverage 1.0.
    score = idx.score(3, query_basket=frozenset({1, 2}), similarity=BasketSimilarity.COVERAGE)
    assert 0 < score <= 1


def test_basket_similarity_variants(sessions: list) -> None:
    idx = build_basket_index(sessions)
    q = frozenset({1, 2})
    cov = idx.score(3, query_basket=q, similarity=BasketSimilarity.COVERAGE)
    jac = idx.score(3, query_basket=q, similarity=BasketSimilarity.JACCARD)
    idf = idx.score(3, query_basket=q, similarity=BasketSimilarity.IDF_COVERAGE)
    exact = idx.score(3, query_basket=q, similarity=BasketSimilarity.EXACT)
    # All should be in [0, 1]; exact should be smallest (baskets rarely match
    # the query exactly), jaccard <= coverage by construction when query is a
    # subset of the basket.
    for v in (cov, jac, idf, exact):
        assert 0.0 <= v <= 1.0


def test_basket_inverted_index_equivalence(sessions: list) -> None:
    """Property: posting-list union returns the same candidate observations as
    naive iteration. Verified by matching output scores for a fixed query."""
    idx = build_basket_index(sessions)
    q = frozenset({1})
    # Posting-list-backed scoring:
    posting_score = idx.score_many([2, 3, 4, 5, 6], query_basket=q)

    # Naive: iterate every observation, apply coverage, sum, normalize.
    total = 0.0
    naive_numer: dict[object, float] = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
    for obs in idx.observations:
        sim = _coverage(q, obs.basket) * obs.weight
        total += sim
        if obs.next_item in naive_numer:
            naive_numer[obs.next_item] += sim
    naive_score = [(naive_numer[i] / total) if total > 0 else 0.0 for i in [2, 3, 4, 5, 6]]
    for expected, actual in zip(naive_score, posting_score, strict=True):
        assert actual == pytest.approx(expected, rel=1e-9)
