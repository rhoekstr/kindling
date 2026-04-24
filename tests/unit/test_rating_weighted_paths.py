"""Rating-weighted path signals.

When the interactions carry a rating column, the path builders
(path_tree, tail_index, basket_index) weight each observation's
count increment by the destination item's rating weight. So a
user who went a->b with b rated 5 stars contributes more to
path/tail/basket counts than the same a->b with b rated 1 star.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kindling.path._sessions import SessionSequence, sessions_from_interactions
from kindling.path.basket_index import build_basket_index
from kindling.path.path_tree import build_path_tree
from kindling.path.tail_index import build_tail_index
from kindling.preprocess import preprocess_interactions


def _tiny_sessions_with_ratings() -> pd.DataFrame:
    """Two users with identical a->b sequence but different b ratings.
    User 1 rates b as 5 stars; user 2 rates b as 1 star.
    Rating-weighted tail: b should get weight ~1.0 from user1 + 0.0 from user2
    (vs binary: 1.0 + 1.0 = 2.0)."""
    base = pd.Timestamp("2026-01-01")
    rows = [
        {"entity_id": 1, "item_id": "a", "timestamp": base + pd.Timedelta(minutes=0), "rating": 5},
        {"entity_id": 1, "item_id": "b", "timestamp": base + pd.Timedelta(minutes=5), "rating": 5},
        {"entity_id": 2, "item_id": "a", "timestamp": base + pd.Timedelta(minutes=0), "rating": 5},
        {"entity_id": 2, "item_id": "b", "timestamp": base + pd.Timedelta(minutes=5), "rating": 1},
    ]
    return pd.DataFrame(rows)


def test_tail_index_weighted_by_destination_rating() -> None:
    df = _tiny_sessions_with_ratings()
    processed, _ = preprocess_interactions(df)
    # One session per user (no real session inference needed for the test).
    sessions = list(sessions_from_interactions(processed, session_ids=np.array([0, 0, 1, 1])))
    idx = build_tail_index(sessions)
    # User 1 contributes weight 1.0 to (a->b); user 2 contributes 0.0.
    # Total: a->b count = 1.0 (not 2.0 as in binary).
    assert idx.counts["a"]["b"] < 1.5  # definitely not binary 2.0
    assert idx.counts["a"]["b"] > 0.5  # at least one user contributed


def test_tail_index_binary_behavior_preserved_without_rating() -> None:
    df = _tiny_sessions_with_ratings().drop(columns=["rating"])
    processed, _ = preprocess_interactions(df)
    sessions = list(sessions_from_interactions(processed, session_ids=np.array([0, 0, 1, 1])))
    idx = build_tail_index(sessions)
    # Both users contribute 1.0 each: a->b count = 2.0.
    assert idx.counts["a"]["b"] == 2.0


def test_basket_index_skips_zero_rating_observations() -> None:
    df = _tiny_sessions_with_ratings()
    processed, _ = preprocess_interactions(df)
    sessions = list(sessions_from_interactions(processed, session_ids=np.array([0, 0, 1, 1])))
    bidx = build_basket_index(sessions)
    # Only user 1's b-observation has positive weight. User 2's b
    # (rating 1, weight 0) is skipped entirely. So we expect 1 obs,
    # not 2.
    assert len(bidx.observations) == 1
    assert bidx.observations[0].next_item == "b"


def test_path_tree_weighted_by_destination() -> None:
    """3-item sessions, user 1 rates c=5 and user 2 rates c=1.
    Path a->b->c should accumulate only user 1's contribution."""
    base = pd.Timestamp("2026-01-01")
    rows = []
    for user, ratings in [(1, {"a": 5, "b": 5, "c": 5}), (2, {"a": 5, "b": 5, "c": 1})]:
        for i, item in enumerate(["a", "b", "c"]):
            rows.append({
                "entity_id": user,
                "item_id": item,
                "timestamp": base + pd.Timedelta(minutes=i),
                "rating": ratings[item],
            })
    df = pd.DataFrame(rows)
    processed, _ = preprocess_interactions(df)
    sessions = list(sessions_from_interactions(processed, session_ids=np.array([0, 0, 0, 1, 1, 1])))
    tree = build_path_tree(sessions, max_prefix=2)
    # Prefix (a, b) -> c: user 1 contributes 1.0, user 2 contributes 0.0
    # (rating 1 -> weight 0).
    ab_c = tree.counts[("a", "b")]["c"]
    assert ab_c == 1.0  # rating-weighted: exactly user 1


def test_path_tree_binary_without_rating() -> None:
    base = pd.Timestamp("2026-01-01")
    rows = []
    for user in [1, 2]:
        for i, item in enumerate(["a", "b", "c"]):
            rows.append({
                "entity_id": user,
                "item_id": item,
                "timestamp": base + pd.Timedelta(minutes=i),
            })
    df = pd.DataFrame(rows)
    processed, _ = preprocess_interactions(df)
    sessions = list(sessions_from_interactions(processed, session_ids=np.array([0, 0, 0, 1, 1, 1])))
    tree = build_path_tree(sessions, max_prefix=2)
    # Without ratings: both users contribute 1.0, count = 2.0.
    assert tree.counts[("a", "b")]["c"] == 2.0
