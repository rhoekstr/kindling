"""Loader tests (plan Phase 7).

Covers:
- Synthetic generators produce valid DatasetSplits that the Engine can
  fit without errors.
- Synthetic grocery has path signals that exercise the basket mechanism.
- Synthetic ratings has no session structure (so path signals stay low).
- Real loaders (Instacart, Amazon, RetailRocket) raise informative
  errors when data is missing, with the error paths going through the
  expected *Error classes.
"""

from __future__ import annotations

import pytest

from kindling import Engine
from kindling.loaders import amazon, instacart, retailrocket, synthetic


def test_grocery_split_valid() -> None:
    split = synthetic.make_grocery(n_entities=30, n_sessions_per_entity=4, seed=1)
    assert split.name == "synthetic-grocery"
    assert len(split.train) > 0
    assert len(split.test) > 0
    assert "entity_id" in split.train.columns
    assert "item_id" in split.train.columns
    assert "session_id" in split.train.columns
    assert split.items is not None
    assert "category" in split.items.columns


def test_ratings_split_valid() -> None:
    split = synthetic.make_ratings(n_entities=20, n_items=40, seed=2)
    assert split.name == "synthetic-ratings"
    assert len(split.train) > 0
    assert "timestamp" in split.train.columns


def test_engine_fits_on_synthetic_grocery() -> None:
    split = synthetic.make_grocery(n_entities=50, seed=0)
    engine = Engine(vi_max_iter=20).fit(split.train)
    recs = engine.recommend(entity_id=0, n=5)
    assert len(recs) <= 5


def test_engine_fits_on_synthetic_ratings() -> None:
    split = synthetic.make_ratings(n_entities=40, n_items=60, seed=0)
    engine = Engine(vi_max_iter=20).fit(split.train)
    # Sample entity id is an int 0..n_entities-1.
    recs = engine.recommend(entity_id=0, n=5)
    assert isinstance(recs, list)


def test_instacart_missing_data_raises() -> None:
    with pytest.raises(instacart.InstacartDataNotAvailableError, match="Missing"):
        instacart.load("/no/such/path/instacart")


def test_amazon_missing_data_raises() -> None:
    with pytest.raises(amazon.AmazonReviewsDataNotAvailableError, match="not found"):
        amazon.load("/no/such/path/Electronics_5.json.gz")


def test_retailrocket_missing_data_raises() -> None:
    with pytest.raises(retailrocket.RetailRocketDataNotAvailableError, match="not found"):
        retailrocket.load("/no/such/path/retailrocket")


def test_synthetic_grocery_has_baskets() -> None:
    """The grocery dataset's session_id column should partition into
    multi-item sessions (basket structure)."""
    split = synthetic.make_grocery(n_entities=20, items_per_session=4, seed=0)
    session_sizes = split.train.groupby("session_id").size()
    # Most sessions should have >= 2 items (else basket signal is silent).
    assert (session_sizes >= 2).mean() > 0.8


def test_synthetic_grocery_vs_ratings_different_structure() -> None:
    """Grocery should have sessions; ratings should have each entity as
    one session (their interactions aren't grouped)."""
    grocery = synthetic.make_grocery(n_entities=20, seed=0)
    ratings = synthetic.make_ratings(n_entities=20, seed=0)
    assert "session_id" in grocery.train.columns
    # The ratings split shouldn't carry session_id (no structure).
    assert "session_id" not in ratings.train.columns
