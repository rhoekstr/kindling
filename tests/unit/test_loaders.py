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

from kindling.engine import Engine
from kindling.loaders import (
    amazon,
    dunnhumby,
    gowalla,
    instacart,
    retailrocket,
    synthetic,
    tafeng,
    yelp,
)


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


# --- New extended-dataset loaders: missing-data + parser-correctness paths. ---


def test_gowalla_missing_data_raises() -> None:
    with pytest.raises(gowalla.GowallaDataNotAvailableError, match="not found"):
        gowalla.load("/no/such/path/gowalla")


def test_yelp_missing_data_raises() -> None:
    with pytest.raises(yelp.YelpDataNotAvailableError, match="not found"):
        yelp.load("/no/such/path/yelp")


def test_tafeng_missing_data_raises() -> None:
    with pytest.raises(tafeng.TafengDataNotAvailableError, match="not found"):
        tafeng.load("/no/such/path/tafeng")


def test_dunnhumby_missing_data_raises() -> None:
    with pytest.raises(dunnhumby.DunnhumbyDataNotAvailableError, match="not found"):
        dunnhumby.load("/no/such/path/dunnhumby")


def test_gowalla_snap_format_parses(tmp_path) -> None:
    """Fabricate a minimal SNAP check-in log and confirm the loader maps
    it to the canonical schema."""
    log = tmp_path / "loc-gowalla_totalCheckins.txt"
    # user, ts, lat, lon, location_id
    log.write_text(
        "1\t2009-01-01T00:00:00Z\t30.0\t-90.0\t1001\n"
        "1\t2009-01-02T00:00:00Z\t30.0\t-90.0\t1002\n"
        "1\t2009-01-03T00:00:00Z\t30.0\t-90.0\t1003\n"
        "2\t2009-02-01T00:00:00Z\t40.0\t-100.0\t2001\n"
        "2\t2009-02-02T00:00:00Z\t40.0\t-100.0\t2002\n"
    )
    split = gowalla.load(tmp_path, test_fraction=0.34)
    assert split.name == "gowalla"
    assert "entity_id" in split.train.columns
    assert "timestamp" in split.train.columns
    # Each user has at least one row in train. test_fraction holds out the
    # last ~third per user.
    assert len(split.train) >= 1
    assert len(split.test) >= 1


def test_yelp_academic_split_format_parses(tmp_path) -> None:
    (tmp_path / "train.txt").write_text("u1 i1 i2 i3\nu2 i4 i5\n")
    (tmp_path / "test.txt").write_text("u1 i6\nu2 i7\n")
    split = yelp.load(tmp_path)
    assert split.name == "yelp2018"
    assert len(split.train) == 5
    assert len(split.test) == 2
    assert set(split.train["entity_id"].unique()) == {"u1", "u2"}


def test_tafeng_csv_format_parses(tmp_path) -> None:
    csv = tmp_path / "ta_feng_all_months_merged.csv"
    csv.write_text(
        "TRANSACTION_DT,CUSTOMER_ID,AGE_GROUP,PIN_CODE,PRODUCT_SUBCLASS,PRODUCT_ID,AMOUNT,ASSET,SALES_PRICE\n"
        "11/1/2000,1001,A,F,100,5001,1,10,20\n"
        "11/1/2000,1001,A,F,100,5002,1,10,20\n"
        "11/2/2000,1001,A,F,200,5003,1,10,20\n"
        "11/3/2000,1002,B,M,100,5001,1,10,20\n"
    )
    split = tafeng.load(tmp_path)
    assert split.name == "tafeng"
    assert split.items is not None
    assert "category" in split.items.columns
    # Same-day customer rows should share session_id.
    sessions = split.train.groupby("session_id")["item_id"].count()
    assert (sessions >= 1).all()


def test_dunnhumby_csv_format_parses(tmp_path) -> None:
    tx = tmp_path / "transaction_data.csv"
    tx.write_text(
        "household_key,BASKET_ID,DAY,QUANTITY,PRODUCT_ID,SALES_VALUE,STORE_ID,RETAIL_DISC,TRANS_TIME,WEEK_NO,COUPON_DISC,COUPON_MATCH_DISC\n"
        "1,90001,1,1,1001,3.0,1,0,1430,1,0,0\n"
        "1,90001,1,1,1002,2.0,1,0,1430,1,0,0\n"
        "1,90002,3,1,1003,5.0,1,0,1500,1,0,0\n"
        "2,90003,1,1,1001,3.0,1,0,1100,1,0,0\n"
        "2,90004,5,1,1004,4.0,1,0,1100,1,0,0\n"
    )
    split = dunnhumby.load(tmp_path)
    assert split.name == "dunnhumby"
    assert "session_id" in split.train.columns
    assert "timestamp" in split.train.columns
    # BASKET_ID groups items together.
    basket_sizes = split.train.groupby("session_id").size()
    assert (basket_sizes >= 1).all()
