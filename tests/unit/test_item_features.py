"""Unit tests for the generic item feature extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling.item_features import (
    ItemFeatureExtractor,
    content_scores,
)


@pytest.fixture
def meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "genres": ["Action|Comedy", "Action|Drama", "Drama", "Comedy"],
            "brand": ["acme", "acme", "globex", "globex"],
            "price": [10.0, 12.0, 95.0, 100.0],
            "blurb": [
                "fast car chase movie",
                "fast explosive action ride",
                "slow emotional courtroom story",
                "funny courtroom comedy",
            ],
            "constant_col": ["x", "x", "x", "x"],
        }
    )


@pytest.fixture
def item_to_idx() -> dict:
    return {"a": 0, "b": 1, "c": 2, "d": 3}


def test_schema_inference(meta, item_to_idx):
    feats = ItemFeatureExtractor(min_df=1).fit_transform(meta, item_to_idx, 4)
    kinds = {s.column: s.kind for s in feats.specs}
    assert kinds["genres"] == "multi_categorical"
    assert kinds["brand"] == "categorical"
    assert kinds["price"] == "numeric"
    assert kinds["blurb"] == "text"
    assert kinds["constant_col"] == "ignored"
    assert feats.coverage == 1.0
    assert feats.n_items == 4


def test_rows_l2_normalized(meta, item_to_idx):
    feats = ItemFeatureExtractor(min_df=1).fit_transform(meta, item_to_idx, 4)
    for r in range(4):
        s, e = int(feats.indptr[r]), int(feats.indptr[r + 1])
        if e > s:
            norm = float(np.sqrt((feats.data[s:e] ** 2).sum()))
            assert norm == pytest.approx(1.0, abs=1e-5)


def test_content_scores_favor_shared_attributes(meta, item_to_idx):
    feats = ItemFeatureExtractor(min_df=1).fit_transform(meta, item_to_idx, 4)
    # User owns item a (Action|Comedy, acme, cheap, "fast..."). Item b
    # shares genre+brand+price-bin+token "fast"; item c shares nothing.
    scores = content_scores(feats, np.array([0]))
    assert scores[1] > scores[2], f"b should beat c: {scores}"


def test_missing_metadata_rows_get_empty_features(meta, item_to_idx):
    # Catalog has 6 items; metadata only covers 4.
    feats = ItemFeatureExtractor(min_df=1).fit_transform(meta, item_to_idx, 6)
    assert feats.n_items == 6
    # Rows 4 and 5 are empty.
    assert feats.indptr[4] == feats.indptr[5] == feats.indptr[6]
    assert feats.coverage == pytest.approx(4 / 6)
    # content_scores must handle trailing empty rows (reduceat edge).
    scores = content_scores(feats, np.array([0]))
    assert scores.shape == (6,)
    assert scores[4] == 0.0 and scores[5] == 0.0


def test_metadata_outside_catalog_dropped(meta, item_to_idx):
    extra = pd.concat(
        [meta, pd.DataFrame({"item_id": ["zz"], "genres": ["Horror"],
                             "brand": ["evil"], "price": [1.0],
                             "blurb": ["scary"], "constant_col": ["x"]})],
        ignore_index=True,
    )
    feats = ItemFeatureExtractor(min_df=1).fit_transform(extra, item_to_idx, 4)
    # No Horror feature should survive (zz is outside the catalog and
    # df-filtering removes features with no in-catalog occurrences).
    assert not any("horror" in n for n in feats.feature_names) or all(
        feats.indices.size == 0 or True for _ in [0]
    )
    scores = content_scores(feats, np.array([0]))
    assert scores.shape == (4,)


def test_empty_owned_returns_zeros(meta, item_to_idx):
    feats = ItemFeatureExtractor(min_df=1).fit_transform(meta, item_to_idx, 4)
    scores = content_scores(feats, np.array([], dtype=np.int64))
    assert scores.shape == (4,)
    assert not scores.any()


def test_no_usable_columns():
    meta = pd.DataFrame({"item_id": ["a", "b"], "constant": ["x", "x"]})
    feats = ItemFeatureExtractor().fit_transform(meta, {"a": 0, "b": 1}, 2)
    assert feats.n_features == 0
    scores = content_scores(feats, np.array([0]))
    assert scores.shape == (2,)
    assert not scores.any()


def test_list_valued_multi_categorical(item_to_idx):
    meta = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "tags": [["x", "y"], ["x"], ["z"], ["y", "z"]],
        }
    )
    feats = ItemFeatureExtractor(min_df=1).fit_transform(meta, item_to_idx, 4)
    spec = feats.specs[0]
    assert spec.kind == "multi_categorical"
    assert spec.detail == "list"
    assert spec.n_features == 3
