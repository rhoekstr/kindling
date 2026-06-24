"""Apples-to-apples comparison: kindling vs. industry-standard baselines.

Runs kindling's Engine and the baselines defined in ``benchmarks.baselines``
against the same chronological train/test split on a reference dataset.
Emits accuracy metrics (NDCG, Recall, MRR, Hit), catalog coverage, fit
wall-time, and per-recommend p50/p95 latency.

CLI:
    python -m kindling.benchmarks.comparison --dataset movielens-1m \
        --output bench/reports/baselines_comparison.json
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kindling.loaders import (
    amazon,
    dunnhumby,
    gowalla,
    instacart,
    movielens,
    retailrocket,
    synthetic,
    tafeng,
    yelp,
)
from kindling.loaders._base import DatasetSplit


def _cache_dir() -> Path:
    import os

    return Path(os.environ.get("KINDLING_CACHE_DIR", Path.home() / ".cache" / "kindling"))


def _load_dataset(name: str, test_fraction: float) -> DatasetSplit:
    if name == "movielens-1m":
        return movielens.load_1m(test_fraction=test_fraction)
    if name == "synthetic-grocery":
        return synthetic.make_grocery(
            n_entities=1500,
            n_items_per_category=20,
            n_categories=8,
            n_sessions_per_entity=10,
            items_per_session=6,
            test_fraction=test_fraction,
        )
    if name == "synthetic-grocery-deep":
        # Longer sessions (10 items) give the path signals enough sequential
        # depth to separate from item-item cosine. Matches the "real session"
        # shape of grocery / e-commerce baskets.
        return synthetic.make_grocery(
            n_entities=1500,
            n_items_per_category=25,
            n_categories=8,
            n_sessions_per_entity=12,
            items_per_session=10,
            test_fraction=test_fraction,
        )
    cache = _cache_dir()
    if name == "retailrocket":
        return retailrocket.load(cache / "retailrocket", test_fraction=test_fraction)
    if name == "instacart":
        return instacart.load(cache / "instacart", test_fraction=test_fraction)
    if name == "gowalla":
        return gowalla.load(cache / "gowalla", test_fraction=test_fraction)
    if name == "yelp2018":
        return yelp.load(cache / "yelp2018", test_fraction=test_fraction)
    if name == "tafeng":
        return tafeng.load(cache / "tafeng", test_fraction=test_fraction)
    if name == "dunnhumby":
        return dunnhumby.load(cache / "dunnhumby", test_fraction=test_fraction)
    if name == "amazon-beauty":
        return _load_amazon_5core(
            cache / "amazon-beauty", test_fraction=test_fraction, label="amazon-beauty"
        )
    if name == "amazon-book":
        return _load_amazon_5core(
            cache / "amazon-book", test_fraction=test_fraction, label="amazon-book"
        )
    if name == "amazon-book-chrono":
        # Realistic-protocol tier for books: 2014 5-core reviews with a
        # chronological global split (vs the timestamp-less LightGCN
        # academic split that plain "amazon-book" falls back to).
        from kindling.loaders.amazon_chrono import load_amazon_chrono, load_amazon_meta

        book_dir = Path("~/.cache/kindling/amazon-book")
        train, test = load_amazon_chrono(
            book_dir / "reviews_Books_5.json.gz",
            cache_dir=book_dir,
            test_fraction=test_fraction,
        )
        items = None
        if (book_dir.expanduser() / "meta_Books.json.gz").exists():
            items = load_amazon_meta(
                book_dir / "meta_Books.json.gz",
                cache_dir=book_dir,
                catalog=set(train["item_id"].unique()),
            )
        return DatasetSplit(
            name="amazon-book-chrono",
            train=train,
            test=test,
            items=items,
            description="Amazon Books 5-core 2014, chronological global split",
        )
    if name == "steam":
        # Realistic-protocol tier: NO k-core filtering, chronological
        # global split. Cold items and one-shot users included — the
        # population 5-core academic preprocessing deletes.
        from kindling.loaders.steam import load_steam

        train, test, items = load_steam(test_fraction=test_fraction)
        return DatasetSplit(
            name="steam",
            train=train,
            test=test,
            items=items,
            description="Steam reviews 2010-2018, no k-core, chronological split",
        )
    raise ValueError(f"Unknown dataset: {name}")


def _load_amazon_5core(data_dir: Path, test_fraction: float, label: str) -> DatasetSplit:
    """Resolve an Amazon dataset under ``data_dir``.

    Two formats supported, in priority order:
    1. McAuley 5-core JSONL.gz (preferred - has timestamps + ratings).
    2. LightGCN academic train.txt/test.txt split (no timestamps;
       used as a fallback when the JSONL isn't available locally).
    """
    if not data_dir.exists():
        raise amazon.AmazonReviewsDataNotAvailableError(
            f"Amazon data dir {data_dir} does not exist."
        )
    candidates = sorted(data_dir.glob("*5*.json*"))
    if candidates:
        meta_candidates = sorted(data_dir.glob("meta_*.jsonl*"))
        meta_file = meta_candidates[0] if meta_candidates else None
        split = amazon.load(
            candidates[0],
            test_fraction=test_fraction,
            meta_file=meta_file,
        )
        return DatasetSplit(
            name=label,
            train=split.train,
            test=split.test,
            items=split.items,
            description=f"{label}: {split.description}",
        )
    # LightGCN academic split fallback.
    train_path = data_dir / "train.txt"
    test_path = data_dir / "test.txt"
    if train_path.exists() and test_path.exists():
        return _load_academic_split(train_path, test_path, name=label, action_type="rate")
    raise amazon.AmazonReviewsDataNotAvailableError(
        f"No 5-core JSON file or LightGCN academic split (train.txt/test.txt) "
        f"under {data_dir} for {label}. "
        "Download a 5-core category JSONL.gz from "
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ "
        "or the academic split from "
        "https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data"
    )


def _load_academic_split(
    train_path: Path, test_path: Path, name: str, action_type: str
) -> DatasetSplit:
    """Parse a LightGCN-style train.txt/test.txt pair.

    Each line: ``user_id item_id1 item_id2 ...``. No timestamps; path
    signals will degrade to manual_fallback sessions.
    """
    train_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []
    for path, sink in [(train_path, train_rows), (test_path, test_rows)]:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                user = parts[0]
                for it in parts[1:]:
                    sink.append((user, it))
    train = pd.DataFrame(train_rows, columns=["entity_id", "item_id"])
    train["action_type"] = action_type
    test = pd.DataFrame(test_rows, columns=["entity_id", "item_id"])
    test["action_type"] = action_type
    return DatasetSplit(
        name=name,
        train=train,
        test=test,
        items=None,
        description=(
            f"{name}: LightGCN academic split (NGCF/LightGCN benchmark); "
            "no timestamps, path signals degrade to manual_fallback sessions."
        ),
    )


# NOTE: the kindling-vs-baselines comparison runner (_EngineAdapter,
# run_comparison, main) was removed with the v1 engine in the production
# consolidation. This module is retained only as the dataset-loading
# registry (_load_dataset) used by the verification harness. The frozen
# comparison results live in bench/reports/baselines_comparison*.json.
