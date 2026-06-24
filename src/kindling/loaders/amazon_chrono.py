"""Amazon chronological loader — realistic-protocol tier.

McAuley 5-core review files (2014 SNAP format, strict JSON lines),
loaded with a chronological GLOBAL split instead of the LightGCN
academic random split. With timestamps present, the full channel stack
(trend, transitions, last-item) activates; the 5-core filter is in the
source data (unavoidable), but the chronological boundary still yields
items that are cold *relative to train*.

First parse caches to parquet.
"""

from __future__ import annotations

import ast
import gzip
import json
from pathlib import Path

import pandas as pd


def load_amazon_chrono(
    reviews_path: str | Path,
    cache_dir: str | Path,
    test_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (train, test) with entity_id/item_id/timestamp columns."""
    cache_dir = Path(cache_dir).expanduser()
    cache = cache_dir / "interactions_chrono.parquet"
    if cache.exists():
        inter = pd.read_parquet(cache)
    else:
        rows = []
        with gzip.open(Path(reviews_path).expanduser(), "rt", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                u = d.get("reviewerID")
                i = d.get("asin")
                t = d.get("unixReviewTime")
                if not u or not i or not t:
                    continue
                rows.append((u, i, float(t), float(d.get("overall") or 1.0)))
        inter = pd.DataFrame(rows, columns=["entity_id", "item_id", "timestamp", "rating"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        inter.to_parquet(cache)
    inter = inter.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cut = int(len(inter) * (1.0 - test_fraction))
    return inter.iloc[:cut].copy(), inter.iloc[cut:].copy()


def load_amazon_meta(
    meta_path: str | Path,
    cache_dir: str | Path,
    catalog: set | None = None,
    extension_top_n: int = 200_000,
) -> pd.DataFrame:
    """Parse 2014 SNAP metadata (Python-literal lines) into an items frame.

    Keeps every item in `catalog` plus the `extension_top_n` best-selling
    items outside it (by salesRank — a legitimate external prior: a real
    system would consider unsold-but-ranked catalog items as candidates).
    Caps the open-catalog extension so a 2.4M-item metadata universe
    doesn't balloon the engine's index on a 24GB machine.
    """
    cache_dir = Path(cache_dir).expanduser()
    cache = cache_dir / "items_meta.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    with gzip.open(Path(meta_path).expanduser(), "rt", errors="replace") as f:
        for line in f:
            try:
                d = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            a = d.get("asin")
            if not a:
                continue
            cats = d.get("categories") or []
            flat = [c for sub in cats for c in (sub if isinstance(sub, list) else [sub])]
            sr = d.get("salesRank")
            rank = None
            if isinstance(sr, dict) and sr:
                rank = min(v for v in sr.values() if isinstance(v, (int, float)))
            rows.append(
                {
                    "item_id": a,
                    "title": d.get("title"),
                    "brand": d.get("brand"),
                    "categories": flat,
                    "price": d.get("price") if isinstance(d.get("price"), (int, float)) else None,
                    "sales_rank": rank,
                }
            )
    items = pd.DataFrame(rows).drop_duplicates(subset="item_id", keep="first")
    if catalog is not None:
        in_cat = items["item_id"].isin(catalog)
        ext = items.loc[~in_cat].sort_values("sales_rank", na_position="last").head(extension_top_n)
        items = pd.concat([items.loc[in_cat], ext], ignore_index=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    items.to_parquet(cache)
    return items
