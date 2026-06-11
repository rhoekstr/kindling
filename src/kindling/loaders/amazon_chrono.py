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
        inter = pd.DataFrame(
            rows, columns=["entity_id", "item_id", "timestamp", "rating"]
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        inter.to_parquet(cache)
    inter = inter.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cut = int(len(inter) * (1.0 - test_fraction))
    return inter.iloc[:cut].copy(), inter.iloc[cut:].copy()
