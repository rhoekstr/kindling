"""Steam loader — the realistic-protocol tier.

Kang & McAuley's Steam crawl (2010-2018): ~7.8M reviews, ~2.5M users,
~32k games, day-granularity timestamps, readable titles + genres/tags.

Deliberately NO k-core filtering: real catalogs are mostly long-tail,
and cold items / one-shot users are the population the content/LLM
channels exist for. 5-core preprocessing (the academic standard)
deletes that population; this loader keeps it. The only constraint is
implicit: users with a single interaction land entirely in train or
test and simply can't be evaluated — they still inform item statistics.

Files (cseweb.ucsd.edu/~wckang/): steam_reviews.json.gz,
steam_games.json.gz — Python-literal dicts per line (ast.literal_eval).
First parse caches to parquet; subsequent loads are instant.
"""

from __future__ import annotations

import ast
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

_CACHE = Path("~/.cache/kindling/steam").expanduser()


def _parse_reviews(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            try:
                d = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            u = d.get("username")
            i = d.get("product_id")
            t = d.get("date")
            if not u or not i or not t:
                continue
            rows.append((u, str(i), t, float(d.get("hours") or 0.0)))
    df = pd.DataFrame(rows, columns=["entity_id", "item_id", "date", "hours"])
    df["timestamp"] = (
        pd.to_datetime(df["date"], errors="coerce").astype("int64") // 10**9
    )
    df = df.dropna(subset=["timestamp"])
    return df[["entity_id", "item_id", "timestamp", "hours"]]


def _parse_games(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            try:
                d = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            gid = d.get("id")
            if not gid:
                url = d.get("url") or ""
                m = re.search(r"/app/(\d+)", url)
                gid = m.group(1) if m else None
            if not gid:
                continue
            rows.append({
                "item_id": str(gid),
                "title": d.get("title") or d.get("app_name"),
                "genres": list(d.get("genres") or []),
                "tags": list(d.get("tags") or []),
                "publisher": d.get("publisher"),
                "price": d.get("price") if isinstance(d.get("price"), (int, float)) else None,
            })
    return pd.DataFrame(rows).drop_duplicates(subset="item_id", keep="first")


def load_steam(
    root: str | Path = _CACHE,
    test_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train, test, items). Chronological global split by
    event order — deployment-shaped: train on the past, predict the
    future, cold items and all."""
    root = Path(root).expanduser()
    inter_cache = root / "interactions.parquet"
    items_cache = root / "items.parquet"
    if inter_cache.exists():
        inter = pd.read_parquet(inter_cache)
    else:
        inter = _parse_reviews(root / "steam_reviews.json.gz")
        inter.to_parquet(inter_cache)
    if items_cache.exists():
        items = pd.read_parquet(items_cache)
    else:
        items = _parse_games(root / "steam_games.json.gz")
        items.to_parquet(items_cache)
    inter = inter.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cut = int(len(inter) * (1.0 - test_fraction))
    return inter.iloc[:cut].copy(), inter.iloc[cut:].copy(), items
