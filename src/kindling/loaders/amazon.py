"""Amazon Reviews loader (plan Phase 7).

Reads the 5-core JSONL format published by the Amazon Customer Reviews
project. Each row: ``{reviewerID, asin, overall (rating), unixReviewTime,
reviewText, ...}``. We map ``reviewerID -> entity_id``, ``asin ->
item_id``, ``overall -> rating``, and synthesize timestamps from
``unixReviewTime``.

Source: https://nijianmo.github.io/amazon/ (5-core datasets per category).
Because file sizes vary (10 MB to 10 GB+) and the mirror requires an
explicit category pick, the loader takes a path to a pre-downloaded
``.json.gz`` file rather than auto-downloading.

Usage:

    python -m kindling.benchmarks.harness --dataset amazon \\
        --data-file /path/to/Electronics_5.json.gz
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from kindling.loaders._base import DatasetSplit


class AmazonReviewsDataNotAvailableError(RuntimeError):
    pass


def load(
    data_file: str | Path,
    test_fraction: float = 0.1,
    meta_file: str | Path | None = None,
) -> DatasetSplit:
    """Parse a gzipped Amazon 5-core JSONL file into canonical format.

    When ``meta_file`` is provided (e.g., the 2023 Amazon Reviews
    ``meta_*.jsonl`` file), per-item brand (``store``) is attached to
    the returned ``DatasetSplit.items`` frame. The 2023 dataset's
    ``categories`` field is empty for all items (the 2018 hierarchical
    category tree was dropped in the rewrite); brand is the only
    flat-partition signal that survived.
    """
    path = Path(data_file)
    if not path.exists():
        raise AmazonReviewsDataNotAvailableError(
            f"Amazon reviews file not found at {path}. Download a 5-core "
            "category JSONL.gz from https://nijianmo.github.io/amazon/"
        )

    rows: list[dict[str, object]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(
                {
                    "entity_id": rec["reviewerID"],
                    "item_id": rec["asin"],
                    "rating": float(rec.get("overall", 0.0)),
                    "timestamp": pd.to_datetime(
                        int(rec.get("unixReviewTime", 0)), unit="s", errors="coerce"
                    ),
                    "action_type": "rate",
                }
            )
    if not rows:
        raise AmazonReviewsDataNotAvailableError(
            f"No reviews parsed from {path}; the file may be empty or malformed."
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cutoff = int(len(df) * (1.0 - test_fraction))
    train = df.iloc[:cutoff].copy()
    test = df.iloc[cutoff:].copy()

    items_frame = None
    if meta_file is not None:
        meta_path = Path(meta_file)
        if meta_path.exists():
            items_frame = _parse_meta(meta_path)

    return DatasetSplit(
        name="amazon",
        train=train,
        test=test,
        items=items_frame,
        description=(
            "Amazon 5-core reviews; rating-driven positive/negative signal; "
            "tests cost graph via explicit low ratings."
            + (" Items frame includes brand (store) when meta_file is present." if items_frame is not None else "")
        ),
    )


def _parse_meta(meta_path: Path) -> pd.DataFrame:
    """Parse meta_*.jsonl from the 2023 Amazon Reviews dataset.

    Returns a DataFrame with columns:
        item_id   — parent_asin (corresponds to reviews 'asin')
        store     — brand string (None when missing)
        category  — main_category (always 'All Beauty' for this loader's
                    typical input; kept for parity with kindling's
                    convention of one category column per item)
    """
    rows: list[dict[str, object]] = []
    opener = gzip.open if meta_path.suffix == ".gz" else open
    with opener(meta_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("parent_asin")
            if not asin:
                continue
            rows.append(
                {
                    "item_id": asin,
                    "store": rec.get("store"),
                    "category": rec.get("main_category"),
                }
            )
    return pd.DataFrame(rows)
