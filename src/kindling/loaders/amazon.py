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


def load(data_file: str | Path, test_fraction: float = 0.1) -> DatasetSplit:
    """Parse a gzipped Amazon 5-core JSONL file into canonical format."""
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
    # No per-item category in the 5-core file; metadata is a separate
    # download. Omit the items frame - callers who want calibration can
    # attach metadata after loading.
    return DatasetSplit(
        name="amazon",
        train=train,
        test=test,
        items=None,
        description=(
            "Amazon 5-core reviews; rating-driven positive/negative signal; "
            "tests cost graph via explicit low ratings."
        ),
    )
