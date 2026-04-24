"""Yelp 2018 review loader (plan Phase 7).

Two file layouts are supported:

1. **Yelp Academic Dataset JSON** (preferred — keeps timestamps + ratings).
   File: ``yelp_academic_dataset_review.json[.gz]`` published per
   challenge year. Each line: ``{user_id, business_id, stars, date, ...}``.
2. **Academic LightGCN/NGCF split** (``train.txt`` / ``test.txt``).
   No timestamps, no ratings; useful when the goal is comparing against
   published GCN baselines.

Pass ``data_dir`` pointing at a directory that holds either layout.

CLI:
    python -m kindling.benchmarks.harness --dataset yelp2018 \\
        --data-dir ~/.cache/kindling/yelp2018
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from kindling.loaders._base import DatasetSplit


class YelpDataNotAvailableError(RuntimeError):
    pass


JSON_FILES = (
    "yelp_academic_dataset_review.json.gz",
    "yelp_academic_dataset_review.json",
    "review.json.gz",
    "review.json",
)
ACADEMIC_FILES = ("train.txt", "test.txt")


def _find_json(base: Path) -> Path | None:
    for name in JSON_FILES:
        path = base / name
        if path.exists():
            return path
    return None


def _load_json(path: Path, test_fraction: float) -> DatasetSplit:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, object]] = []
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
                    "entity_id": rec.get("user_id"),
                    "item_id": rec.get("business_id"),
                    "rating": float(rec.get("stars", 0.0)),
                    "timestamp": pd.to_datetime(rec.get("date"), errors="coerce"),
                    "action_type": "review",
                }
            )
    if not rows:
        raise YelpDataNotAvailableError(
            f"Parsed zero reviews from {path}; file may be empty or malformed."
        )
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["timestamp", "entity_id", "item_id"]).reset_index(drop=True)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cutoff = int(len(df) * (1.0 - test_fraction))
    train = df.iloc[:cutoff].copy()
    test = df.iloc[cutoff:].copy()
    return DatasetSplit(
        name="yelp2018",
        train=train,
        test=test,
        items=None,
        description=(
            "Yelp Academic reviews; user -> business with star ratings + "
            "timestamps; ratings drive cost-graph negative signal."
        ),
    )


def _load_academic(base: Path) -> DatasetSplit:
    train_rows: list[tuple[str, str]] = []
    test_rows: list[tuple[str, str]] = []
    for split_path, sink in [(base / "train.txt", train_rows), (base / "test.txt", test_rows)]:
        with open(split_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                user = parts[0]
                for it in parts[1:]:
                    sink.append((user, it))
    if not train_rows:
        raise YelpDataNotAvailableError(
            f"Parsed zero rows from {base / 'train.txt'}; file may be malformed."
        )
    train = pd.DataFrame(train_rows, columns=["entity_id", "item_id"])
    train["action_type"] = "review"
    test = pd.DataFrame(test_rows, columns=["entity_id", "item_id"])
    test["action_type"] = "review"
    return DatasetSplit(
        name="yelp2018",
        train=train,
        test=test,
        items=None,
        description=(
            "Yelp 2018 academic split (NGCF/LightGCN benchmark); no "
            "timestamps so path signals degrade to manual_fallback sessions."
        ),
    )


def load(data_dir: str | Path, test_fraction: float = 0.1) -> DatasetSplit:
    base = Path(data_dir)
    json_path = _find_json(base)
    if json_path is not None:
        return _load_json(json_path, test_fraction)
    if all((base / name).exists() for name in ACADEMIC_FILES):
        return _load_academic(base)
    raise YelpDataNotAvailableError(
        f"Yelp data not found under {base}. Provide either "
        f"{JSON_FILES[0]} (academic JSON) or {ACADEMIC_FILES} (NGCF/LightGCN split). "
        "JSON source: https://www.yelp.com/dataset"
    )
