"""Gowalla check-ins loader (plan Phase 7).

Gowalla is a location-based social-network check-in log: a user visits
a venue at a given time. Treating user -> venue check-ins as an
interaction graph gives a session-light, timestamp-rich corpus
complementary to ML-1M (rating bursts) and Instacart (basket-heavy).

Two file layouts are supported:

1. **SNAP raw check-in log** (preferred). Source:
   https://snap.stanford.edu/data/loc-gowalla.html
   File: ``loc-gowalla_totalCheckins.txt[.gz]``
   Tab-separated: ``user\\tcheckin_time\\tlat\\tlon\\tlocation_id``.
2. **Academic LightGCN/NGCF split**. Source: published code releases.
   Files: ``train.txt`` / ``test.txt`` with one user per line
   ``user_id item_id1 item_id2 ...``. No timestamps; path signals
   degrade to manual_fallback sessions.

Pass ``data_dir`` pointing at a directory that holds either layout.
Auto-download is intentionally skipped because the user already has the
file (this is a benchmark dataset that ships with reproducibility kits).

CLI:
    python -m kindling.benchmarks.harness --dataset gowalla \\
        --data-dir ~/.cache/kindling/gowalla
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from kindling.loaders._base import DatasetSplit


class GowallaDataNotAvailableError(RuntimeError):
    """Raised when neither the SNAP nor the academic-split files are present."""


SNAP_FILES = ("loc-gowalla_totalCheckins.txt.gz", "loc-gowalla_totalCheckins.txt")
ACADEMIC_FILES = ("train.txt", "test.txt")


def _has_snap(base: Path) -> Path | None:
    for name in SNAP_FILES:
        path = base / name
        if path.exists():
            return path
    return None


def _has_academic(base: Path) -> bool:
    return all((base / name).exists() for name in ACADEMIC_FILES)


def _load_snap(path: Path, test_fraction: float) -> DatasetSplit:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[tuple[int, int, str]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            user, ts, _lat, _lon, loc = parts[:5]
            try:
                rows.append((int(user), int(loc), ts))
            except ValueError:
                continue
    if not rows:
        raise GowallaDataNotAvailableError(f"Parsed zero rows from {path}; file may be malformed.")
    df = pd.DataFrame(rows, columns=["entity_id", "item_id", "timestamp_str"])
    df["timestamp"] = pd.to_datetime(df["timestamp_str"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).drop(columns=["timestamp_str"]).reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    df["action_type"] = "checkin"
    df = df.sort_values(["entity_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Chronological per-user split.
    cutoff = df.groupby("entity_id")["timestamp"].quantile(1.0 - test_fraction).to_dict()
    train_mask = df["timestamp"] <= df["entity_id"].map(cutoff)
    train = df[train_mask].reset_index(drop=True)
    test = df[~train_mask].reset_index(drop=True)
    return DatasetSplit(
        name="gowalla",
        train=train,
        test=test,
        items=None,
        description=(
            "Gowalla SNAP check-ins; user -> venue with real timestamps; "
            "session-light, timestamp-rich corpus."
        ),
    )


def _load_academic(base: Path) -> DatasetSplit:
    train_rows: list[tuple[int, int]] = []
    test_rows: list[tuple[int, int]] = []
    for split_path, sink in [(base / "train.txt", train_rows), (base / "test.txt", test_rows)]:
        with open(split_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    user = int(parts[0])
                    items = [int(p) for p in parts[1:]]
                except ValueError:
                    continue
                for it in items:
                    sink.append((user, it))
    if not train_rows:
        raise GowallaDataNotAvailableError(
            f"Parsed zero rows from {base / 'train.txt'}; file may be malformed."
        )
    train = pd.DataFrame(train_rows, columns=["entity_id", "item_id"])
    train["action_type"] = "checkin"
    test = pd.DataFrame(test_rows, columns=["entity_id", "item_id"])
    test["action_type"] = "checkin"
    return DatasetSplit(
        name="gowalla",
        train=train,
        test=test,
        items=None,
        description=(
            "Gowalla academic split (NGCF/LightGCN benchmark); no "
            "timestamps so path signals degrade to manual_fallback sessions."
        ),
    )


def load(data_dir: str | Path, test_fraction: float = 0.1) -> DatasetSplit:
    base = Path(data_dir)
    snap = _has_snap(base)
    if snap is not None:
        return _load_snap(snap, test_fraction)
    if _has_academic(base):
        return _load_academic(base)
    raise GowallaDataNotAvailableError(
        f"Gowalla data not found under {base}. Provide either "
        f"{SNAP_FILES[0]} (SNAP raw) or {ACADEMIC_FILES} (academic split). "
        "SNAP source: https://snap.stanford.edu/data/loc-gowalla.html"
    )
