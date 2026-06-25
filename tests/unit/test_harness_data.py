"""Eval-harness input handling: CSV aliasing, chronological split, resolution."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kindling.harness.data import (
    chronological_split,
    load_interactions_csv,
    read_csv_aliased,
    resolve_dataset,
)


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    df.to_csv(path, index=False)
    return path


def test_load_interactions_csv_normalizes_aliases(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path / "log.csv",
        pd.DataFrame(
            {
                "user_id": [1, 1, 2],
                "product_id": [10, 11, 10],
                "ts": ["2026-01-01", "2026-01-02", "2026-01-03"],
            }
        ),
    )
    df = load_interactions_csv(csv)
    assert list(df.columns) == ["entity_id", "item_id", "timestamp"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_load_interactions_csv_missing_required_raises(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "bad.csv", pd.DataFrame({"user_id": [1], "qty": [2]}))
    with pytest.raises(ValueError, match="missing required column"):
        load_interactions_csv(csv)


def test_read_csv_aliased_does_not_validate(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "meta.csv", pd.DataFrame({"product_id": [10], "color": ["red"]}))
    meta = read_csv_aliased(csv)
    assert "item_id" in meta.columns  # aliased, but no entity_id required


def test_chronological_split_uses_time_order() -> None:
    df = pd.DataFrame(
        {
            "entity_id": list(range(10)),
            "item_id": list(range(10)),
            "timestamp": pd.to_datetime([f"2026-01-{d:02d}" for d in range(1, 11)]),
        }
    )
    train, test = chronological_split(df, test_fraction=0.2)
    assert len(test) >= 1 and len(train) >= 1
    # No leakage: every test timestamp is strictly after every train timestamp.
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_chronological_split_without_timestamp_falls_back_to_tail() -> None:
    df = pd.DataFrame({"entity_id": list(range(10)), "item_id": list(range(10))})
    train, test = chronological_split(df, test_fraction=0.3)
    assert len(train) == 7 and len(test) == 3


def test_chronological_split_rejects_bad_fraction() -> None:
    df = pd.DataFrame({"entity_id": [1], "item_id": [1]})
    with pytest.raises(ValueError, match="test_fraction"):
        chronological_split(df, test_fraction=1.5)


def test_resolve_dataset_from_csv(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path / "mydata.csv",
        pd.DataFrame(
            {
                "entity_id": [1, 1, 2, 2, 3],
                "item_id": [10, 11, 10, 12, 11],
                "timestamp": pd.to_datetime(
                    ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
                ),
            }
        ),
    )
    split = resolve_dataset(csv, test_fraction=0.2)
    assert split.name == "mydata"
    assert len(split.train) + len(split.test) == 5
    assert split.items is None


def test_resolve_dataset_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match=r"Unknown dataset|not a file"):
        resolve_dataset("definitely-not-a-real-dataset")
