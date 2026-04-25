"""Input contract tests."""

from __future__ import annotations

import pandas as pd
import pytest

from kindling.ingest.contract import (
    InteractionContractError,
    canonicalize,
    validate_interactions,
)


def test_minimal_valid_input() -> None:
    df = pd.DataFrame({"entity_id": [1, 2, 3], "item_id": ["a", "b", "c"]})
    schema = validate_interactions(df)
    assert not schema.has_timestamp
    assert not schema.has_session_id
    assert not schema.has_action_type
    assert not schema.has_rating
    assert not schema.supports_time_decay
    assert not schema.supports_sessions


def test_rejects_non_dataframe() -> None:
    with pytest.raises(InteractionContractError, match="pandas DataFrame"):
        validate_interactions([1, 2, 3])  # type: ignore[arg-type]


def test_rejects_empty() -> None:
    df = pd.DataFrame({"entity_id": [], "item_id": []})
    with pytest.raises(InteractionContractError, match="empty"):
        validate_interactions(df)


def test_rejects_missing_required_columns() -> None:
    df = pd.DataFrame({"entity_id": [1], "rating": [5.0]})
    with pytest.raises(InteractionContractError, match="Missing required columns"):
        validate_interactions(df)


def test_rejects_null_entity_id() -> None:
    df = pd.DataFrame({"entity_id": [1, None], "item_id": ["a", "b"]})
    with pytest.raises(InteractionContractError, match="entity_id"):
        validate_interactions(df)


def test_rejects_unknown_action_type() -> None:
    # "purchase" became canonical (tafeng / dunnhumby loaders).
    # An invented action type still rejects.
    df = pd.DataFrame({"entity_id": [1], "item_id": [2], "action_type": ["teleport"]})
    with pytest.raises(InteractionContractError, match="unknown"):
        validate_interactions(df)


def test_rejects_non_numeric_rating() -> None:
    df = pd.DataFrame({"entity_id": [1], "item_id": [2], "rating": ["five"]})
    with pytest.raises(InteractionContractError, match="numeric"):
        validate_interactions(df)


def test_timestamp_convertible_from_string() -> None:
    df = pd.DataFrame(
        {
            "entity_id": [1, 2],
            "item_id": ["a", "b"],
            "timestamp": ["2026-01-01", "2026-01-02"],
        }
    )
    schema = validate_interactions(df)
    assert schema.has_timestamp
    canon = canonicalize(df, schema)
    assert pd.api.types.is_datetime64_any_dtype(canon["timestamp"])


def test_rejects_nat_timestamp() -> None:
    df = pd.DataFrame(
        {
            "entity_id": [1, 2],
            "item_id": ["a", "b"],
            "timestamp": pd.to_datetime(["2026-01-01", pd.NaT]),
        }
    )
    with pytest.raises(InteractionContractError, match="NaT"):
        validate_interactions(df)


def test_canonicalize_column_order(tiny_interactions: pd.DataFrame) -> None:
    schema = validate_interactions(tiny_interactions)
    canon = canonicalize(tiny_interactions, schema)
    assert list(canon.columns) == ["entity_id", "item_id", "timestamp"]
