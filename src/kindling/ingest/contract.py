"""Input contract for interactions (PRD §4.1-§4.2).

Required: entity_id, item_id.
Optional: timestamp, session_id, action_type, rating.

Phase 1 deliberately does not use the optional columns for scoring; it only
validates their presence is well-formed. Later phases read them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = ("entity_id", "item_id")
OPTIONAL_COLUMNS = ("timestamp", "session_id", "action_type", "rating")
VALID_ACTION_TYPES = frozenset(
    {
        "add",  # cart add, basket add, session add (positive)
        "remove",  # explicit removal (negative for cost graph)
        "positive_rating",  # explicit positive rating
        "negative_rating",  # explicit negative rating
        "rate",  # generic rating event (sign captured in rating col)
        "view",  # impression / view (weak positive)
        "review",  # written review (yelp / amazon) - treated like rate
        "purchase",  # transaction (tafeng / dunnhumby) - strong positive
        "checkin",  # location check-in (gowalla) - positive engagement
    }
)


class InteractionContractError(ValueError):
    """Raised when an input DataFrame violates the interaction contract."""


@dataclass(frozen=True)
class InteractionSchema:
    """Resolved schema for a specific input DataFrame: which optional columns
    are actually present, plus normalized dtypes."""

    has_timestamp: bool
    has_session_id: bool
    has_action_type: bool
    has_rating: bool

    @property
    def supports_time_decay(self) -> bool:
        return self.has_timestamp

    @property
    def supports_sessions(self) -> bool:
        return self.has_timestamp or self.has_session_id

    @property
    def supports_cost_graph(self) -> bool:
        return self.has_action_type


def _check_required(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise InteractionContractError(f"Expected pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise InteractionContractError(f"Missing required columns: {missing}")
    if len(df) == 0:
        raise InteractionContractError("Interaction DataFrame is empty")
    if df["entity_id"].isna().any():
        raise InteractionContractError("entity_id contains null values")
    if df["item_id"].isna().any():
        raise InteractionContractError("item_id contains null values")


def _check_timestamp(ts: pd.Series) -> None:
    if not pd.api.types.is_datetime64_any_dtype(ts):
        try:
            pd.to_datetime(ts, errors="raise")
        except (ValueError, TypeError) as e:
            raise InteractionContractError(
                f"timestamp column is not datetime-convertible: {e}"
            ) from e
    if pd.api.types.is_datetime64_any_dtype(ts) and ts.isna().any():
        raise InteractionContractError("timestamp contains NaT values")


def _check_action_type(col: pd.Series) -> None:
    unknown = set(col.dropna().unique()) - VALID_ACTION_TYPES
    if unknown:
        raise InteractionContractError(
            f"action_type contains unknown values: {sorted(unknown)}. "
            f"Expected one of {sorted(VALID_ACTION_TYPES)}"
        )


def validate_interactions(df: pd.DataFrame) -> InteractionSchema:
    """Validate required columns, dtypes, and optional column well-formedness.

    Returns a schema describing which optional columns are usable.
    Raises InteractionContractError with a specific message on any violation.
    """
    _check_required(df)

    has_timestamp = "timestamp" in df.columns
    if has_timestamp:
        _check_timestamp(df["timestamp"])

    has_session_id = "session_id" in df.columns
    if has_session_id and df["session_id"].isna().any():
        raise InteractionContractError(
            "session_id contains null values; either drop the column or fill all rows"
        )

    has_action_type = "action_type" in df.columns
    if has_action_type:
        _check_action_type(df["action_type"])

    has_rating = "rating" in df.columns
    if has_rating and not pd.api.types.is_numeric_dtype(df["rating"]):
        raise InteractionContractError("rating column must be numeric")

    return InteractionSchema(
        has_timestamp=has_timestamp,
        has_session_id=has_session_id,
        has_action_type=has_action_type,
        has_rating=has_rating,
    )


def canonicalize(df: pd.DataFrame, schema: InteractionSchema) -> pd.DataFrame:
    """Return a copy with canonical dtypes and consistent column ordering.

    Phase 1: entity_id and item_id are kept as-is (no interning yet — that's
    Phase 2 when the path tree and graph want compact integer IDs).
    """
    out = df.copy()
    if schema.has_timestamp and not pd.api.types.is_datetime64_any_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"])

    ordered_cols = list(REQUIRED_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in out.columns]
    return out[ordered_cols].reset_index(drop=True)
