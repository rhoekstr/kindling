"""Shared helper: turn validated interactions + inferred session ids into
ordered per-session item sequences, one item per event."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SessionSequence:
    """An ordered sequence of items for a single session."""

    session_id: int
    entity_id: object
    items: tuple[object, ...]
    # End-of-session timestamp as a Unix-seconds float. Used by decay at query
    # time; None when the input had no timestamp column.
    end_timestamp: float | None


def sessions_from_interactions(
    interactions: pd.DataFrame,
    session_ids: np.ndarray,
) -> Iterator[SessionSequence]:
    """Yield one ``SessionSequence`` per session id, with items ordered by
    timestamp (or by original row order when no timestamp column exists)."""
    sort_cols: list[str] = []
    if "timestamp" in interactions.columns:
        sort_cols.append("timestamp")
    # Stable-sort by (session_id, timestamp) so rows with identical timestamps
    # retain their relative input order.
    work = interactions.assign(_session_id=session_ids)
    sort_keys = ["_session_id", *sort_cols] if sort_cols else ["_session_id"]
    work = work.sort_values(sort_keys, kind="mergesort")

    for session_id, group in work.groupby("_session_id", sort=False):
        items = tuple(group["item_id"].tolist())
        end_ts: float | None = None
        if "timestamp" in group.columns:
            end_ts = float(group["timestamp"].iloc[-1].timestamp())
        entity = group["entity_id"].iloc[0]
        yield SessionSequence(
            session_id=int(session_id),  # type: ignore[arg-type]
            entity_id=entity,
            items=items,
            end_timestamp=end_ts,
        )
