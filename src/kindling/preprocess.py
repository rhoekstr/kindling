"""Centralized interaction preprocessing.

Every positive-signal builder in kindling reads its weight per row from
a single column (``_interaction_weight``) the preprocessor attaches
here. This keeps rating-awareness out of the individual builders
(cooc, cosine, ALS, persona, paths, popularity) and out of anything
downstream - they just read the column and stay agnostic.

Auto-detection:
- ``use_ratings=None`` (default): detect. Rating column present and
  has any non-null numeric value → rating-weighted.
- ``use_ratings=True``: force on (raise if no rating column).
- ``use_ratings=False``: force off (ignore the column if present).

Weight transform (when ratings are used):
    w = max(0, (rating - threshold) / (scale_max - threshold))
    clipped to [0, 1], NaN rating -> 1.0 (implicit positive).

Ratings below the threshold contribute zero positive weight - the cost
graph handles them as negative signal. Zero double-counting.

Returns:
    (processed_df, context)
    processed_df: copy with the ``_interaction_weight`` column.
    context: InteractionContext metadata object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEIGHT_COLUMN = "_interaction_weight"


@dataclass(frozen=True)
class InteractionContext:
    """Metadata about how a dataset was preprocessed.

    Attached to the engine at fit time so downstream code (ADRs,
    diagnostics, persistence) can introspect whether ratings were
    used and how many rows were dropped.
    """

    uses_ratings: bool
    rating_threshold: float
    rating_scale_max: float
    n_rows_in: int
    n_rows_out: int
    n_rows_zero_weight: int

    def as_dict(self) -> dict[str, object]:
        return {
            "uses_ratings": self.uses_ratings,
            "rating_threshold": self.rating_threshold,
            "rating_scale_max": self.rating_scale_max,
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "n_rows_zero_weight": self.n_rows_zero_weight,
        }


def preprocess_interactions(
    interactions: pd.DataFrame,
    use_ratings: bool | None = None,
    rating_threshold: float = 2.5,
    rating_scale_max: float = 5.0,
) -> tuple[pd.DataFrame, InteractionContext]:
    """Return a copy of ``interactions`` with a ``_interaction_weight``
    column, plus a context object describing the transform."""
    n_in = len(interactions)
    uses_ratings = _resolve_use_ratings(interactions, use_ratings)

    df = interactions.copy()
    if uses_ratings:
        ratings = pd.to_numeric(df["rating"], errors="coerce").to_numpy(dtype=np.float64)
        missing = np.isnan(ratings)
        denom = max(rating_scale_max - rating_threshold, 1e-9)
        weights = np.maximum(0.0, (ratings - rating_threshold) / denom)
        weights = np.clip(weights, 0.0, 1.0)
        weights[missing] = 1.0  # NaN ratings -> implicit positive
    else:
        weights = np.ones(n_in, dtype=np.float64)

    df[WEIGHT_COLUMN] = weights.astype(np.float32)
    n_zero = int((df[WEIGHT_COLUMN] == 0.0).sum())

    context = InteractionContext(
        uses_ratings=uses_ratings,
        rating_threshold=rating_threshold,
        rating_scale_max=rating_scale_max,
        n_rows_in=n_in,
        n_rows_out=n_in,
        n_rows_zero_weight=n_zero,
    )
    return df, context


def _resolve_use_ratings(interactions: pd.DataFrame, user_choice: bool | None) -> bool:
    has_rating = "rating" in interactions.columns
    has_any_value = False
    if has_rating:
        try:
            has_any_value = bool(
                pd.to_numeric(interactions["rating"], errors="coerce").notna().any()
            )
        except (TypeError, ValueError):
            has_any_value = False

    if user_choice is True:
        if not has_rating:
            raise ValueError("use_ratings=True but the interactions have no 'rating' column.")
        if not has_any_value:
            raise ValueError("use_ratings=True but the 'rating' column has no numeric values.")
        return True
    if user_choice is False:
        return False
    # Auto: use ratings iff the column exists AND has numeric values.
    return has_rating and has_any_value


def weights_of(interactions: pd.DataFrame) -> np.ndarray:
    """Read the per-row weight column from a preprocessed DataFrame.

    Falls back to an all-ones vector when the DataFrame hasn't been
    through preprocess_interactions (e.g. unit tests that build signals
    directly). This preserves the pre-preprocessor behavior as the
    implicit default - no builder breaks if it receives a raw df.
    """
    if WEIGHT_COLUMN in interactions.columns:
        return interactions[WEIGHT_COLUMN].to_numpy(dtype=np.float32)
    return np.ones(len(interactions), dtype=np.float32)
