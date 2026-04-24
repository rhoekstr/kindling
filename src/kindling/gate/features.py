"""Context feature extraction for the gating network.

Per-entity summary features the gate uses as input. All features are
scalar floats; the gate's first layer learns the mapping from these to
hidden representation.

Features (in fixed order - must match the gate's input dimension):
0. log1p(n_interactions)            - how active the entity is
1. session_density                  - interactions per unique session
2. mean_rating                      - mean of _interaction_weight across entity's rows
3. rating_std                       - std of same
4. item_diversity                   - unique items / total interactions
5. recency_log_days                 - log-days between first and last interaction
6. has_persona_match                - 1.0 if entity is assigned to a persona, else 0.0
7. has_als_factor                   - 1.0 if ALS factors are fitted, else 0.0

Missing data defaults documented per feature. Context features come
from Engine state, not from the current query, so they're computed
once per entity at fit time (cached on the Engine).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from kindling.engine import Engine


CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    "log_n_interactions",
    "session_density",
    "mean_rating",
    "rating_std",
    "item_diversity",
    "recency_log_days",
    "has_persona_match",
    "has_als_factor",
)


def compute_context_features(engine: "Engine") -> dict[object, np.ndarray]:
    """Return a dict mapping entity_id -> (n_context_features,) array.

    Builds the static per-entity context matrix the gate consumes at
    both train and inference time. Uses the Engine's fitted state so
    every feature is available by the time the gate fits.
    """
    from kindling.preprocess import WEIGHT_COLUMN

    assert engine._interactions is not None
    df = engine._interactions
    has_weights = WEIGHT_COLUMN in df.columns
    has_als = engine._als_factors is not None
    persona_idx = engine._persona_index

    # Groupby once and walk over the resulting groups.
    out: dict[object, np.ndarray] = {}
    grouped = df.groupby("entity_id", sort=False)
    for entity, group in grouped:
        n_interactions = len(group)
        n_unique_items = group["item_id"].nunique()
        item_diversity = n_unique_items / max(n_interactions, 1)

        sessions_count = 1
        if "session_id" in group.columns:
            sessions_count = group["session_id"].nunique()
        session_density = n_interactions / max(sessions_count, 1)

        if has_weights:
            w = group[WEIGHT_COLUMN].to_numpy(dtype=np.float64)
            mean_rating = float(w.mean()) if w.size else 1.0
            rating_std = float(w.std()) if w.size else 0.0
        else:
            mean_rating = 1.0
            rating_std = 0.0

        if "timestamp" in group.columns:
            ts = pd.to_datetime(group["timestamp"])
            days = (ts.max() - ts.min()).total_seconds() / 86400.0
            recency_log_days = float(np.log1p(max(days, 0.0)))
        else:
            recency_log_days = 0.0

        has_persona = 0.0
        if persona_idx is not None and persona_idx.n_personas > 0:
            p_idx = persona_idx.persona_of_entity(entity)
            has_persona = 1.0 if p_idx >= 0 else 0.0

        out[entity] = np.array(
            [
                float(np.log1p(n_interactions)),
                session_density,
                mean_rating,
                rating_std,
                item_diversity,
                recency_log_days,
                has_persona,
                1.0 if has_als else 0.0,
            ],
            dtype=np.float32,
        )
    return out
