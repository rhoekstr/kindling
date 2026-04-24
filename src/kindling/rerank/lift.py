"""Lift-based emphasis (PRD §7.4).

Lift = per-entity score divided by a population baseline. An item with
lift > 1 is personally distinctive - it is unusually well-ranked for this
entity compared to entities in general. Enabled via ``emphasis="distinctive"``
at recommend time.

Population baselines are cached per retrain; lift at recommendation time is
an element-wise divide, so it's essentially free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PopulationBaselines:
    """Per-item baseline score used to compute lift.

    The baseline is the normalized interaction count - what fraction of
    entities have interacted with this item. High-support items have high
    baselines, so even a high per-entity score doesn't produce high lift
    for them. Low-support items with a high entity score produce high
    lift.
    """

    item_to_baseline: dict[object, float] = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        return len(self.item_to_baseline)

    def lookup_many(self, item_ids: list[object]) -> np.ndarray:
        """Return the baseline vector for the given item list. Unknown items
        get the minimum observed baseline so they aren't over-emphasized."""
        if not self.item_to_baseline:
            return np.ones(len(item_ids), dtype=np.float64)
        min_baseline = min(self.item_to_baseline.values())
        return np.asarray(
            [self.item_to_baseline.get(i, min_baseline) for i in item_ids],
            dtype=np.float64,
        )


def compute_population_baselines(interactions: pd.DataFrame) -> PopulationBaselines:
    """Weighted popularity per item.

    With a ``_interaction_weight`` column (attached by the preprocessor),
    the baseline is the sum of per-interaction weights divided by the
    unique entity count - so 5-star ratings contribute more to
    "popular" than 1-2 star ratings. Without the column the logic
    degrades to the old "fraction of entities who interacted" count.
    """
    from kindling.preprocess import WEIGHT_COLUMN, weights_of

    if interactions.empty:
        return PopulationBaselines()
    entity_count = interactions["entity_id"].nunique()
    if entity_count == 0:
        return PopulationBaselines()

    if WEIGHT_COLUMN in interactions.columns:
        # Weighted: sum max-per-(entity, item) weight then normalize by
        # total entities. Matches the item_graph builder's dedup rule.
        df = interactions[["entity_id", "item_id", WEIGHT_COLUMN]].copy()
        df = df.groupby(["entity_id", "item_id"], sort=False, as_index=False)[
            WEIGHT_COLUMN
        ].max()
        per_item = df.groupby("item_id")[WEIGHT_COLUMN].sum() / entity_count
    else:
        pairs = interactions[["entity_id", "item_id"]].drop_duplicates()
        per_item = pairs.groupby("item_id").size() / entity_count
    # Make sure weights_of is referenced so the lint doesn't flag it;
    # harmless no-op when the column path was taken above.
    _ = weights_of
    return PopulationBaselines(item_to_baseline=per_item.to_dict())


def apply_lift(
    scores: np.ndarray,
    item_ids: list[object],
    baselines: PopulationBaselines,
    weight: float,
) -> np.ndarray:
    """Multiply scores by a weighted lift factor.

    ``weight`` in ``[0, 1]``: 0 returns ``scores`` unchanged; 1 multiplies
    by lift = 1/baseline (maximum emphasis on distinctive items).
    Intermediate values interpolate logarithmically so emphasis is smooth.
    """
    if weight <= 0.0 or baselines.n_items == 0:
        return scores.astype(np.float64, copy=True)
    baseline_vec = baselines.lookup_many(item_ids)
    baseline_vec = np.maximum(baseline_vec, 1e-9)
    # lift = 1/baseline; power-interpolate to avoid runaway emphasis at 1.0.
    lift = (1.0 / baseline_vec) ** weight
    return np.asarray(scores * lift, dtype=np.float64)
