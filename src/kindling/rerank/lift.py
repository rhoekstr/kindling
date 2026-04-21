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
    """Count the fraction of unique entities that interacted with each item."""
    if interactions.empty:
        return PopulationBaselines()
    pairs = interactions[["entity_id", "item_id"]].drop_duplicates()
    entity_count = pairs["entity_id"].nunique()
    if entity_count == 0:
        return PopulationBaselines()
    per_item = pairs.groupby("item_id").size() / entity_count
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
