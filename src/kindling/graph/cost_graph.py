"""Cost distance graph for negative signal (PRD §3.6).

Models negative interactions as traversal resistance rather than binary
blacklisting. Three layers compose into a single effective cost:

    effective_cost(item, entity, owned_set) =
        alpha_pop * population_cost(item)
        + entity_cost(item, entity)
        + context_cost(item, owned_set)

where:

- ``population_cost``: how often this item is rejected across the whole
  population. Globally-muted signal; default weight ``alpha_pop = 0.3``.
- ``entity_cost``: how often this specific entity has rejected this item.
  Strong local signal.
- ``context_cost``: how often this item is rejected when paired with
  similar owned sets. Set-conditional signal.

The design keeps the item graph (positive signal) and cost graph (negative
signal) as independent structures. This is the PRD's stated design
commitment: items become expensive to reach in traversal rather than
impossible, matching how humans treat preferences.

Phase 5 ships the three-layer structure. The context_cost layer uses a
basket-overlap keying approximation to keep storage bounded; exact context
matching lives in v1.x.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_ALPHA_POP = 0.3
# Negative action types the cost graph consumes. action_type values not in
# this set are ignored (they contribute to the positive item graph instead).
NEGATIVE_ACTIONS = frozenset({"remove", "negative_rating"})


@dataclass(frozen=True)
class CostGraph:
    """Three-layer cost store.

    Attributes
    ----------
    population_cost:
        ``population_cost[item]`` = rejection-count / total_entities.
        Zero for items never rejected. In ``[0, 1]``.
    entity_cost:
        ``entity_cost[(entity, item)]`` = rejection count of this item by
        this entity. Unbounded integer.
    context_cost:
        ``context_cost[(context_key, item)]`` = rejection count of this
        item given the context. ``context_key`` is a hashable summary of
        an owned set; Phase 5 uses ``frozenset`` of at-most 8 items drawn
        from the owned set to keep the key space bounded.
    alpha_pop:
        Weight on the population layer. Default 0.3 per PRD.
    """

    population_cost: dict[object, float] = field(default_factory=dict)
    entity_cost: dict[tuple[object, object], float] = field(default_factory=dict)
    context_cost: dict[tuple[frozenset[object], object], float] = field(default_factory=dict)
    alpha_pop: float = DEFAULT_ALPHA_POP

    @property
    def n_rejected_items(self) -> int:
        return len(self.population_cost)

    @property
    def n_entity_rejections(self) -> int:
        return len(self.entity_cost)

    def effective_cost(
        self,
        item: object,
        entity: object,
        owned_set: set[object] | frozenset[object] | None = None,
    ) -> float:
        """Composed effective cost for a single item. Zero when the item
        appears in no cost layer."""
        pop = self.population_cost.get(item, 0.0)
        ent = self.entity_cost.get((entity, item), 0.0)
        ctx = 0.0
        if owned_set:
            key = _context_key(owned_set)
            ctx = self.context_cost.get((key, item), 0.0)
        return float(self.alpha_pop * pop + ent + ctx)

    def effective_cost_many(
        self,
        items: list[object],
        entity: object,
        owned_set: set[object] | frozenset[object] | None = None,
    ) -> np.ndarray:
        """Vectorized effective-cost lookup over a candidate list."""
        key = _context_key(owned_set) if owned_set else None
        out = np.zeros(len(items), dtype=np.float64)
        for i, item in enumerate(items):
            pop = self.population_cost.get(item, 0.0)
            ent = self.entity_cost.get((entity, item), 0.0)
            ctx = self.context_cost.get((key, item), 0.0) if key is not None else 0.0
            out[i] = self.alpha_pop * pop + ent + ctx
        return out

    def population_costs_many(self, items: list[object]) -> np.ndarray:
        return np.asarray([self.population_cost.get(i, 0.0) for i in items], dtype=np.float64)

    def entity_costs_many(self, items: list[object], entity: object) -> np.ndarray:
        return np.asarray([self.entity_cost.get((entity, i), 0.0) for i in items], dtype=np.float64)

    def context_costs_many(
        self,
        items: list[object],
        owned_set: set[object] | frozenset[object] | None,
    ) -> np.ndarray:
        if not owned_set:
            return np.zeros(len(items), dtype=np.float64)
        key = _context_key(owned_set)
        return np.asarray([self.context_cost.get((key, i), 0.0) for i in items], dtype=np.float64)


def _context_key(owned_set: set[object] | frozenset[object]) -> frozenset[object]:
    """Bounded hashable summary of an owned set. Caps at 8 items sorted by
    their string repr to produce a deterministic key regardless of set-
    iteration order."""
    if not owned_set:
        return frozenset()
    items = sorted(owned_set, key=str)[:8]
    return frozenset(items)


def build_cost_graph(
    interactions: pd.DataFrame,
    alpha_pop: float = DEFAULT_ALPHA_POP,
) -> CostGraph:
    """Build the three-layer cost graph from interactions.

    Reads rows where ``action_type`` is in ``NEGATIVE_ACTIONS``. For
    ``rating`` columns, rows where ``rating`` is explicitly negative
    (below 2.5 on a 5-point scale or below 0 otherwise) also count when
    ``action_type`` is ``"rate"`` or ``"negative_rating"``.
    """
    if "action_type" not in interactions.columns:
        return CostGraph(alpha_pop=alpha_pop)

    negatives_mask = interactions["action_type"].isin(NEGATIVE_ACTIONS)
    if "rating" in interactions.columns:
        neg_rating_mask = interactions["action_type"].isin({"rate", "negative_rating"}) & (
            interactions["rating"] < 2.5
        )
        negatives_mask = negatives_mask | neg_rating_mask
    negatives = interactions[negatives_mask]
    if negatives.empty:
        return CostGraph(alpha_pop=alpha_pop)

    n_entities_total = int(interactions["entity_id"].nunique())
    if n_entities_total == 0:
        return CostGraph(alpha_pop=alpha_pop)

    # Population layer: fraction of entities rejecting each item.
    pop_counts = negatives.groupby("item_id")["entity_id"].nunique()
    population_cost = {item: float(count) / n_entities_total for item, count in pop_counts.items()}

    # Entity layer: per (entity, item) rejection count.
    entity_cost: dict[tuple[object, object], float] = defaultdict(float)
    for _, row in negatives.iterrows():
        entity_cost[(row["entity_id"], row["item_id"])] += 1.0

    # Context layer: for each negative event, key the entity's
    # already-owned positives at that point as the context. If we had
    # session structure and timestamps we could do this precisely; Phase 5
    # uses the simplified per-entity aggregate.
    context_cost: dict[tuple[frozenset[object], object], float] = defaultdict(float)
    if "timestamp" in interactions.columns:
        _populate_context_layer(interactions, negatives, context_cost)

    return CostGraph(
        population_cost=population_cost,
        entity_cost=dict(entity_cost),
        context_cost=dict(context_cost),
        alpha_pop=alpha_pop,
    )


def _populate_context_layer(
    interactions: pd.DataFrame,
    negatives: pd.DataFrame,
    context_cost: dict[tuple[frozenset[object], object], float],
) -> None:
    """For each negative event, compute the entity's already-owned positive
    items at that timestamp and key the context."""
    positives_mask = _positives_mask(interactions)
    positives = interactions[positives_mask].sort_values("timestamp", kind="mergesort")
    # Group positives by entity for O(log n) "items before timestamp" lookups.
    positives_by_entity: dict[object, list[tuple[pd.Timestamp, object]]] = {}
    for entity, grp in positives.groupby("entity_id", sort=False):
        positives_by_entity[entity] = list(
            zip(grp["timestamp"].tolist(), grp["item_id"].tolist(), strict=True)
        )
    for _, row in negatives.iterrows():
        entity = row["entity_id"]
        ts = row["timestamp"]
        item = row["item_id"]
        events = positives_by_entity.get(entity, [])
        # Owned at the moment of the negative event = positives with
        # timestamp <= ts. Binary search-able in principle; linear scan is
        # fine at Phase 5 scale.
        owned = {i for t, i in events if t <= ts and i != item}
        if owned:
            context_cost[(_context_key(owned), item)] += 1.0


def _positives_mask(interactions: pd.DataFrame) -> pd.Series:
    """Boolean mask of positive interactions (for context-layer keying)."""
    if "action_type" not in interactions.columns:
        return pd.Series([True] * len(interactions), index=interactions.index)
    pos_actions = {"add", "positive_rating", "rate", "view"}
    mask = interactions["action_type"].isin(pos_actions)
    if "rating" in interactions.columns:
        # A rate action at < 2.5 is negative; rate at >= 2.5 (or missing)
        # counts as positive.
        high_rating = interactions["rating"].isna() | (interactions["rating"] >= 2.5)
        mask = mask & ((interactions["action_type"] != "rate") | high_rating)
    return mask
