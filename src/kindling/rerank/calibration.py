"""Steck 2018 calibrated recommendation re-rank (PRD §7.5).

Matches the category distribution of the recommendation list to the
entity's historical category distribution. Prevents the system from
over-representing the entity's dominant category while still honoring
their taste profile.

Requires categorical metadata. Disabled by default (``calibration_weight=0``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CategoryIndex:
    """Per-item category and per-entity category distribution.

    Phase 4 supports a single categorical column. Multi-column calibration
    (e.g., both genre and decade for movies) is a v1.x extension.
    """

    item_to_category: dict[object, str]
    entity_to_distribution: dict[object, dict[str, float]] = field(default_factory=dict)

    @property
    def categories(self) -> list[str]:
        return sorted(set(self.item_to_category.values()))

    def item_category_vec(self, item_ids: list[object]) -> list[str | None]:
        return [self.item_to_category.get(i) for i in item_ids]

    def entity_distribution(self, entity: object) -> dict[str, float]:
        return self.entity_to_distribution.get(entity, {})


def build_category_index(
    interactions: pd.DataFrame,
    item_metadata: pd.DataFrame,
    category_column: str,
) -> CategoryIndex | None:
    """Build the category index from item metadata and per-entity interactions.

    Returns ``None`` if ``category_column`` is missing from ``item_metadata``.
    """
    if item_metadata is None or category_column not in item_metadata.columns:
        return None
    if "item_id" not in item_metadata.columns:
        raise ValueError("item_metadata must contain 'item_id' column")
    itm = item_metadata[["item_id", category_column]].dropna()
    item_to_cat = {row["item_id"]: str(row[category_column]) for _, row in itm.iterrows()}
    if not item_to_cat:
        return None

    # Per-entity distribution over owned-item categories.
    merged = interactions.merge(itm, on="item_id", how="inner")
    if merged.empty:
        return CategoryIndex(item_to_category=item_to_cat)
    entity_dists: dict[object, dict[str, float]] = {}
    for entity, grp in merged.groupby("entity_id", sort=False):
        counts = grp[category_column].value_counts(normalize=True)
        entity_dists[entity] = {str(k): float(v) for k, v in counts.items()}
    return CategoryIndex(
        item_to_category=item_to_cat,
        entity_to_distribution=entity_dists,
    )


def apply_calibration(
    ordered_indices: list[int],
    item_ids: list[object],
    scores: np.ndarray,
    entity_id: object,
    index: CategoryIndex,
    weight: float,
    k: int,
) -> list[int]:
    """Greedy Steck re-rank.

    For each target slot, pick the candidate that maximizes
    ``(1 - weight) * score + weight * (-KL(target || current list dist))``
    where ``target`` is the entity's historical category distribution and
    ``current list dist`` is the running category histogram of the
    already-selected list.

    Returns a new ordered index list of length ``k`` drawn from
    ``ordered_indices``.
    """
    if weight <= 0.0 or not ordered_indices:
        return ordered_indices[:k]
    target = index.entity_distribution(entity_id)
    if not target:
        return ordered_indices[:k]

    pool_cats = [index.item_to_category.get(item_ids[i]) for i in ordered_indices]
    cats = sorted(set(c for c in pool_cats if c is not None) | set(target.keys()))
    if not cats:
        return ordered_indices[:k]
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    target_vec = np.asarray([target.get(c, 0.0) for c in cats], dtype=np.float64)
    target_vec = target_vec / max(target_vec.sum(), 1e-12)

    score_by_pool_idx = {
        pool_pos: float(scores[original]) for pool_pos, original in enumerate(ordered_indices)
    }
    max_score = max(score_by_pool_idx.values()) if score_by_pool_idx else 1.0
    norm = max(abs(max_score), 1e-9)

    selected: list[int] = []
    running = np.zeros(len(cats), dtype=np.float64)
    available = list(range(len(ordered_indices)))
    for _ in range(min(k, len(ordered_indices))):
        best_pool: int | None = None
        best_score = -np.inf
        for pool_pos in available:
            orig = ordered_indices[pool_pos]
            cat = index.item_to_category.get(item_ids[orig])
            test_running = running.copy()
            if cat is not None and cat in cat_to_idx:
                test_running[cat_to_idx[cat]] += 1.0
            test_dist = test_running / max(test_running.sum(), 1e-12)
            kl = float(_kl_divergence(target_vec, test_dist))
            norm_score = float(scores[orig]) / norm
            combined = (1.0 - weight) * norm_score - weight * kl
            if combined > best_score:
                best_score = combined
                best_pool = pool_pos
        if best_pool is None:
            break
        orig = ordered_indices[best_pool]
        cat = index.item_to_category.get(item_ids[orig])
        if cat is not None and cat in cat_to_idx:
            running[cat_to_idx[cat]] += 1.0
        selected.append(orig)
        available.remove(best_pool)
    return selected


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return float((p * (np.log(p) - np.log(q))).sum())
