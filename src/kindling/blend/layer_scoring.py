"""Evaluation metrics: ``MetricReport`` + the per-list metric functions
(precision / recall / NDCG / MRR / hit / diversity) and ``aggregate``.

``benchmarks.metrics`` re-exports these for the verification harness
(``bench/verify.py``) and the gap-decomposition diagnostic. (The v1
layer-scoring signal helpers that once lived here were removed with the
v1 engine in the production consolidation.)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Evaluation metrics
#
# Per-entity recommendation lists vs. per-entity ground-truth relevant sets,
# aggregated across entities via arithmetic mean (coverage/diversity
# documented per-metric). Used by the benchmark harness *and* by the
# layered calibrator's held-out grid sweep.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricReport:
    """Aggregate metrics from a benchmark run."""

    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    hit_rate: float
    coverage: float
    intra_list_diversity: float
    n_entities_evaluated: int
    k: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr": self.mrr,
            "hit_rate": self.hit_rate,
            "coverage": self.coverage,
            "intra_list_diversity": self.intra_list_diversity,
            "n_entities_evaluated": self.n_entities_evaluated,
            "k": self.k,
        }


def precision_at_k(recs: list[object], relevant: set[object], k: int) -> float:
    if k <= 0:
        return 0.0
    top = recs[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if r in relevant)
    return hits / k


def recall_at_k(recs: list[object], relevant: set[object], k: int) -> float:
    if not relevant:
        return 0.0
    top = recs[:k]
    hits = sum(1 for r in top if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(recs: list[object], relevant: set[object], k: int) -> float:
    """Binary-relevance NDCG at k."""
    if not relevant or k <= 0:
        return 0.0
    top = recs[:k]
    gains = np.array([1.0 if r in relevant else 0.0 for r in top])
    if gains.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
    dcg = float((gains * discounts).sum())
    ideal_hits = min(len(relevant), k)
    ideal = float(np.sum(1.0 / np.log2(np.arange(2, ideal_hits + 2))))
    return dcg / ideal


def reciprocal_rank(recs: list[object], relevant: set[object]) -> float:
    for i, r in enumerate(recs, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def hit(recs: list[object], relevant: set[object], k: int) -> float:
    return 1.0 if any(r in relevant for r in recs[:k]) else 0.0


def intra_list_diversity(
    rec_items: list[object],
    similarity_fn: Callable[[object, object], float] | None = None,
) -> float:
    """Mean pairwise dissimilarity within a single list.

    Phase 1 uses a trivial "identity" similarity (0 for distinct items, 1 for
    same), so this effectively measures uniqueness — which is 1.0 when the
    list contains no duplicates. Phase 4 replaces this with cosine similarity
    over SBERT or co-occurrence-derived features.
    """
    n = len(rec_items)
    if n < 2:
        return 0.0
    if similarity_fn is None:
        return 1.0 if len(set(rec_items)) == n else float(len(set(rec_items)) / n)
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - similarity_fn(rec_items[i], rec_items[j])
            pairs += 1
    return total / pairs if pairs else 0.0


def aggregate(
    per_entity: list[tuple[list[object], set[object]]],
    catalog_size: int,
    k: int = 10,
) -> MetricReport:
    """Aggregate metrics over all evaluated entities.

    Each element of ``per_entity`` is ``(rec_items, relevant_items)``.
    Entities with an empty relevant set are skipped from accuracy metrics
    but still contribute to coverage.
    """
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    rrs: list[float] = []
    hits: list[float] = []
    diversity: list[float] = []
    recommended_items: set[object] = set()

    for recs, relevant in per_entity:
        recommended_items.update(recs[:k])
        if not relevant:
            continue
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        rrs.append(reciprocal_rank(recs[:k], relevant))
        hits.append(hit(recs, relevant, k))
        diversity.append(intra_list_diversity(recs[:k]))

    coverage = len(recommended_items) / catalog_size if catalog_size > 0 else 0.0

    return MetricReport(
        precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else 0.0,
        mrr=float(np.mean(rrs)) if rrs else 0.0,
        hit_rate=float(np.mean(hits)) if hits else 0.0,
        coverage=coverage,
        intra_list_diversity=float(np.mean(diversity)) if diversity else 0.0,
        n_entities_evaluated=len(precisions),
        k=k,
    )
