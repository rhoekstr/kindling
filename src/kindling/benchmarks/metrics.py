"""Recommender evaluation metrics.

These metric primitives and the ``aggregate`` reducer now live in
``kindling.blend.layer_scoring`` — a production module — so that the
fit-time layered calibrator (invoked from ``Engine``) can use them
without the engine transitively importing the benchmarks package.

This module re-exports them unchanged for backward compatibility with the
benchmark harness and the ``bench/`` scripts. The dependency direction is
deliberate: ``kindling.benchmarks`` re-imports from ``kindling.blend``,
never the reverse.

Precision@K, Recall@K, NDCG@K, MRR, Hit Rate, Coverage, intra-list
diversity. All metrics operate on per-entity recommendation lists vs.
per-entity ground-truth relevant sets, aggregated across entities via
arithmetic mean; coverage and diversity are aggregated differently and
documented per-metric in the source module.
"""

from __future__ import annotations

from kindling.blend.layer_scoring import (
    MetricReport,
    aggregate,
    hit,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "MetricReport",
    "aggregate",
    "hit",
    "intra_list_diversity",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
