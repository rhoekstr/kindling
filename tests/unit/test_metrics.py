"""Evaluation metric tests."""

from __future__ import annotations

import math

from kindling.benchmarks.metrics import (
    aggregate,
    hit,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_empty() -> None:
    assert precision_at_k([], {1, 2}, k=5) == 0.0


def test_precision_all_relevant() -> None:
    assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_precision_none_relevant() -> None:
    assert precision_at_k([1, 2, 3], {4, 5}, k=3) == 0.0


def test_recall_zero_when_no_relevant() -> None:
    assert recall_at_k([1, 2], set(), k=5) == 0.0


def test_recall_full_when_all_found() -> None:
    assert recall_at_k([1, 2, 3, 4], {1, 2}, k=4) == 1.0


def test_ndcg_perfect() -> None:
    # All relevant items in top positions
    assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0


def test_ndcg_zero_when_none_relevant_in_top() -> None:
    assert ndcg_at_k([1, 2, 3], {4, 5, 6}, k=3) == 0.0


def test_ndcg_position_matters() -> None:
    """Relevant item earlier should produce higher NDCG than later."""
    early = ndcg_at_k([1, 2, 3], {1}, k=3)
    late = ndcg_at_k([2, 3, 1], {1}, k=3)
    assert early > late


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([1, 2, 3], {2}) == 0.5
    assert reciprocal_rank([1, 2, 3], {1}) == 1.0
    assert reciprocal_rank([1, 2, 3], {9}) == 0.0


def test_hit_rate_binary() -> None:
    assert hit([1, 2, 3], {3}, k=3) == 1.0
    assert hit([1, 2, 3], {9}, k=3) == 0.0


def test_intra_list_diversity_unique() -> None:
    assert intra_list_diversity([1, 2, 3, 4]) == 1.0


def test_intra_list_diversity_duplicates() -> None:
    diversity = intra_list_diversity([1, 1, 2, 3])
    assert 0.0 < diversity < 1.0


def test_aggregate_empty_per_entity() -> None:
    report = aggregate([], catalog_size=100, k=10)
    assert report.ndcg_at_k == 0.0
    assert report.coverage == 0.0
    assert report.n_entities_evaluated == 0


def test_aggregate_coverage_counts_unique_recommended() -> None:
    per_entity = [
        ([1, 2, 3], {1}),
        ([2, 3, 4], {4}),
    ]
    # Recommended union is {1, 2, 3, 4} from a catalog of 10 items.
    report = aggregate(per_entity, catalog_size=10, k=3)
    assert math.isclose(report.coverage, 0.4)


def test_aggregate_skips_entities_with_no_relevant() -> None:
    per_entity = [
        ([1, 2, 3], {1}),
        ([4, 5, 6], set()),  # no relevant - skipped from accuracy metrics
    ]
    report = aggregate(per_entity, catalog_size=10, k=3)
    assert report.n_entities_evaluated == 1
