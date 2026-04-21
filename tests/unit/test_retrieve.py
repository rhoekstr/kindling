"""Retriever tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kindling.graph.item_graph import build_item_graph
from kindling.retrieve.cooccurrence import CoOccurrenceRetriever


def test_retriever_excludes_owned_items(tiny_interactions: pd.DataFrame) -> None:
    graph = build_item_graph(tiny_interactions)
    retriever = CoOccurrenceRetriever(graph)
    owned = np.array([1, 2])
    candidates = retriever.retrieve(owned, budget=10)
    returned_items = {c.item_id for c in candidates}
    assert 1 not in returned_items
    assert 2 not in returned_items


def test_retriever_returns_empty_for_empty_owned(tiny_interactions: pd.DataFrame) -> None:
    graph = build_item_graph(tiny_interactions)
    retriever = CoOccurrenceRetriever(graph)
    assert retriever.retrieve(np.array([]), budget=10) == []


def test_retriever_budget_respected(tiny_interactions: pd.DataFrame) -> None:
    graph = build_item_graph(tiny_interactions)
    retriever = CoOccurrenceRetriever(graph)
    owned = np.array([1])
    candidates = retriever.retrieve(owned, budget=2)
    assert len(candidates) <= 2


def test_retriever_returns_scores_sorted_descending(
    tiny_interactions: pd.DataFrame,
) -> None:
    graph = build_item_graph(tiny_interactions)
    retriever = CoOccurrenceRetriever(graph)
    candidates = retriever.retrieve(np.array([1, 2]), budget=10)
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_retriever_ignores_unknown_owned_items(tiny_interactions: pd.DataFrame) -> None:
    graph = build_item_graph(tiny_interactions)
    retriever = CoOccurrenceRetriever(graph)
    # Mix of known item 1 with unknown item "zzz"
    candidates = retriever.retrieve(np.array([1, "zzz"], dtype=object), budget=5)
    # Should still produce neighbors of item 1
    assert len(candidates) > 0


def test_retriever_source_attribution(tiny_interactions: pd.DataFrame) -> None:
    graph = build_item_graph(tiny_interactions)
    retriever = CoOccurrenceRetriever(graph)
    candidates = retriever.retrieve(np.array([1]), budget=10)
    assert all(c.source == "cooccurrence" for c in candidates)
