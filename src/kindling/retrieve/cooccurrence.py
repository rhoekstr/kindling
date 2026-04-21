"""Co-occurrence neighbor retriever (PRD §5.2).

For each item in the entity's owned set, fetches the highest-weight
neighbors in the item graph. Scores are the max edge weight across the
owned set (a candidate that co-occurs with many owned items inherits the
strongest of those ties). Items already in the owned set are excluded.
"""

from __future__ import annotations

import numpy as np

from kindling.graph.item_graph import ItemGraph
from kindling.retrieve.protocol import Candidate


class CoOccurrenceRetriever:
    """Return items that co-occur strongly with the entity's owned set."""

    name = "cooccurrence"

    def __init__(self, graph: ItemGraph, budget_fraction: float = 1.0) -> None:
        self.graph = graph
        self.budget_fraction = budget_fraction

    def retrieve(self, owned_items: np.ndarray, budget: int) -> list[Candidate]:
        if self.graph.n_items == 0 or owned_items.size == 0 or budget <= 0:
            return []

        owned_indices: list[int] = []
        for item in owned_items:
            idx = self.graph.item_index.get(item)
            if idx is not None:
                owned_indices.append(idx)
        if not owned_indices:
            return []

        # Sum the rows of owned items — each column is the total
        # co-occurrence of that candidate across the owned set.
        summed = self.graph.adjacency[owned_indices].sum(axis=0)
        scores = np.asarray(summed).ravel()

        # Exclude the owned items themselves.
        scores[owned_indices] = 0.0

        if budget >= self.graph.n_items:
            ranked = np.argsort(-scores)
        else:
            top_k = min(budget, int((scores > 0).sum()))
            if top_k == 0:
                return []
            part = np.argpartition(-scores, top_k - 1)[:top_k]
            ranked = part[np.argsort(-scores[part])]

        return [
            Candidate(
                item_id=self.graph.item_ids[i],
                score=float(scores[i]),
                source=self.name,
            )
            for i in ranked
            if scores[i] > 0.0
        ][:budget]
