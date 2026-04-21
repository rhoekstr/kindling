"""Co-occurrence neighbor retriever (PRD §5.2).

For each item in the entity's owned set, fetches the highest-weight
neighbors in the item graph. Scores are the max edge weight across the
owned set (a candidate that co-occurs with many owned items inherits the
strongest of those ties). Items already in the owned set are excluded.
"""

from __future__ import annotations

import numpy as np

from kindling._native import NATIVE_AVAILABLE, kindling_native
from kindling.graph.item_graph import ItemGraph
from kindling.retrieve.protocol import Candidate


class CoOccurrenceRetriever:
    """Return items that co-occur strongly with the entity's owned set.

    Routes through ``kindling_native.cooccurrence_retrieve`` when the
    Rust extension is present - that path folds the CSR row-sum,
    owned-item exclusion, and top-k selection into a single pass.
    Pure-Python fallback keeps the reference behavior.
    """

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

        if NATIVE_AVAILABLE and kindling_native is not None:
            adj = self.graph.adjacency
            indices, scores = kindling_native.cooccurrence_retrieve(
                adj.data.astype(np.float32, copy=False),
                adj.indices.astype(np.int32, copy=False),
                adj.indptr.astype(np.int32, copy=False),
                owned_indices,
                int(budget),
            )
            return [
                Candidate(
                    item_id=self.graph.item_ids[i],
                    score=float(s),
                    source=self.name,
                )
                for i, s in zip(indices, scores, strict=True)
            ]

        # Sum the rows of owned items - each column is the total
        # co-occurrence of that candidate across the owned set.
        summed = self.graph.adjacency[owned_indices].sum(axis=0)
        scores = np.asarray(summed).ravel()

        # Exclude the owned items themselves.
        scores[owned_indices] = 0.0

        # Deterministic ordering: positives first, sort descending by
        # score with ties broken on idx ascending. Matches the Rust
        # path exactly for differential testing.
        positives = np.where(scores > 0.0)[0]
        if positives.size == 0:
            return []
        order = np.lexsort((positives, -scores[positives]))
        ranked = positives[order][:budget]

        return [
            Candidate(
                item_id=self.graph.item_ids[i],
                score=float(scores[i]),
                source=self.name,
            )
            for i in ranked
        ]
