"""Item co-occurrence graph.

Phase 1: simple undirected co-occurrence counts, scipy CSR-backed for fast
row slicing at recommend time. No time decay, no action_type handling —
those land in Phase 2 and Phase 5.

Construction uses the standard bipartite-matrix trick: build a sparse
entity-item matrix ``U`` of shape (n_entities, n_items) with 1s where an
entity interacted with an item, then the item co-occurrence graph is
``U.T @ U`` with the diagonal zeroed. This is orders of magnitude faster
than iterating pairs explicitly and has well-bounded memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class ItemGraph:
    """Sparse item-item co-occurrence graph.

    Attributes
    ----------
    adjacency:
        CSR matrix with shape (n_items, n_items). adjacency[i, j] is the
        number of entities that interacted with both items i and j.
        Symmetric; diagonal is zero.
    item_ids:
        Array mapping internal index -> external item_id.
    item_index:
        Dict mapping external item_id -> internal index.
    """

    adjacency: sparse.csr_matrix
    item_ids: np.ndarray
    item_index: dict[object, int]

    @property
    def n_items(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def n_edges(self) -> int:
        """Count of stored non-zero directed entries. Undirected edge count
        is n_edges / 2 since the matrix is symmetric."""
        return int(self.adjacency.nnz)

    def prune_below(self, support_threshold: float) -> tuple[int, float]:
        """Drop edges whose weight is below ``support_threshold``. Returns
        ``(n_pruned_edges, total_pruned_weight)``. Mutates ``adjacency``
        in place via the underlying CSR data array (compatible with the
        frozen dataclass)."""
        if support_threshold <= 0.0 or self.adjacency.nnz == 0:
            return 0, 0.0
        data = self.adjacency.data
        mask = data < support_threshold
        n_pruned = int(mask.sum())
        if n_pruned == 0:
            return 0, 0.0
        pruned_weight = float(data[mask].sum())
        # Zero out pruned entries; eliminate_zeros compacts the CSR.
        data[mask] = 0.0
        self.adjacency.eliminate_zeros()
        return n_pruned, pruned_weight

    def neighbors(self, item_id: object, top_k: int | None = None) -> pd.DataFrame:
        """Return the top-k co-occurring neighbors of an item by weight.

        Returns a DataFrame with columns ``item_id`` and ``weight``.
        """
        if item_id not in self.item_index:
            return pd.DataFrame({"item_id": [], "weight": []})
        idx = self.item_index[item_id]
        row = self.adjacency.getrow(idx).toarray().ravel()
        if top_k is not None and top_k < self.n_items:
            top = np.argpartition(-row, top_k)[:top_k]
            top = top[np.argsort(-row[top])]
        else:
            top = np.argsort(-row)
        mask = row[top] > 0
        top = top[mask]
        return pd.DataFrame({"item_id": self.item_ids[top], "weight": row[top]})


def build_item_graph(interactions: pd.DataFrame) -> ItemGraph:
    """Build an item co-occurrence graph from validated interactions.

    Reads the per-row positive-preference weight from the preprocessor-
    attached ``_interaction_weight`` column (falls back to ones when the
    column is absent, which preserves the old binary-implicit behavior
    and keeps unit tests building raw graphs without preprocess working).

    For duplicate (entity, item) rows the builder keeps the MAX weight
    so a rating upgrade overrides a prior lower-weight row.
    """
    from kindling.preprocess import weights_of

    if interactions.empty:
        return _empty_graph()

    weights = weights_of(interactions)
    pairs = interactions[["entity_id", "item_id"]].copy()
    pairs["_w"] = weights
    # Max weight per (entity, item) - handles duplicates and rating upgrades.
    pairs = pairs.groupby(  # type: ignore[assignment]
        ["entity_id", "item_id"], sort=False, as_index=False
    )["_w"].max()
    # Drop pairs with zero weight - ratings below threshold contribute
    # nothing to positive signals (cost graph handles them as negatives).
    pairs = pairs[pairs["_w"] > 0.0]
    if pairs.empty:
        return _empty_graph()

    unique_items = np.sort(pairs["item_id"].unique())
    unique_entities = np.sort(pairs["entity_id"].unique())
    item_index = {item: idx for idx, item in enumerate(unique_items)}
    entity_index = {e: idx for idx, e in enumerate(unique_entities)}
    item_ids_array = np.asarray(unique_items)

    row = pairs["entity_id"].map(entity_index).to_numpy(dtype=np.int64)
    col = pairs["item_id"].map(item_index).to_numpy(dtype=np.int64)
    data = pairs["_w"].to_numpy(dtype=np.float32)

    # Bipartite user-item matrix U: entities x items. Values are the
    # positive-preference weights above; a binary dataset lands on data=1
    # and recovers the old behavior exactly.
    bipartite = sparse.csr_matrix(
        (data, (row, col)),
        shape=(len(unique_entities), len(unique_items)),
        dtype=np.float32,
    )
    # Co-occurrence = U^T U. The result is item-by-item where entry (i, j)
    # is the number of entities that hit both. The diagonal is each item's
    # entity-support count - we zero it out.
    adjacency = (bipartite.T @ bipartite).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return ItemGraph(
        adjacency=adjacency,
        item_ids=item_ids_array,
        item_index=item_index,
    )


def _empty_graph() -> ItemGraph:
    return ItemGraph(
        adjacency=sparse.csr_matrix((0, 0), dtype=np.float32),
        item_ids=np.array([]),
        item_index={},
    )
