"""Neighbor pooling for items with insufficient individual data.

For each item with fewer than ``min_observations_individual`` intervals,
pool its intervals with those of its K cooc-nearest items, then fit the
profile on the combined distribution.

The docstring on the design doc also mentions "blend individual + pooled
proportional to data volume." In v1 we keep it simple: if an item has
>= min, use its own data; else use pooled-only. The blending is a clean
follow-up once we have baseline measurements.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def neighbors_by_cooccurrence(
    item_graph_adj: sp.csr_matrix,
    item_idx: int,
    k: int,
) -> np.ndarray:
    """Return the indices of the top-K nearest items by raw cooccurrence.

    ``item_graph_adj`` is the engine's item-item adjacency (rows and
    cols are item indices). Self-edge is excluded.
    """
    row = item_graph_adj.getrow(item_idx).toarray().ravel()
    row[item_idx] = 0.0  # never neighbor of self
    if k >= row.size:
        order = np.argsort(-row)
        return order[row[order] > 0.0]
    part = np.argpartition(-row, k)[:k]
    part = part[np.argsort(-row[part])]
    return part[row[part] > 0.0]
