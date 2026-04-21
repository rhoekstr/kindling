"""Determinantal Point Processes for diversity re-ranking (PRD §7.2).

The DPP kernel factorizes as ``L[i,j] = alpha * quality_i * quality_j *
similarity(i,j)``. Greedy MAP inference picks items one at a time to
maximize the marginal gain in log-determinant of the selected principal
minor. Standard Cholesky-update trick keeps each step O(N * k) where
N is the candidate pool and k is the current list length.

Similarity is pluggable. Phase 4 ships ``CooccurrenceCosineKernel`` which
uses normalized item-graph rows as item feature vectors. Phase 6 adds an
SBERT-backed kernel when descriptive metadata is available.

The alpha parameter (``diversity_weight``) interpolates quality vs.
diversity. alpha=0 collapses to pure argmax (quality only); alpha=1 pushes
toward maximal diversity subject to quality tiebreaking. Default 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from scipy import sparse

from kindling.graph.item_graph import ItemGraph

DEFAULT_DIVERSITY_WEIGHT = 0.5


@runtime_checkable
class SimilarityKernel(Protocol):
    """Similarity over item ids. Must return values in ``[0, 1]``.

    Implementations must also handle unknown items (return 0). The kernel
    is called for every candidate pair during DPP greedy MAP, so
    implementations should be vectorized when possible.
    """

    name: str

    def pairwise(self, item_ids: list[object]) -> np.ndarray:
        """Return a symmetric ``(N, N)`` similarity matrix over the given
        item ids. Diagonal must be 1."""
        ...


@dataclass
class CooccurrenceCosineKernel:
    """Cosine similarity over co-occurrence row profiles.

    Two items that co-occur with similar entity subsets get high
    similarity. This is a reasonable default when descriptive metadata
    isn't available - the same information that powers co-occurrence
    retrieval also seeds diversity.
    """

    item_graph: ItemGraph
    name: str = "cooccurrence_cosine"

    def pairwise(self, item_ids: list[object]) -> np.ndarray:
        if self.item_graph.n_items == 0 or not item_ids:
            return np.eye(len(item_ids), dtype=np.float64)

        indices = [self.item_graph.item_index.get(i) for i in item_ids]
        rows: list[np.ndarray] = []
        for idx in indices:
            if idx is None:
                rows.append(np.zeros(self.item_graph.n_items, dtype=np.float32))
            else:
                rows.append(self.item_graph.adjacency.getrow(idx).toarray().ravel())
        mat = sparse.csr_matrix(np.vstack(rows))
        norms = sparse.linalg.norm(mat, axis=1)
        safe_norms = np.where(norms > 0, norms, 1.0)
        # Normalize rows; cosine similarity = normalized dot product.
        diag = sparse.diags(1.0 / safe_norms)
        normalized = diag @ mat
        sim = (normalized @ normalized.T).toarray().astype(np.float64)
        # Zero-row items stay at similarity 0 to everyone except themselves.
        np.fill_diagonal(sim, 1.0)
        return np.asarray(sim, dtype=np.float64)


@dataclass
class DPPGreedy:
    """Greedy MAP inference over the quality-diversity DPP kernel."""

    kernel: SimilarityKernel
    diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT
    _eps: float = field(default=1e-8, init=False)

    def rerank(
        self,
        item_ids: list[object],
        qualities: np.ndarray,
        k: int,
    ) -> list[int]:
        """Return the indices (into ``item_ids``) selected by greedy MAP,
        in selection order. ``qualities`` must have shape ``(len(item_ids),)``
        and be non-negative."""
        n = len(item_ids)
        if n == 0 or k <= 0:
            return []
        k = min(k, n)
        q = np.asarray(qualities, dtype=np.float64)
        # When diversity_weight == 0 the kernel reduces to quality^2 on the
        # diagonal with no cross terms; greedy MAP collapses to argsort
        # which we short-circuit.
        if self.diversity_weight <= 0.0:
            return list(np.argsort(-q)[:k])

        sim = self.kernel.pairwise(item_ids)
        # Build L: alpha * q_i * q_j * sim(i, j). We blend identity into sim
        # by (1 - alpha) + alpha * sim, which keeps L PSD and interpolates
        # cleanly between pure quality (alpha=0) and quality-weighted
        # similarity (alpha=1).
        alpha = float(self.diversity_weight)
        blended_sim = (1.0 - alpha) * np.eye(n) + alpha * sim
        outer_q = np.outer(q, q)
        kernel_L = outer_q * blended_sim  # noqa: N806

        # Greedy MAP with Cholesky updates.
        selected: list[int] = []
        d = np.diag(kernel_L).copy()
        # Per-step Cholesky column storage.
        c = np.zeros((n, k), dtype=np.float64)

        for step in range(k):
            # Candidate gain = log(d_i); pick argmax on the set of unselected.
            available_mask = np.ones(n, dtype=bool)
            available_mask[selected] = False
            if not available_mask.any():
                break
            masked = np.where(available_mask, d, -np.inf)
            pick = int(np.argmax(masked))
            if d[pick] <= self._eps:
                break
            selected.append(pick)
            if step + 1 == k:
                break
            # Update Cholesky column and diagonal.
            # c_pick, step = sqrt(d_pick)
            c[pick, step] = np.sqrt(d[pick])
            # For j != pick: c_{j, step} = (L[j, pick] - sum_{s<step} c_{j,s} c_{pick,s}) / c_{pick, step}
            prev_dot = c[:, :step] @ c[pick, :step]
            col = (kernel_L[:, pick] - prev_dot) / c[pick, step]
            # Only update unselected rows.
            mask = np.ones(n, dtype=bool)
            mask[selected] = False
            c[mask, step] = col[mask]
            # d_j -= c_{j, step}^2
            d[mask] -= col[mask] ** 2
            d = np.maximum(d, 0.0)
            d[selected] = -np.inf
        return selected
