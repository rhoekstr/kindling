"""Item-item cosine similarity matrix as an 8th kindling signal.

Best-in-class offline baseline for session-poor data is item-item
cosine kNN (Sarwar et al. 2001). Growth-curve experiments on ML-1M
show kindling's Bayesian blend tracks cosine within 3% - cosine is
essentially what the blend is rediscovering. Including cosine as an
explicit signal lets the Bayesian posterior decide how much extra
weight to give it beyond the raw cooccurrence-count signal kindling
already carries.

Built once at fit time; scored at recommend time by summing cosine
similarity over the entity's owned items.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class ItemCosineMatrix:
    """Sparse cosine matrix with top-K per row pruning for memory."""

    matrix: sp.csr_matrix  # (n_items, n_items), 0 diagonal, top-K per row
    n_items: int

    def score_many(
        self,
        candidate_indices: np.ndarray,
        owned_indices: np.ndarray,
    ) -> np.ndarray:
        """Return per-candidate cosine-kNN scores.

        Score(c) = sum_{j in owned} cos(c, j). Normalized by max so the
        signal lives in [0, 1] alongside the other blend features.
        """
        n = candidate_indices.size
        if n == 0 or owned_indices.size == 0:
            return np.zeros(n, dtype=np.float64)
        # Build a sparse owned-indicator, then score = (matrix @ owned_vec)[candidates]
        owned_vec = np.zeros(self.n_items, dtype=np.float64)
        owned_vec[owned_indices] = 1.0
        full_scores = np.asarray(self.matrix @ owned_vec).ravel()
        scores = full_scores[candidate_indices]
        max_s = float(scores.max()) if scores.size else 0.0
        if max_s > 0:
            scores = scores / max_s
        return scores


def build_item_cosine_matrix(
    cooccurrence: sp.csr_matrix,
    item_counts: np.ndarray,
    top_k: int = 200,
    min_cosine: float = 0.01,
) -> ItemCosineMatrix:
    """Construct a kNN-pruned item-item cosine matrix.

    Parameters
    ----------
    cooccurrence:
        (n_items, n_items) sparse count matrix. Entry [i, j] is the
        number of entities that interacted with both i and j.
    item_counts:
        (n_items,) array of per-item entity counts (i.e., diagonal of
        the user-item Gram that the caller typically doesn't store).
    top_k:
        Keep only the top-k cosine values per row. Dense cosine would
        be n_items^2; top-K keeps the signal compact and the scoring
        linear in n_owned.
    min_cosine:
        Drop entries below this cosine value. Prevents long noise
        tails in well-mixed catalogs.
    """
    n = cooccurrence.shape[0]
    safe_counts = np.maximum(item_counts.astype(np.float64), 1.0)
    inv_sqrt = 1.0 / np.sqrt(safe_counts)
    diag = sp.diags(inv_sqrt)
    cos = diag @ cooccurrence @ diag  # (n, n), cosine
    cos = cos.tocsr()
    # Strip diagonal.
    cos.setdiag(0.0)
    cos.eliminate_zeros()
    # Threshold on absolute cosine.
    if min_cosine > 0 and cos.nnz:
        cos.data[cos.data < min_cosine] = 0.0
        cos.eliminate_zeros()
    # Top-K per row.
    if top_k > 0 and cos.nnz:
        cos = _top_k_per_row(cos, top_k)
    return ItemCosineMatrix(matrix=cos, n_items=n)


def _top_k_per_row(mat: sp.csr_matrix, k: int) -> sp.csr_matrix:
    rows, cols, data = [], [], []
    indptr = mat.indptr
    indices = mat.indices
    values = mat.data
    for i in range(mat.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if end - start <= k:
            rows.extend([i] * (end - start))
            cols.extend(indices[start:end].tolist())
            data.extend(values[start:end].tolist())
            continue
        row_data = values[start:end]
        picks = np.argpartition(-row_data, k)[:k]
        rows.extend([i] * k)
        cols.extend(indices[start:end][picks].tolist())
        data.extend(row_data[picks].tolist())
    return sp.csr_matrix((data, (rows, cols)), shape=mat.shape, dtype=mat.dtype)
