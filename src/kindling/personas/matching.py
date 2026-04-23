"""Online matching + candidate scoring (PRD supplement §2.4, §2.5).

Both stages are sparse-matrix linear algebra; numpy/scipy hand off to
BLAS for the inner loop, so Python orchestration is fine here. If
profiling later shows matching is a hot path, we port it to Rust
following the pattern in ``native/src/personas.rs``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from kindling.personas.index import PersonaIndex


def build_user_query_vector(
    owned_items: np.ndarray,
    history_items: tuple[object, ...],
    index: PersonaIndex,
) -> sp.csr_matrix:
    """Build the entity's TF-IDF-weighted interaction vector in item space.

    We use the union of ``owned_items`` and recent history as the
    entity's "footprint" for matching. Binary weighting (supplement
    §2.4 step 1 default) - TF-IDF weighting happens below.

    Returns a (1, n_items) sparse CSR row in the same basis as
    ``index.persona_vectors``.
    """
    item_ids: set[object] = set()
    if owned_items.size:
        item_ids.update(owned_items.tolist())
    item_ids.update(history_items)
    n_items = index.n_items
    cols: list[int] = []
    for item in item_ids:
        idx = index.item_id_to_idx.get(item)
        if idx is not None:
            cols.append(idx)
    if not cols:
        return sp.csr_matrix((1, n_items), dtype=np.float64)

    # Binary TF → log1p(1) = log(2). Multiply by stored idf.
    idf_subset = index.idf[cols]
    data = np.log1p(1.0) * idf_subset
    rows = np.zeros(len(cols), dtype=np.int64)
    vec = sp.csr_matrix((data, (rows, np.asarray(cols))), shape=(1, n_items))

    # L2 normalize so cosine becomes dot product.
    norm = float(np.sqrt(vec.multiply(vec).sum()))
    if norm > 0.0:
        vec = vec.multiply(1.0 / norm).tocsr()
    return vec


def match_user(
    user_vec: sp.csr_matrix,
    index: PersonaIndex,
) -> np.ndarray:
    """Cosine similarity between the user vector and each persona vector.

    Both sides are L2-normalized, so this is a sparse-sparse dot.
    Returns a dense (n_personas,) array in [0, 1].
    """
    if user_vec.nnz == 0 or index.n_personas == 0:
        return np.zeros(index.n_personas, dtype=np.float64)
    # (1, n_items) @ (n_items, n_personas)
    scores = user_vec @ index.persona_vectors.T
    dense = np.asarray(scores.todense()).ravel()
    return np.clip(dense, 0.0, 1.0)


def score_candidates(
    persona_match: np.ndarray,
    index: PersonaIndex,
    candidate_item_ids: list[object],
) -> np.ndarray:
    """Per-candidate persona score = Σ_P match(P) × persona_weight(c, P).

    With L2-normalized persona vectors, ``persona_weight(c, P)`` is the
    stored (TF-IDF, then row-normalized) value. Returns a (n_candidates,)
    array.
    """
    n = len(candidate_item_ids)
    if n == 0 or index.n_personas == 0:
        return np.zeros(n, dtype=np.float64)

    # Gather persona_vectors columns for the candidates.
    col_idx: list[int] = []
    slot: list[int] = []
    for i, item in enumerate(candidate_item_ids):
        j = index.item_id_to_idx.get(item)
        if j is not None:
            col_idx.append(j)
            slot.append(i)
    if not col_idx:
        return np.zeros(n, dtype=np.float64)

    # Sub-matrix: (n_personas, len(col_idx)) dense-friendly because
    # len(col_idx) is typically ~500 and n_personas ~30-100.
    sub = index.persona_vectors[:, col_idx]  # (n_personas, n_cols)
    # weighted sum by persona match: match @ sub → (n_cols,)
    scores_subset = np.asarray(sub.T @ persona_match).ravel()

    # Cold-start: add the overperformance contribution for items that
    # have it. Scaled by ``cold_start_weight`` so the supervised
    # persona vector stays dominant on well-represented items.
    if index.cold_start_weights is not None and index.cold_start_weight > 0.0:
        sub_cs = index.cold_start_weights[:, col_idx]
        cs_scores = np.asarray(sub_cs.T @ persona_match).ravel()
        scores_subset = scores_subset + index.cold_start_weight * cs_scores

    out = np.zeros(n, dtype=np.float64)
    out[slot] = scores_subset
    return out
