"""ALS latent-factor signal (9th kindling signal).

Implicit ALS (Hu-Koren-Volinsky 2008) factorizes the entity-item matrix
into low-rank factors and predicts scores via U @ V^T. Kindling's other
signals are neighborhood-based (paths, cosine, cost); ALS adds a latent
perspective that generalizes across items with no direct coocurrence -
this is where ALS owns coverage on sparse datasets (ADR-growth-curves
finding #5).

Optional: requires the 'implicit' package. Without it, the signal stays
zero and the blend proceeds as an 8-signal mix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class ALSFactors:
    """Fitted ALS factor matrices + index mappings."""

    item_factors: np.ndarray  # (n_items, k)
    entity_factors: np.ndarray  # (n_entities, k)
    entity_index: dict[object, int]
    item_index: dict[object, int]
    k: int

    def score_many(
        self,
        entity_id: object,
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        """Return ALS scores for the given candidate indices.

        Indices are external-item -> internal-index from the Engine's
        item_graph (the ALS index mirrors it). Unknown entities score
        zero to keep the signal well-behaved on cold-starts.
        """
        n = candidate_indices.size
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        entity_idx = self.entity_index.get(entity_id)
        if entity_idx is None:
            return np.zeros(n, dtype=np.float64)
        # Scores = item_factors[candidates] @ entity_factors[entity]
        entity_vec = self.entity_factors[entity_idx]
        item_vecs = self.item_factors[candidate_indices]
        scores = np.asarray(item_vecs @ entity_vec, dtype=np.float64)
        # Normalize to [0, 1] per-query so the signal lives on a
        # comparable scale to the others.
        max_s = float(scores.max()) if scores.size else 0.0
        if max_s > 0:
            scores = scores / max_s
            scores = np.maximum(scores, 0.0)
        else:
            scores = np.zeros(n, dtype=np.float64)
        return scores


class ALSNotAvailableError(ImportError):
    """Raised when ALSFactors.fit is called without ``implicit`` installed."""


def build_als_factors(
    interactions,  # type: ignore[no-untyped-def]
    item_graph_item_index: dict[object, int],
    factors: int = 32,
    regularization: float = 0.01,
    iterations: int = 10,
    random_state: int = 0,
) -> "ALSFactors | None":
    """Fit ALS on the entity-item matrix and return the factor tables.

    Returns None if ``implicit`` is not installed (engine gracefully
    continues with the 8-signal blend). The engine's item_graph index
    is passed in so the ALS column order matches - signal computation
    at query time uses the same internal item indices.
    """
    try:
        import os

        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        return None

    # Stable entity order (sorted) so the ALS entity_index is deterministic
    # under row-shuffle - required by test_row_order_doesnt_affect_recommendations.
    entities = sorted(interactions["entity_id"].unique(), key=str)
    entity_index = {e: i for i, e in enumerate(entities)}
    # Use the item_graph's ordering as the canonical item index so the
    # ALS columns align with the other signals at query time.
    n_items = len(item_graph_item_index)

    from kindling.preprocess import weights_of

    rows = interactions["entity_id"].map(entity_index).to_numpy()
    cols = interactions["item_id"].map(lambda x: item_graph_item_index.get(x, -1)).to_numpy()
    weights = weights_of(interactions)
    keep = cols >= 0
    rows = rows[keep]
    cols = cols[keep]
    data = weights[keep].astype(np.float32)
    ui = sp.csr_matrix(
        (data, (rows, cols)), shape=(len(entities), n_items), dtype=np.float32
    )
    ui.sum_duplicates()
    # Cap confidence at 1.0 per pair. Without rating (data=1), this preserves
    # the prior implicit-feedback behavior exactly. With rating, 4-5 star
    # pairs keep their elevated confidence while summed duplicates don't
    # inflate unboundedly.
    ui.data = np.minimum(ui.data, 1.0)
    # Drop zero-weight entries (ratings below threshold) so ALS doesn't
    # treat them as observed with confidence 0 (which would be a no-op
    # but wastes CSR slots).
    ui.eliminate_zeros()

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
        use_gpu=False,
    )
    model.fit(ui, show_progress=False)
    return ALSFactors(
        item_factors=np.asarray(model.item_factors, dtype=np.float64),
        entity_factors=np.asarray(model.user_factors, dtype=np.float64),
        entity_index=entity_index,
        item_index=dict(item_graph_item_index),
        k=factors,
    )
