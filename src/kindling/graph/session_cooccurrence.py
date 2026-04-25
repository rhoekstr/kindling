"""Session-row item-item cooccurrence graph.

Same structural shape as ``ItemGraph`` but the user-item bipartite
matrix's row dimension is **session_id** instead of **entity_id**:

    S[session_id, item_id] = max(_interaction_weight) within session
    adjacency = S.T @ S      # n_items x n_items

So ``adjacency[i, j]`` = number of sessions containing both items
``i`` and ``j``. The classical "items often bought together" signal,
distinct from:

- ``ItemGraph``: rows are entities (users). Captures lifetime
  co-touching by the same user, regardless of when.
- ``TemporalInteractionGraph``: rows are entities again, but pair
  contributions are kernel-weighted by inter-event time.

Session-cooc fills the niche of "explicit basket co-membership" -
distinct from per-user history aggregation. Useful when sessions
are real (grocery / instacart / dunnhumby / tafeng) and explicit
or cleanly inferred from timestamps.

**Deep-session gate**: this is only built when at least
``min_deep_session_fraction`` of sessions contain 2+ items (default
0.3). Below that threshold, most sessions are singletons and
``S.T @ S`` reduces to a sparse degenerate of the entity cooc -
not worth the column. The builder returns None and the engine
proceeds without the signal (the column stays zero in the matrix
and the Bayesian posterior learns weight -> 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class SessionCooccurrenceGraph:
    """Sparse item-item cooccurrence weighted by session co-membership.

    Attributes
    ----------
    adjacency:
        Symmetric CSR (n_items, n_items). adjacency[i, j] = number of
        sessions containing both items i and j (weight-aggregated when
        the preprocessor's _interaction_weight column is present).
    item_ids:
        Internal-index -> external item_id mapping (mirrors ItemGraph).
    item_index:
        item_id -> internal-index mapping.
    n_sessions:
        Distinct sessions whose data fed at least one row of S.
    median_session_size:
        Median (item_count) across sessions in S. Diagnostic; surfaces
        why the deep-session gate triggered or didn't.
    deep_session_fraction:
        Fraction of sessions with at least 2 unique items. Reported
        in the engine's posterior_summary so users see the activation
        decision.
    """

    adjacency: sparse.csr_matrix
    item_ids: np.ndarray
    item_index: dict[object, int]
    n_sessions: int
    median_session_size: float
    deep_session_fraction: float

    @property
    def n_items(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def n_edges(self) -> int:
        """Stored non-zero entries (undirected edge count is n_edges/2)."""
        return int(self.adjacency.nnz)

    def score_against_owned(
        self,
        owned_indices: np.ndarray,
        exclude_indices: set[int] | None = None,
    ) -> np.ndarray:
        """Score each item by sum of session-cooc weight against the
        user's owned items. Mirrors ``_cooccurrence_signal``'s shape.

        Returns a length-n_items array, max-normalized to [0, 1].
        """
        n = self.n_items
        if owned_indices.size == 0 or n == 0:
            return np.zeros(n, dtype=np.float64)
        owned_indices = owned_indices[(owned_indices >= 0) & (owned_indices < n)]
        if owned_indices.size == 0:
            return np.zeros(n, dtype=np.float64)
        rows = self.adjacency[owned_indices]
        scores = np.asarray(rows.sum(axis=0)).ravel().astype(np.float64)
        if exclude_indices:
            for idx in exclude_indices:
                if 0 <= idx < n:
                    scores[idx] = 0.0
        max_s = float(scores.max())
        if max_s > 0:
            scores = scores / max_s
        return scores


def build_session_cooccurrence_graph(
    interactions: pd.DataFrame,
    item_index: dict[object, int],
    session_ids: np.ndarray,
    min_deep_session_fraction: float = 0.3,
    session_strategy: str | None = None,
    session_gap_seconds: float | None = None,
    min_session_gap_seconds: float = 300.0,
) -> SessionCooccurrenceGraph | None:
    """Build a session-row item-cooc graph from preprocessed interactions.

    Parameters
    ----------
    interactions:
        Validated interaction DataFrame. Must include ``entity_id`` and
        ``item_id``. ``_interaction_weight`` (preprocessor-attached) is
        used when present; otherwise falls back to weight=1.
    item_index:
        Engine ItemGraph's item_id -> internal-index mapping. Items not
        in this index are dropped (they're not in the served catalog).
    session_ids:
        Length-len(interactions) array of session ids aligned to the
        rows of ``interactions``. Typically produced by
        ``ingest.sessions.infer_sessions(interactions).session_ids``.
        When the engine has explicit ``session_id`` in the input, this
        is the input column verbatim.
    min_deep_session_fraction:
        Minimum fraction of sessions that must contain 2+ unique items
        for the build to proceed. Below this, most sessions are
        singletons (gowalla-style isolated check-ins, ml1m's per-row
        manual_fallback) and the resulting graph would degenerate.
        Default 0.3.
    session_strategy:
        ``ingest.sessions.SessionInference.strategy``: ``"explicit"`` |
        ``"gmm"`` | ``"manual_fallback"``. Used together with
        ``session_gap_seconds`` to detect rating-burst patterns where
        the inferred session is actually a UI click burst (ml1m: 87s
        midpoint, sessions are 50+ items in <2 min). When detected,
        we skip the build because session_cooc would amplify ratings-
        burst noise.
    session_gap_seconds:
        ``SessionInference.gap_threshold_seconds``. When the strategy
        is ``"gmm"`` and this value is below ``min_session_gap_seconds``,
        the dataset is rating-burst-shaped and we return None.
    min_session_gap_seconds:
        Threshold (default 300s = 5 min) below which an inferred GMM
        session boundary is interpreted as rating-burst rather than
        real consumption adjacency. Mirrors the
        ``temporal_interaction.calibrate_kernel`` logic.

    Returns
    -------
    SessionCooccurrenceGraph when build conditions hold, else None.
    """
    # Rating-burst guard: same logic as the temporal kernel's
    # rating_burst_detected branch. Sessions inferred with a sub-300s
    # gap are UI click bursts, not real consumption sessions.
    if (
        session_strategy == "gmm"
        and session_gap_seconds is not None
        and session_gap_seconds < min_session_gap_seconds
    ):
        return None
    from kindling.preprocess import weights_of

    n_items = max(item_index.values()) + 1 if item_index else 0
    if n_items == 0 or len(interactions) == 0:
        return None

    weights = weights_of(interactions)
    item_idx_array = np.asarray(
        [item_index.get(x, -1) for x in interactions["item_id"].to_numpy()],
        dtype=np.int64,
    )
    keep = (item_idx_array >= 0) & (weights > 0)
    if not keep.any():
        return None

    session_ids_kept = np.asarray(session_ids)[keep]
    item_idx_array = item_idx_array[keep]
    weights = weights[keep].astype(np.float32)

    # Group by (session_id, item_idx), max weight per pair.
    df = pd.DataFrame({
        "session_id": session_ids_kept,
        "item_idx": item_idx_array,
        "_w": weights,
    })
    agg = df.groupby(["session_id", "item_idx"], sort=False, as_index=False)["_w"].max()
    if agg.empty:
        return None

    # Diagnostic + deep-session gate: how many sessions have 2+ unique items?
    session_sizes = agg.groupby("session_id", sort=False).size()
    n_sessions = int(len(session_sizes))
    deep_fraction = float((session_sizes >= 2).mean()) if n_sessions > 0 else 0.0
    median_size = float(session_sizes.median()) if n_sessions > 0 else 0.0
    if deep_fraction < min_deep_session_fraction:
        return None

    # Build session-item bipartite matrix S.
    unique_sessions = np.unique(agg["session_id"].to_numpy())
    session_index = {s: i for i, s in enumerate(unique_sessions)}
    rows = agg["session_id"].map(session_index).to_numpy(dtype=np.int64)
    cols = agg["item_idx"].to_numpy(dtype=np.int64)
    data = agg["_w"].to_numpy(dtype=np.float32)

    bipartite = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(unique_sessions), n_items),
        dtype=np.float32,
    )
    adjacency = (bipartite.T @ bipartite).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    item_ids = np.empty(n_items, dtype=object)
    for item_id, idx in item_index.items():
        item_ids[idx] = item_id

    return SessionCooccurrenceGraph(
        adjacency=adjacency,
        item_ids=item_ids,
        item_index=dict(item_index),
        n_sessions=n_sessions,
        median_session_size=median_size,
        deep_session_fraction=deep_fraction,
    )
