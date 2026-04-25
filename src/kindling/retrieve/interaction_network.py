"""interaction_network signal: PPR walks on the temporal interaction graph.

For a recommendation request, walks initiate from the user's recent
items (within a configurable lookback window). Personalized PageRank is
computed via the iterative power method on the temporal graph's
normalized adjacency: at each step, a random walker either moves to a
neighbor with probability proportional to edge weight, or restarts to
one of the user's recent items with probability ``alpha`` (default
0.15, PinSage / Pixie convention).

After convergence (or a fixed iteration cap), candidates are ranked by
stationary visit probability. Items the user has already interacted
with are filtered before final ranking.

Reuses the temporal substrate from ``graph/temporal_interaction.py``.
On no-timestamp datasets the substrate falls back to pure-count and
this signal essentially reduces to PPR-on-cooc - whether that's
distinctive value over the existing ``ppr.py`` retriever is the empirical
question Probe-A answers.

Reference:
- Page et al. (1999), "The PageRank Citation Ranking" - PPR formulation.
- Eksombatchai et al. (2018, Pinterest), "Pixie" - random walks with
  restart on a content graph for real-time recs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from kindling.retrieve.protocol import Candidate

if TYPE_CHECKING:
    from kindling.graph.temporal_interaction import TemporalInteractionGraph


@dataclass
class InteractionNetworkConfig:
    """Knobs for interaction_network.

    Attributes
    ----------
    alpha:
        Restart probability. 0.15 is the PinSage / Pixie default;
        higher values bias the walker toward staying near the seed,
        lower values explore further.
    n_iterations:
        Power-method iteration cap. PPR converges quickly on the
        normalized adjacency - 30 iterations is comfortably above the
        convergence threshold for typical recommender graphs.
    convergence_tol:
        Stop early if the L1 change in the rank vector falls below
        this between iterations.
    seed_window:
        Use only the most-recent ``seed_window`` of the user's history
        as PPR seed nodes. Older interactions are weak signal.
    min_seed_count:
        If the user has fewer than this many seeds, signal returns
        empty (cold-start fallback).
    """

    alpha: float = 0.15
    n_iterations: int = 30
    convergence_tol: float = 1e-6
    seed_window: int = 50
    min_seed_count: int = 1


@dataclass(frozen=True)
class InteractionNetworkModel:
    """Precomputed state for interaction_network scoring.

    The transition matrix is the row-normalized temporal adjacency
    (``P[i, j] = adjacency[i, j] / row_sum(adjacency[i])``). Computing
    this once at fit time avoids repeated normalization at query time.

    Item index alignment matches the temporal graph (and the engine's
    ItemGraph).
    """

    transition_matrix: sp.csr_matrix
    item_ids: np.ndarray
    item_index: dict[object, int]
    config: InteractionNetworkConfig
    nonzero_rows: np.ndarray  # rows with at least one outgoing edge

    @property
    def n_items(self) -> int:
        return int(self.transition_matrix.shape[0])

    def score_many(
        self,
        seed_indices: np.ndarray,
        exclude_indices: set[int] | None = None,
    ) -> np.ndarray:
        """Run PPR power iteration from ``seed_indices`` and return
        ``score[i]`` = stationary probability for each item.

        Returns a length-n_items array. Excluded items are zeroed
        before return so they don't surface as candidates.
        """
        n = self.n_items
        if seed_indices.size == 0 or n == 0:
            return np.zeros(n, dtype=np.float64)

        # Restart distribution: uniform over seeds.
        s = np.zeros(n, dtype=np.float64)
        seeds = np.unique(seed_indices[seed_indices >= 0])
        if seeds.size == 0:
            return np.zeros(n, dtype=np.float64)
        s[seeds] = 1.0 / seeds.size

        r = s.copy()
        alpha = self.config.alpha
        P = self.transition_matrix
        for _ in range(self.config.n_iterations):
            # PPR update: r' = (1 - alpha) * P^T @ r + alpha * s
            # P is row-stochastic; P^T propagates rank along incoming edges.
            r_next = (1.0 - alpha) * (P.T @ r) + alpha * s
            if np.abs(r_next - r).sum() < self.config.convergence_tol:
                r = r_next
                break
            r = r_next

        if exclude_indices:
            for idx in exclude_indices:
                if 0 <= idx < n:
                    r[idx] = 0.0
        return r

    def retrieve(
        self,
        entity_id: object,
        owned_items: np.ndarray,
        history: tuple,
        budget: int,
        exclude: set[object] | None = None,
    ) -> list[Candidate]:
        """Top-budget candidates by stationary PPR rank from the user's seeds.

        Parameters
        ----------
        owned_items:
            Items the user has already interacted with. Used as PPR
            seeds (recent interactions; ``seed_window`` limits the
            effective set) and as exclusion filter on the output.
        history:
            Ordered (oldest -> newest) interaction sequence. Used to
            select the most-recent ``seed_window`` items as seeds.
        """
        cfg = self.config
        # Prefer the temporal-ordered history when available; fall back to owned.
        if history:
            seed_pool = list(history[-cfg.seed_window :])
        else:
            seed_pool = list(owned_items.tolist()) if owned_items.size else []

        if len(seed_pool) < cfg.min_seed_count:
            return []

        seed_indices = np.fromiter(
            (self.item_index.get(s, -1) for s in seed_pool),
            dtype=np.int64,
            count=len(seed_pool),
        )
        seed_indices = seed_indices[seed_indices >= 0]
        if seed_indices.size == 0:
            return []

        # Build exclusion set of internal indices.
        exclude_set: set[int] = set()
        if owned_items.size:
            for it in owned_items.tolist():
                idx = self.item_index.get(it, -1)
                if idx >= 0:
                    exclude_set.add(int(idx))
        if exclude:
            for it in exclude:
                idx = self.item_index.get(it, -1)
                if idx >= 0:
                    exclude_set.add(int(idx))

        scores = self.score_many(seed_indices, exclude_indices=exclude_set)
        if scores.max() <= 0.0:
            return []

        # Normalize to [0, 1] per-query so the signal scale matches others.
        scores = scores / scores.max()

        # Top-budget by score.
        if budget < scores.size:
            top_idx = np.argpartition(-scores, budget)[:budget]
            top_idx = top_idx[scores[top_idx] > 0.0]
            order = np.argsort(-scores[top_idx])
            top_idx = top_idx[order]
        else:
            top_idx = np.argsort(-scores)
            top_idx = top_idx[scores[top_idx] > 0.0]

        return [
            Candidate(
                item_id=self.item_ids[i],
                score=float(scores[i]),
                source="interaction_network",
            )
            for i in top_idx
        ]


def build_interaction_network(
    temporal_graph: "TemporalInteractionGraph",
    config: InteractionNetworkConfig | None = None,
) -> InteractionNetworkModel | None:
    """Precompute the row-normalized transition matrix for PPR scoring.

    Returns None if the graph has no edges (signal can't activate).
    """
    cfg = config or InteractionNetworkConfig()
    adj = temporal_graph.adjacency
    if adj.nnz == 0:
        return None

    # Row-normalize: P[i, j] = adj[i, j] / sum(adj[i, :])
    row_sums = np.asarray(adj.sum(axis=1)).ravel()
    nonzero = row_sums > 0
    inv = np.zeros_like(row_sums)
    inv[nonzero] = 1.0 / row_sums[nonzero]
    P = sp.diags(inv) @ adj
    P = P.tocsr()

    return InteractionNetworkModel(
        transition_matrix=P,
        item_ids=temporal_graph.item_ids,
        item_index=temporal_graph.item_index,
        config=cfg,
        nonzero_rows=np.where(nonzero)[0],
    )


# --- per-candidate scoring helper for the engine's _compute_signal_features ---


def score_candidates_interaction_network(
    model: "InteractionNetworkModel",
    candidates: list[Candidate],
    seed_indices: np.ndarray,
    exclude_indices: set[int] | None = None,
) -> np.ndarray:
    """Score each candidate by its PPR stationary probability for the query.

    Returns a length-len(candidates) array on the same scale as
    score_many's output (normalized to [0, 1] per-query).
    """
    if not candidates or seed_indices.size == 0:
        return np.zeros(len(candidates), dtype=np.float64)
    full = model.score_many(seed_indices, exclude_indices=exclude_indices)
    if full.max() <= 0.0:
        return np.zeros(len(candidates), dtype=np.float64)
    full = full / full.max()
    cand_idx = np.fromiter(
        (model.item_index.get(c.item_id, -1) for c in candidates),
        dtype=np.int64,
        count=len(candidates),
    )
    valid = cand_idx >= 0
    out = np.zeros(len(candidates), dtype=np.float64)
    if valid.any():
        out[valid] = full[cand_idx[valid]]
    return out
