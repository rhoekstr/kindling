"""interaction_neighborhood signal: communities + pluggable centrality.

For each query:

1. Find which communities the user's recent items belong to (Louvain
   computed once at fit time on the temporal interaction graph).
2. Score communities by representation strength against the user's
   recent items, weighted by recency.
3. Take the top-N communities (default N=2) by representation score.
4. Build a subgraph from the union of the top-N communities.
5. Compute a centrality measure on that subgraph.
6. Surface candidates the user hasn't interacted with, ranked by
   their centrality scores within the subgraph.

The centrality measure is **pluggable** as the proposal flagged
betweenness as the riskiest design choice. Five variants ship from
day one so the empirical question can be answered without a refactor:

- ``betweenness`` (proposal default): structural-bridge items.
- ``pagerank``: link-propagated importance, less degree-correlated than
  eigenvector but closer to it.
- ``eigenvector``: classical "popular among popular" measure.
- ``degree``: in-community popularity. Cheap sanity baseline.
- ``closeness``: avg-shortest-path centrality. Position-based.

For Probe-B we run all five on the same subgraphs and report whichever
delivers — and which dataset shapes prefer which measure.

Cost guards:
- Cap union-subgraph size at ``max_subgraph_nodes`` (default 2000).
  Above that, fall back to top-1 community only.
- Cache centrality scores per (community-tuple) so repeat queries are
  instant.
- Communities themselves stored as a dict {item_idx: community_id}.

Reference:
- Blondel et al. (2008), Louvain.
- Newman (2010), Networks: An Introduction (centrality measures).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.sparse as sp

from kindling.retrieve.protocol import Candidate

if TYPE_CHECKING:
    from kindling.graph.temporal_interaction import TemporalInteractionGraph


CentralityKind = Literal["betweenness", "pagerank", "eigenvector", "degree", "closeness"]
ALL_CENTRALITIES: tuple[CentralityKind, ...] = (
    "betweenness", "pagerank", "eigenvector", "degree", "closeness"
)


@dataclass
class InteractionNeighborhoodConfig:
    centrality: CentralityKind = "betweenness"
    top_n_communities: int = 2
    max_subgraph_nodes: int = 2000
    seed_window: int = 50
    min_community_match_score: float = 1e-6
    # Louvain knobs.
    louvain_resolution: float = 1.0
    louvain_seed: int = 0


@dataclass(frozen=True)
class InteractionNeighborhoodModel:
    """Precomputed state for interaction_neighborhood scoring.

    Attributes
    ----------
    adjacency:
        Symmetric CSR (n_items, n_items) — the temporal graph's edges.
    item_index_to_community:
        Length-n_items array. ``[-1]`` for items in no community
        (isolated or community below min size).
    community_members:
        Dict {community_id: np.ndarray of item indices}. Members
        sorted ascending so subgraph slicing is deterministic.
    item_ids:
        Internal-index -> external item_id mapping (mirrors temporal graph).
    item_index:
        item_id -> internal-index mapping.
    config:
        How communities were detected and how queries should select them.
    centrality_cache:
        Mutable cache: ``(frozenset community ids, centrality kind) ->
        np.ndarray of length n_items`` with centrality scores for items
        in the subgraph, zero elsewhere. Built lazily on first query
        per (community-tuple, centrality) pair.
    """

    adjacency: sp.csr_matrix
    item_index_to_community: np.ndarray
    community_members: dict[int, np.ndarray]
    item_ids: np.ndarray
    item_index: dict[object, int]
    config: InteractionNeighborhoodConfig
    centrality_cache: dict = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def n_communities(self) -> int:
        return len(self.community_members)

    def _select_top_communities(
        self,
        seed_indices: np.ndarray,
    ) -> list[int]:
        """Score each community by how many of the user's seeds fall in it
        (weighted by recency: later positions weight higher)."""
        if seed_indices.size == 0:
            return []
        n_seeds = seed_indices.size
        weights = np.linspace(0.5, 1.0, n_seeds)
        scores: dict[int, float] = {}
        for w, idx in zip(weights, seed_indices, strict=True):
            cid = int(self.item_index_to_community[idx])
            if cid < 0:
                continue
            scores[cid] = scores.get(cid, 0.0) + float(w)
        if not scores:
            return []
        # Sort by score desc, take top N.
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        ordered = [
            (cid, s) for cid, s in ordered
            if s >= self.config.min_community_match_score
        ]
        return [cid for cid, _ in ordered[: self.config.top_n_communities]]

    def _subgraph_centrality(
        self,
        community_ids: list[int],
        centrality: CentralityKind,
    ) -> np.ndarray:
        """Compute centrality on the subgraph induced by ``community_ids``.

        Returns a length-n_items array; entries outside the subgraph
        are zero. Cached across queries.
        """
        cache_key = (tuple(sorted(community_ids)), centrality)
        cached = self.centrality_cache.get(cache_key)
        if cached is not None:
            return cached

        # Union of community member indices.
        member_lists = [self.community_members[cid] for cid in community_ids]
        if not member_lists:
            scores = np.zeros(self.n_items, dtype=np.float64)
            self.centrality_cache[cache_key] = scores
            return scores
        nodes = np.unique(np.concatenate(member_lists))
        if nodes.size > self.config.max_subgraph_nodes and len(community_ids) > 1:
            # Drop to top-1 community when union explodes.
            biggest = max(community_ids, key=lambda c: self.community_members[c].size)
            nodes = self.community_members[biggest]
            cache_key = ((biggest,), centrality)
            cached = self.centrality_cache.get(cache_key)
            if cached is not None:
                return cached

        if nodes.size == 0:
            scores = np.zeros(self.n_items, dtype=np.float64)
            self.centrality_cache[cache_key] = scores
            return scores

        sub_adj = self.adjacency[nodes][:, nodes]
        sub_scores = _compute_centrality(sub_adj, centrality)
        # Map back to full item index.
        full = np.zeros(self.n_items, dtype=np.float64)
        full[nodes] = sub_scores
        self.centrality_cache[cache_key] = full
        return full

    def retrieve(
        self,
        entity_id: object,
        owned_items: np.ndarray,
        history: tuple,
        budget: int,
        exclude: set[object] | None = None,
        centrality_override: CentralityKind | None = None,
    ) -> list[Candidate]:
        cfg = self.config
        centrality = centrality_override or cfg.centrality
        if history:
            seed_pool = list(history[-cfg.seed_window :])
        else:
            seed_pool = list(owned_items.tolist()) if owned_items.size else []
        if not seed_pool:
            return []

        seed_indices = np.fromiter(
            (self.item_index.get(s, -1) for s in seed_pool),
            dtype=np.int64,
            count=len(seed_pool),
        )
        seed_indices = seed_indices[seed_indices >= 0]
        if seed_indices.size == 0:
            return []

        community_ids = self._select_top_communities(seed_indices)
        if not community_ids:
            return []

        scores = self._subgraph_centrality(community_ids, centrality).copy()
        # Exclude owned + caller-specified items.
        if owned_items.size:
            for it in owned_items.tolist():
                idx = self.item_index.get(it, -1)
                if idx >= 0:
                    scores[idx] = 0.0
        if exclude:
            for it in exclude:
                idx = self.item_index.get(it, -1)
                if idx >= 0:
                    scores[idx] = 0.0
        if scores.max() <= 0.0:
            return []
        scores = scores / scores.max()

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
                source="interaction_neighborhood",
            )
            for i in top_idx
        ]


# ----------- centrality dispatch -----------


def _compute_centrality(
    sub_adj: sp.csr_matrix,
    kind: CentralityKind,
) -> np.ndarray:
    """Compute centrality on a subgraph adjacency.

    Returns length-n_nodes scores in [0, 1] (max-normalized per call).
    Uses networkx for betweenness/eigenvector/closeness; rolls own for
    degree (cheap) and pagerank (scipy power method, cheap).
    """
    n = sub_adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    if kind == "degree":
        scores = np.asarray(sub_adj.sum(axis=1)).ravel()
    elif kind == "pagerank":
        scores = _pagerank_power_method(sub_adj)
    else:
        # networkx-backed centralities. Build only when needed because
        # graph construction is O(n_edges).
        import networkx as nx

        G = nx.from_scipy_sparse_array(sub_adj, edge_attribute="weight")
        if kind == "betweenness":
            d = nx.betweenness_centrality(G, weight="weight", normalized=True)
        elif kind == "eigenvector":
            try:
                d = nx.eigenvector_centrality_numpy(G, weight="weight")
            except (nx.NetworkXError, np.linalg.LinAlgError):
                # Fallback to power-iteration version which handles
                # disconnected components more gracefully.
                d = nx.eigenvector_centrality(G, max_iter=500, weight="weight")
        elif kind == "closeness":
            d = nx.closeness_centrality(G, distance="weight")
        else:
            raise ValueError(f"Unknown centrality kind: {kind}")
        scores = np.fromiter((d.get(i, 0.0) for i in range(n)), dtype=np.float64, count=n)

    max_s = float(scores.max())
    if max_s > 0:
        scores = scores / max_s
    return scores


def _pagerank_power_method(
    sub_adj: sp.csr_matrix,
    alpha: float = 0.85,
    tol: float = 1e-7,
    n_iter: int = 100,
) -> np.ndarray:
    """Plain (un-personalized) PageRank via power iteration on the
    column-normalized transition matrix. Reasonable for subgraphs of
    a few thousand nodes - networkx's pagerank is slower than this
    direct scipy version."""
    n = sub_adj.shape[0]
    col_sums = np.asarray(sub_adj.sum(axis=1)).ravel()
    nonzero = col_sums > 0
    inv = np.zeros_like(col_sums)
    inv[nonzero] = 1.0 / col_sums[nonzero]
    P = sp.diags(inv) @ sub_adj
    P = P.tocsr()
    teleport = np.ones(n, dtype=np.float64) / n
    r = teleport.copy()
    for _ in range(n_iter):
        r_next = (1 - alpha) * teleport + alpha * (P.T @ r)
        # Renormalize for dangling nodes (rows with no outgoing edges).
        dangling = r[~nonzero].sum() if (~nonzero).any() else 0.0
        r_next += alpha * dangling * teleport
        r_next /= r_next.sum() or 1.0
        if np.abs(r_next - r).sum() < tol:
            r = r_next
            break
        r = r_next
    return r


# ----------- builder -----------


def build_interaction_neighborhood(
    temporal_graph: "TemporalInteractionGraph",
    config: InteractionNeighborhoodConfig | None = None,
) -> InteractionNeighborhoodModel | None:
    """Run Louvain community detection and assemble the neighborhood model.

    Returns None if the graph has no edges or community detection
    produces no usable communities.
    """
    cfg = config or InteractionNeighborhoodConfig()
    adj = temporal_graph.adjacency
    if adj.nnz == 0:
        return None

    import networkx as nx

    G = nx.from_scipy_sparse_array(adj, edge_attribute="weight")
    # nx.community.louvain_communities returns list of frozensets.
    communities = nx.community.louvain_communities(
        G, weight="weight", resolution=cfg.louvain_resolution, seed=cfg.louvain_seed,
    )
    if not communities:
        return None

    n_items = adj.shape[0]
    item_to_comm = -np.ones(n_items, dtype=np.int64)
    members: dict[int, np.ndarray] = {}
    for cid, members_set in enumerate(communities):
        idx = np.fromiter(members_set, dtype=np.int64, count=len(members_set))
        idx.sort()
        members[cid] = idx
        item_to_comm[idx] = cid

    return InteractionNeighborhoodModel(
        adjacency=adj,
        item_index_to_community=item_to_comm,
        community_members=members,
        item_ids=temporal_graph.item_ids,
        item_index=temporal_graph.item_index,
        config=cfg,
        centrality_cache={},
    )
