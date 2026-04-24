"""Per-signal retrievers for the standalone-evaluation diagnostic
(ADR-signal-audit followup).

Each of the 10 signals in ``SIGNAL_ORDER`` gets a retriever that treats
*that signal's structure* as the sole source of candidates. We then
evaluate each retriever as a complete recommender (skip blend, skip
re-rank) to see what the signal actually knows.

These are intentionally independent of the Engine's current retriever
setup. They share the fitted state (``path_tree``, ``item_cosine``,
``als_factors``, ``persona_index``, ...) but produce their own
candidate sets and scores.

Cost signals are NEGATIVE and don't make sense as retrievers (you
wouldn't retrieve items to RECOMMEND based on "the entity will
probably hate them"), so they don't get standalone retrievers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from kindling.retrieve.protocol import Candidate

if TYPE_CHECKING:
    import scipy.sparse as sp

    from kindling.graph.als_factors import ALSFactors
    from kindling.graph.item_cosine import ItemCosineMatrix
    from kindling.graph.item_graph import ItemGraph
    from kindling.graph.lightgcn import LightGCNModel
    from kindling.path.basket_index import BasketIndex, BasketSimilarity
    from kindling.path.path_tree import PathTree
    from kindling.path.tail_index import TailIndex
    from kindling.personas.index import PersonaIndex


def _top_k_from_scores(
    item_ids: np.ndarray,
    scores: np.ndarray,
    exclude: set[object],
    budget: int,
    source: str,
) -> list[Candidate]:
    """Return up to ``budget`` Candidates with the highest scores,
    skipping items in ``exclude`` and any score <= 0."""
    if budget <= 0 or item_ids.size == 0:
        return []
    # argpartition top-(budget + len(exclude)) to account for filtering.
    k = min(len(item_ids), budget + len(exclude))
    if k <= 0:
        return []
    if k < len(scores):
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part])]
    else:
        order = np.argsort(-scores)
    out: list[Candidate] = []
    for idx in order:
        if len(out) >= budget:
            break
        item = item_ids[idx]
        if item in exclude:
            continue
        s = float(scores[idx])
        if s <= 0.0:
            continue
        out.append(Candidate(item_id=item, score=s, source=source))
    return out


@dataclass
class PathTailRetriever:
    """Retriever driven by the tail distribution of the last item.

    score(c) = P(c | last_item) from the TailIndex.
    """

    tail_index: "TailIndex"
    item_ids: np.ndarray
    name: str = "path_tail"
    budget_fraction: float = 1.0

    def retrieve(
        self,
        recent_history: tuple[object, ...],
        budget: int,
        exclude: set[object],
    ) -> list[Candidate]:
        if not recent_history or budget <= 0:
            return []
        last = recent_history[-1]
        scores = self.tail_index.score_many(self.item_ids.tolist(), last)
        return _top_k_from_scores(
            self.item_ids, np.asarray(scores, dtype=np.float64), exclude, budget, self.name
        )


@dataclass
class PathFullRetriever:
    """Retriever driven by the prefix-tree distribution.

    score(c) = P(c | longest-matching-prefix-of-history) from the PathTree.
    """

    path_tree: "PathTree"
    item_ids: np.ndarray
    name: str = "path_full"
    budget_fraction: float = 1.0

    def retrieve(
        self,
        recent_history: tuple[object, ...],
        budget: int,
        exclude: set[object],
    ) -> list[Candidate]:
        if not recent_history or budget <= 0:
            return []
        scores = self.path_tree.score_many(self.item_ids.tolist(), recent_history)
        return _top_k_from_scores(
            self.item_ids, np.asarray(scores, dtype=np.float64), exclude, budget, self.name
        )


@dataclass
class PathBasketRetriever:
    """Retriever driven by the basket-index score.

    score(c) = coverage-weighted aggregate over baskets similar to Q.
    Uses BasketIndex's native pair-index fast path when Q has >=2 items.
    """

    basket_index: "BasketIndex"
    item_ids: np.ndarray
    similarity: "BasketSimilarity | None" = None
    name: str = "path_basket"
    budget_fraction: float = 1.0

    def retrieve(
        self,
        query_basket: frozenset[object],
        budget: int,
        exclude: set[object],
    ) -> list[Candidate]:
        if budget <= 0 or not query_basket:
            return []
        from kindling.path.basket_index import BasketSimilarity

        sim = self.similarity if self.similarity is not None else BasketSimilarity.COVERAGE
        scores = self.basket_index.score_many(
            self.item_ids.tolist(), query_basket=query_basket, similarity=sim
        )
        return _top_k_from_scores(
            self.item_ids, np.asarray(scores, dtype=np.float64), exclude, budget, self.name
        )


@dataclass
class CosineRetriever:
    """Retriever driven by aggregate item-item cosine similarity.

    score(c) = sum over owned items j of cosine(c, j), normalized by
    |owned|. Uses the fitted ItemCosineMatrix.
    """

    cosine_matrix: "ItemCosineMatrix"
    item_graph: "ItemGraph"
    item_ids: np.ndarray
    name: str = "item_item_cosine"
    budget_fraction: float = 1.0

    def retrieve(
        self,
        owned_items: np.ndarray,
        budget: int,
        exclude: set[object] | None = None,
    ) -> list[Candidate]:
        if budget <= 0 or owned_items.size == 0:
            return []
        exclude = exclude if exclude is not None else set()
        owned_indices = np.asarray(
            [
                self.item_graph.item_index[o]
                for o in owned_items.tolist()
                if o in self.item_graph.item_index
            ],
            dtype=np.int64,
        )
        if owned_indices.size == 0:
            return []
        cand_indices = np.arange(len(self.item_ids), dtype=np.int64)
        scores = self.cosine_matrix.score_many(cand_indices, owned_indices)
        return _top_k_from_scores(
            self.item_ids, np.asarray(scores, dtype=np.float64), exclude, budget, self.name
        )


@dataclass
class ALSRetriever:
    """Retriever driven by ALS latent factors.

    score(c) = U_entity . V_c from the fitted ALSFactors.
    """

    als_factors: "ALSFactors"
    item_graph: "ItemGraph"
    item_ids: np.ndarray
    name: str = "als_factor"
    budget_fraction: float = 1.0

    def retrieve(
        self,
        entity_id: object,
        budget: int,
        exclude: set[object] | None = None,
    ) -> list[Candidate]:
        if budget <= 0:
            return []
        exclude = exclude if exclude is not None else set()
        cand_indices = np.arange(len(self.item_ids), dtype=np.int64)
        scores = self.als_factors.score_many(entity_id, cand_indices)
        return _top_k_from_scores(
            self.item_ids, np.asarray(scores, dtype=np.float64), exclude, budget, self.name
        )


@dataclass
class PersonaRetriever:
    """Retriever driven by persona-cluster taste match.

    score(c) = sum over personas P of match(entity, P) * persona_weight(c, P)
    using the fitted PersonaIndex. Items with no presence in any matched
    persona get no candidate entry.
    """

    persona_index: "PersonaIndex"
    item_ids: np.ndarray
    name: str = "persona"
    budget_fraction: float = 1.0

    def retrieve(
        self,
        entity_id: object,
        owned_items: np.ndarray,
        history: tuple[object, ...],
        budget: int,
        exclude: set[object] | None = None,
    ) -> list[Candidate]:
        if budget <= 0 or self.persona_index.n_personas == 0:
            return []
        exclude = exclude if exclude is not None else set()
        from kindling.personas.matching import build_user_query_vector, match_user

        user_vec = build_user_query_vector(
            owned_items=owned_items, history_items=history, index=self.persona_index
        )
        matches = match_user(user_vec, self.persona_index)
        if not matches.any():
            return []
        # persona_weight(c, P) = persona_vectors[P, c] + cold_start_weight * cold_start_weights[P, c]
        # Combine at retrieve time so a new item with persona cold-start
        # signal shows up as a candidate even when persona_vectors is
        # empty for it.
        item_scores = np.asarray(self.persona_index.persona_vectors.T @ matches).ravel()
        if self.persona_index.cold_start_weights is not None and self.persona_index.cold_start_weight > 0.0:
            cs_scores = np.asarray(self.persona_index.cold_start_weights.T @ matches).ravel()
            item_scores = item_scores + self.persona_index.cold_start_weight * cs_scores
        return _top_k_from_scores(
            self.item_ids, item_scores, exclude, budget, self.name
        )


@dataclass
class LightGCNRetriever:
    """Retriever driven by LightGCN's graph-smoothed latent factors.

    score(c) = U_entity . V_c from the fitted LightGCNModel. Mirrors
    ALSRetriever but uses graph-propagated embeddings so the retrieved
    candidates can differ from ALS's (different objective, different
    structural bias, potentially non-overlapping top-K).
    """

    lightgcn: "LightGCNModel"
    item_graph: "ItemGraph"
    item_ids: np.ndarray
    name: str = "lightgcn"
    budget_fraction: float = 1.0

    def retrieve(
        self,
        entity_id: object,
        budget: int,
        exclude: set[object] | None = None,
    ) -> list[Candidate]:
        if budget <= 0:
            return []
        exclude = exclude if exclude is not None else set()
        cand_indices = np.arange(len(self.item_ids), dtype=np.int64)
        scores = self.lightgcn.score_many(entity_id, cand_indices)
        return _top_k_from_scores(
            self.item_ids, np.asarray(scores, dtype=np.float64), exclude, budget, self.name
        )
