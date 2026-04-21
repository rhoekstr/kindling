"""Public Engine — the user-facing class wiring the three stages.

Phase 1 deliberately ships a minimal pipeline: a single co-occurrence
retriever, the heuristic ranker (pass-through score), and constraint
filtering as the only rerank op. The purpose is an end-to-end, testable,
benchmarked skeleton — not good recommendations yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kindling.explain import Explanation
from kindling.explain.templates import explain_from_source
from kindling.graph.item_graph import ItemGraph, build_item_graph
from kindling.ingest.contract import (
    InteractionSchema,
    canonicalize,
    validate_interactions,
)
from kindling.rank.heuristic import HeuristicRanker
from kindling.rank.protocol import RankerProtocol
from kindling.rerank.constraints import ConstraintPredicate, apply_constraints
from kindling.retrieve.cooccurrence import CoOccurrenceRetriever
from kindling.retrieve.protocol import Candidate, RetrieverProtocol

DEFAULT_RETRIEVAL_BUDGET = 500


@dataclass(frozen=True)
class Recommendation:
    """A single recommendation — the Phase 1 output shape.

    Intervals and per-signal debug payloads are placeholder today; the real
    Bayesian credible intervals arrive in Phase 3.
    """

    item_id: object
    score: float
    explanation: Explanation


class EngineNotFittedError(RuntimeError):
    pass


class Engine:
    """The primary kindling entry point.

    Minimal Phase 1 API:

    >>> engine = Engine()
    >>> engine.fit(interactions_df)
    >>> recs = engine.recommend(entity_id=42, n=10)
    """

    def __init__(
        self,
        retrieval_budget: int = DEFAULT_RETRIEVAL_BUDGET,
        ranker: RankerProtocol | None = None,
    ) -> None:
        self.retrieval_budget = retrieval_budget
        self._ranker: RankerProtocol = ranker if ranker is not None else HeuristicRanker()
        self._schema: InteractionSchema | None = None
        self._interactions: pd.DataFrame | None = None
        self._item_graph: ItemGraph | None = None
        self._retrievers: list[RetrieverProtocol] = []
        self._owned_by_entity: dict[object, np.ndarray] = {}

    # ---- fitting -----------------------------------------------------------

    def fit(self, interactions: pd.DataFrame) -> Engine:
        """Validate, canonicalize, and build derived structures."""
        schema = validate_interactions(interactions)
        self._schema = schema
        self._interactions = canonicalize(interactions, schema)
        self._item_graph = build_item_graph(self._interactions)
        self._retrievers = [CoOccurrenceRetriever(self._item_graph)]
        self._owned_by_entity = {
            entity: group["item_id"].to_numpy()
            for entity, group in self._interactions.groupby("entity_id", sort=False)
        }
        return self

    # ---- introspection (PRD §10.2 power-user surface) ----------------------

    @property
    def item_graph(self) -> ItemGraph:
        self._require_fitted()
        assert self._item_graph is not None
        return self._item_graph

    @property
    def schema(self) -> InteractionSchema:
        self._require_fitted()
        assert self._schema is not None
        return self._schema

    def data_density(self) -> dict[str, float | int]:
        self._require_fitted()
        assert self._interactions is not None
        assert self._item_graph is not None
        n_items = self._item_graph.n_items
        n_entities = self._interactions["entity_id"].nunique()
        n_interactions = len(self._interactions)
        max_edges = max(n_items * (n_items - 1), 1)
        return {
            "n_items": n_items,
            "n_entities": n_entities,
            "n_interactions": n_interactions,
            "graph_density": self._item_graph.n_edges / max_edges,
        }

    # ---- recommending ------------------------------------------------------

    def recommend(
        self,
        entity_id: object,
        n: int = 10,
        constraints: list[ConstraintPredicate] | None = None,
    ) -> list[Recommendation]:
        """Return up to ``n`` recommendations for the given entity."""
        self._require_fitted()
        owned = self._owned_by_entity.get(entity_id, np.array([]))

        # Stage 1: retrieve
        raw_candidates: list[Candidate] = []
        for retriever in self._retrievers:
            raw_candidates.extend(retriever.retrieve(owned, self.retrieval_budget))
        candidates = _dedup_max_score(raw_candidates, self.retrieval_budget)

        # Constraints apply before ranking (plan departure from PRD §7.6).
        if constraints:
            candidates = apply_constraints(candidates, constraints)

        # Stage 2: rank
        if not candidates:
            return []
        scores = self._ranker.score(candidates, owned)
        order = np.argsort(-scores)

        # Stage 3: rerank — Phase 1 is top-N slice
        top = order[:n]

        return [
            Recommendation(
                item_id=candidates[i].item_id,
                score=float(scores[i]),
                explanation=explain_from_source(candidates[i].source, float(scores[i])),
            )
            for i in top
        ]

    # ---- internals ---------------------------------------------------------

    def _require_fitted(self) -> None:
        if self._interactions is None:
            raise EngineNotFittedError("Engine.fit must be called before use")


def _dedup_max_score(candidates: list[Candidate], budget: int) -> list[Candidate]:
    """Merge candidates across retrievers, keeping the max score per item.

    When an item appears in multiple retrievers' results, its final retrieval
    score is the maximum. Source of the winning candidate is preserved.
    """
    if not candidates:
        return []
    best: dict[object, Candidate] = {}
    for c in candidates:
        existing = best.get(c.item_id)
        if existing is None or c.score > existing.score:
            best[c.item_id] = c
    deduped = sorted(best.values(), key=lambda c: -c.score)
    return deduped[:budget]
