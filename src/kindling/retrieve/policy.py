"""Retriever policy: data-adaptive selection + budget allocation.

Given the engine's fitted state + a DataFeatures snapshot, build the
list of retrievers to run at query time. Each entry is (name, retriever,
per_retriever_budget, rrf_weight).

Data-adaptive gating (ADR-retriever-union.md):
- Cooccurrence retriever: always on.
- ALS factor retriever: on if ALS factors were fitted.
- Item-item cosine retriever: on if cosine matrix was fitted.
- Path-basket retriever: on when sessions are explicit OR session
  density suggests real basket structure.
- Persona retriever: on when persona index was fitted AND sessions
  are strong enough to make cluster assignment meaningful.
- path_endpoint (the old default): NOT included. Standalone eval shows
  it underperforms path_tail alone, and under RRF it actively hurts.

Per-retriever budgets sum to ``retrieval_budget``. RRF weights are
per-retriever contribution multipliers for the reciprocal rank fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from kindling.retrieve.cooccurrence import CoOccurrenceRetriever
from kindling.retrieve.signal_retrievers import (
    ALSRetriever,
    CosineRetriever,
    PathBasketRetriever,
    PersonaRetriever,
)

if TYPE_CHECKING:
    from kindling.blend.priors import DataFeatures
    from kindling.engine import Engine


@dataclass(frozen=True)
class RetrieverEntry:
    name: str
    retriever: object
    budget: int
    rrf_weight: float


def build_retriever_stack(
    engine: "Engine",
    features: "DataFeatures",
    retrieval_budget: int,
) -> list[RetrieverEntry]:
    """Assemble the data-adaptive retriever stack.

    Budgets + weights are picked empirically from the standalone-retriever
    measurements in ADR-standalone-retrievers.md:

    - cooc: consistently top-2 on both datasets; budget 0.35 × total,
      weight 1.0.
    - als:  highest recall@10 on both datasets; 0.25 × total, weight 1.0.
    - cosine: near-identical to cooc; 0.2 × total, weight 0.9 (slight
      discount for redundancy with cooc).
    - path_basket: strong on sessions (6× ml1m); 0.1 × total on session
      data, 0 on ratings. Weight 0.9 on session data.
    - persona: middle of the pack on sessions; 0.1 × total on session
      data, 0 on ratings. Weight 0.8.
    """
    session_strong = features.has_explicit_sessions
    item_ids = np.asarray(engine.item_graph.item_ids, dtype=object)

    # Define the full candidate list of (name, builder, base_share,
    # session_only, rrf_weight). Budgets are normalized after gating.
    candidates: list[tuple[str, object, float, bool, float]] = []

    # Cooccurrence (always on).
    candidates.append(
        ("cooccurrence", CoOccurrenceRetriever(engine._item_graph), 0.35, False, 1.0)
    )

    # ALS (if fitted).
    if engine._als_factors is not None:
        candidates.append(
            (
                "als_factor",
                ALSRetriever(engine._als_factors, engine._item_graph, item_ids),
                0.25,
                False,
                1.0,
            )
        )

    # Cosine (if fitted).
    if engine._item_cosine is not None:
        candidates.append(
            (
                "item_item_cosine",
                CosineRetriever(engine._item_cosine, engine._item_graph, item_ids),
                0.20,
                False,
                0.9,
            )
        )

    # Path-basket (session-gated).
    if engine._basket_index is not None:
        candidates.append(
            (
                "path_basket",
                PathBasketRetriever(engine._basket_index, item_ids),
                0.10,
                True,
                0.9,
            )
        )

    # Persona (session-gated + requires fitted index).
    if (
        engine._persona_index is not None
        and engine._persona_index.n_personas > 0
    ):
        candidates.append(
            (
                "persona",
                PersonaRetriever(engine._persona_index, item_ids),
                0.10,
                True,
                0.8,
            )
        )

    # Apply session gate.
    gated = [
        (name, r, share, weight)
        for (name, r, share, session_only, weight) in candidates
        if (not session_only) or session_strong
    ]

    # Normalize budget shares after gating + compute integer budgets.
    total_share = sum(share for _, _, share, _ in gated) or 1.0
    entries: list[RetrieverEntry] = []
    allocated = 0
    for name, r, share, weight in gated:
        budget = max(1, int(round(retrieval_budget * share / total_share)))
        allocated += budget
        entries.append(
            RetrieverEntry(name=name, retriever=r, budget=budget, rrf_weight=weight)
        )
    # Hand any rounding slack to the first entry (cooc).
    if entries and allocated != retrieval_budget:
        remaining = retrieval_budget - allocated
        entries[0] = RetrieverEntry(
            name=entries[0].name,
            retriever=entries[0].retriever,
            budget=max(1, entries[0].budget + remaining),
            rrf_weight=entries[0].rrf_weight,
        )
    return entries
