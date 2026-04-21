"""Constraint filtering (PRD §7.6).

Plan departure from the PRD: constraints apply between retrieval and
ranking, not after ranking, so filtered items don't waste ranker compute.
Semantically identical for hard filters.
"""

from __future__ import annotations

from collections.abc import Callable

from kindling.retrieve.protocol import Candidate

ConstraintPredicate = Callable[[object], bool]
"""A predicate takes an item_id and returns whether the item passes."""


def apply_constraints(
    candidates: list[Candidate],
    predicates: list[ConstraintPredicate],
) -> list[Candidate]:
    """Return candidates that satisfy all predicates.

    Predicates evaluate in declaration order; first failure short-circuits
    that candidate.
    """
    if not predicates:
        return candidates
    return [c for c in candidates if all(p(c.item_id) for p in predicates)]
