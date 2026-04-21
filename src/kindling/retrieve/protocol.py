"""Retriever protocol — the pluggable Stage 1 interface (PRD §5.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Candidate:
    """A single retrieval candidate.

    Carries the item id, the retrieval-stage score, and the source retriever
    name for provenance in debug output.
    """

    item_id: object
    score: float
    source: str


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Pluggable retriever interface.

    Implementations take an entity's owned-item set and return a scored
    candidate list. The Engine handles union, deduplication (max score
    across retrievers), and budget enforcement.
    """

    name: str
    budget_fraction: float

    def retrieve(
        self,
        owned_items: np.ndarray,
        budget: int,
    ) -> list[Candidate]:
        """Return up to ``budget`` candidates for the given owned set."""
        ...
