"""Ranker protocol — the Stage 2 pluggable interface (PRD §6.3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from kindling.retrieve.protocol import Candidate


@runtime_checkable
class RankerProtocol(Protocol):
    """A ranker scores retrieval candidates.

    Phase 1 ships only the heuristic ranker. The LightGBM/XGBoost/CatBoost
    implementations from PRD §6.3 arrive in a later phase; the protocol is
    defined now so the engine can be written against it.
    """

    name: str

    def score(
        self,
        candidates: list[Candidate],
        owned_items: np.ndarray,
    ) -> np.ndarray:
        """Return a score per candidate. Order matches the input list."""
        ...
