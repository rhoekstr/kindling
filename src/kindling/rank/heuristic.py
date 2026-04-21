"""Heuristic ranker — Phase 1.

Passes the retrieval-stage score through unchanged. This is deliberately
trivial; the Bayesian blend arrives in Phase 3. The purpose here is that
the Engine's contract (retrieve → rank → rerank) works end-to-end and the
benchmark harness produces numbers we can regress against.
"""

from __future__ import annotations

import numpy as np

from kindling.retrieve.protocol import Candidate


class HeuristicRanker:
    name = "heuristic"

    def score(
        self,
        candidates: list[Candidate],
        owned_items: np.ndarray,
    ) -> np.ndarray:
        if not candidates:
            return np.array([], dtype=np.float64)
        return np.array([c.score for c in candidates], dtype=np.float64)
