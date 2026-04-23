"""Cold-start affinity for new items (PRD supplement §2.7).

Stub in commit 1 — full implementation lands in commit 3. The API is
fixed here so Engine wiring can reference it without churn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kindling.personas.index import PersonaIndex


@dataclass(frozen=True)
class ColdStartAffinity:
    """Per-persona overperformance ratio for a single new item.

    affinity:
        Raw per-persona affinity score, aggregated across the item's
        early interactions.
    overperformance:
        affinity / expected_affinity (base-rate-normalized per
        user-base-updated supplement §2.7).
    """

    affinity: np.ndarray  # (n_personas,)
    overperformance: np.ndarray  # (n_personas,)


def compute_cold_start_affinity(
    early_interaction_users: list[object],
    index: PersonaIndex,
) -> ColdStartAffinity:
    """Placeholder — real implementation lands in commit 3."""
    n = index.n_personas
    return ColdStartAffinity(
        affinity=np.zeros(n, dtype=np.float64),
        overperformance=np.zeros(n, dtype=np.float64),
    )
