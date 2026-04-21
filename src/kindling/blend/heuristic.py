"""Heuristic blend (Phase 2).

Signals live in two blocks:

1. **Path family**: ``path_full``, ``path_tail``, ``path_basket``. Gram-
   Schmidt-decorrelated against each other in that fixed order (PRD §6.2).
   After decorrelation the columns are on a standardized scale.
2. **Other**: ``cooccurrence`` (Phase 2), later similarity / graph topology.
   Pass through unchanged.

Because the two blocks live on different scales (z-scores vs raw counts),
blending happens on a rank-aggregated scale: each column is rank-normalized
within the current candidate pool, then combined by a weighted sum. This is
scale- and sparsity-invariant.

Weights are fixed by domain intuition; Phase 3 replaces them with Bayesian
posterior weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kindling.blend.decorrelate import DecorrelationBasis

PATH_FAMILY: tuple[str, ...] = ("path_full", "path_tail", "path_basket")

# Phase 2 defaults tuned on ML-1M. Path signals contribute a modest +2% on
# that dataset because ML-1M is ratings (not sessions) and therefore a weak
# exercise of the path family. Phase 3's Bayesian posterior will replace
# these with data-adaptive weights; on session-heavy datasets (Instacart,
# RetailRocket) the path weights are expected to rise substantially.
DEFAULT_WEIGHTS: dict[str, float] = {
    "path_full": 0.10,
    "path_tail": 0.20,
    "path_basket": 0.00,
    "cooccurrence": 0.70,
}


@dataclass(frozen=True)
class SignalFeatures:
    """Signal matrix for a candidate list.

    ``matrix`` shape: ``(N_candidates, K_signals)``. ``signal_names`` gives
    the column order. Path family columns must come first in that order.
    """

    matrix: np.ndarray
    signal_names: tuple[str, ...]


@dataclass
class HeuristicBlend:
    """Block-decorrelated, rank-normalized weighted sum."""

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    path_basis: DecorrelationBasis | None = None

    def score(self, features: SignalFeatures) -> np.ndarray:
        matrix = features.matrix
        n, k = matrix.shape
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        processed = matrix.astype(np.float64, copy=True)

        # Decorrelate the path family block in place.
        if self.path_basis is not None:
            path_indices = [
                features.signal_names.index(name)
                for name in self.path_basis.signal_names
                if name in features.signal_names
            ]
            if len(path_indices) == len(self.path_basis.signal_names):
                block = processed[:, path_indices]
                processed[:, path_indices] = self.path_basis.apply(block)

        # Max-normalize each column to [0, 1] preserving within-column
        # gradients (unlike rank normalization, which collapses them to
        # equally-spaced ranks and loses the signal that "the top item is
        # meaningfully better than #2").
        normed = _max_normalize(processed)

        w = np.zeros(k, dtype=np.float64)
        for i, name in enumerate(features.signal_names):
            w[i] = self.weights.get(name, 0.0)

        return normed @ w


def _max_normalize(matrix: np.ndarray) -> np.ndarray:
    """Per-column divide-by-max. All-zero columns stay zero. Negative
    standardized columns (from decorrelation) are shifted to non-negative
    first so the max normalization is meaningful."""
    out = matrix.astype(np.float64, copy=True)
    if out.size == 0:
        return out
    # Shift so the minimum of each column is 0 (handles negative decorrelation
    # outputs). A column with all-equal values becomes all-zero.
    col_min = out.min(axis=0)
    out = out - col_min
    col_max = out.max(axis=0)
    nontrivial = col_max > 1e-12
    out[:, nontrivial] = out[:, nontrivial] / col_max[nontrivial]
    out[:, ~nontrivial] = 0.0
    return np.asarray(out, dtype=np.float64)
