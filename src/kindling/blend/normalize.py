"""Per-query signal-column normalization for the linear blend.

The Bayesian blend computes ``score(c) = Σ w_k · feature_k(c)``. When
signal columns live on wildly different raw scales (cooccurrence in
the thousands, cosine in [0, 1]), the linear combination is dominated
by whichever signal has the largest raw magnitude - regardless of
posterior weight. Dead-in-the-blend behavior documented repeatedly in
the ADRs (signal-audit, retriever-union, LightGCN).

This module runs between ``_compute_signal_features`` and the blend's
scoring call. It reshapes every column onto a comparable scale so a
signal's posterior weight actually controls its contribution.

Four modes:
- ``zscore`` (default): ``(x - mean(x)) / std(x)`` per column. Allows
  negative contributions; dead signals (all zero) pass through
  unchanged (std guard).
- ``minmax``: ``(x - min) / (max - min)`` per column. Preserves the
  [0, 1] convention most signals already use.
- ``softmax``: ``exp(x/T) / sum`` per column with temperature T=1.0.
  Bounded, smooth, preserves ordering, squashes outliers.
- ``none``: no-op. Preserves the pre-normalization architecture for
  A/B comparison.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

NormalizeMode = Literal["zscore", "minmax", "softmax", "none"]


def normalize_columns(
    matrix: np.ndarray,
    mode: NormalizeMode = "zscore",
    softmax_temperature: float = 1.0,
) -> np.ndarray:
    """Return a copy of ``matrix`` with each column normalized per the mode.

    Columns that are all-zero (or all-identical) pass through unchanged
    rather than producing NaN / division-by-zero. Signals that produce
    no information for a query should stay silent, not get amplified.
    """
    if mode == "none" or matrix.size == 0:
        return matrix

    out = matrix.astype(np.float64, copy=True)
    n_rows, n_cols = out.shape

    if mode == "zscore":
        for c in range(n_cols):
            col = out[:, c]
            mean = col.mean()
            std = col.std()
            if std <= 1e-12:
                out[:, c] = 0.0
                continue
            out[:, c] = (col - mean) / std
        return out

    if mode == "minmax":
        for c in range(n_cols):
            col = out[:, c]
            lo = col.min()
            hi = col.max()
            rng = hi - lo
            if rng <= 1e-12:
                out[:, c] = 0.0
                continue
            out[:, c] = (col - lo) / rng
        return out

    if mode == "softmax":
        for c in range(n_cols):
            col = out[:, c]
            if col.max() - col.min() <= 1e-12:
                out[:, c] = 1.0 / max(n_rows, 1)
                continue
            scaled = col / softmax_temperature
            scaled -= scaled.max()  # numerical stability
            exp = np.exp(scaled)
            out[:, c] = exp / exp.sum()
        return out

    raise ValueError(f"unknown normalization mode: {mode!r}")
