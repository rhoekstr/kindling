"""Signal decorrelation (PRD §6.2).

Per PRD §6.2 the decorrelation is applied **within the path family only**:
``full -> tail -> basket``. Non-path signals (cooccurrence, similarity,
graph topology) are a separate block that passes through unchanged. Cross-
block decorrelation is explicitly not performed because it conflates two
different scales of signal (sequence-order probabilities in [0, 1] vs.
raw co-occurrence counts in the thousands).

Plan gap closed: the held-out set used to estimate correlations is the
**last 10% chronological slice** of training interactions (by timestamp
when available). The basis is persisted with the engine and applied
verbatim at inference.

Scale hygiene: signals are z-standardized within the held-out sample before
Gram-Schmidt and the same per-signal mean/std is applied at inference. This
is not in the PRD but is necessary to produce stable projection
coefficients when signal magnitudes differ; without standardization the
coefficients explode on sparse signals (a lesson learned from a Phase 2
empirical run).

Classical Gram-Schmidt: given signal column vectors ``s_1, ..., s_K``,
produce orthogonal ``u_1, ..., u_K`` where ``u_k`` is ``s_k`` minus its
projection onto ``span(u_1, ..., u_{k-1})``. At inference, each candidate
produces a raw signal vector that is standardized (using fit-time mean/std)
and then run through the same triangular transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DecorrelationBasis:
    """Persisted decorrelation basis for a single signal block.

    Stores: the signal names in order, the per-signal mean and std from the
    fit sample (used to standardize at inference), and the Gram-Schmidt
    projection coefficients.

    Transform applied to a (N, K) signal block ``S``:
        Z = (S - mean) / std
        U[:, 0] = Z[:, 0]
        U[:, k] = Z[:, k] - sum_{j<k} coefficients[k][j] * U[:, j]

    Returns the decorrelated ``U`` matrix (still on standardized scale).
    """

    signal_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]

    @property
    def n_signals(self) -> int:
        return len(self.signal_names)

    def apply(self, signals: np.ndarray) -> np.ndarray:
        """Apply standardization + orthogonalization."""
        if signals.ndim != 2 or signals.shape[1] != self.n_signals:
            raise ValueError(f"signals must have shape (N, {self.n_signals}); got {signals.shape}")
        means = np.asarray(self.means, dtype=np.float64)
        stds = np.asarray(self.stds, dtype=np.float64)
        stds = np.where(stds > 0, stds, 1.0)
        z = (signals - means) / stds
        u = np.zeros_like(z, dtype=np.float64)
        for k in range(self.n_signals):
            u[:, k] = z[:, k]
            for j, coef in enumerate(self.coefficients[k]):
                u[:, k] -= coef * u[:, j]
        return u


def fit_decorrelation(
    signal_matrix: np.ndarray,
    signal_names: list[str] | tuple[str, ...],
) -> DecorrelationBasis:
    """Fit the basis on a (N, K) signal-score matrix. Columns are
    standardized in place, then classical Gram-Schmidt computes the
    projection coefficients that reproduce an orthogonal basis."""
    if signal_matrix.ndim != 2:
        raise ValueError(f"signal_matrix must be 2-D, got shape {signal_matrix.shape}")
    _, k = signal_matrix.shape
    if k != len(signal_names):
        raise ValueError(f"Got {k} columns but {len(signal_names)} names")

    means = signal_matrix.mean(axis=0)
    stds = signal_matrix.std(axis=0)
    safe_stds = np.where(stds > 0, stds, 1.0)
    z = (signal_matrix - means) / safe_stds

    u = np.zeros_like(z, dtype=np.float64)
    all_coefs: list[tuple[float, ...]] = []
    for i in range(k):
        u[:, i] = z[:, i]
        row: list[float] = []
        for j in range(i):
            denom = float((u[:, j] ** 2).sum())
            if denom <= 1e-12:
                row.append(0.0)
                continue
            num = float((z[:, i] * u[:, j]).sum())
            coef = num / denom
            row.append(coef)
            u[:, i] -= coef * u[:, j]
        all_coefs.append(tuple(row))

    return DecorrelationBasis(
        signal_names=tuple(signal_names),
        means=tuple(float(m) for m in means),
        stds=tuple(float(s) for s in stds),
        coefficients=tuple(all_coefs),
    )
