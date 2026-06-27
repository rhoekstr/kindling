# ruff: noqa: N802, N803, N806  (linear-algebra matrix names: X, G, B, P)
"""EASE-family item-item models as warming-curve baselines (Stage 4).

Closed-form / ADMM linear autoencoders that plug into run_warming_curve's model
protocol (fit(df) → self, recommend(entity_id, n) → [item_id]) so they share the
SAME eval as kindling and the other baselines:

  EDLAE  popularity-scaled diagonal ridge on EASE (denoising)
  RLAE   relaxed zero-diagonal EASE
  ADMM-SLIM  L1+L2 sparse SLIM via ADMM (Steck 2020)

Only feasible on the ≤20k-item catalogs (dense n×n solve); the runner gates them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class _LinearItemItem:
    """Shared plumbing: binarize (entity,item) → X, solve item-item B, serve."""

    name = "linear"

    def _solve_B(self, G: np.ndarray, lam: float) -> np.ndarray:  # noqa: ARG002
        raise NotImplementedError

    def fit(self, df: pd.DataFrame):
        items = pd.Index(df["item_id"].unique())
        self._i2c = {it: c for c, it in enumerate(items)}
        self._items = items.to_numpy()
        ents = pd.Index(df["entity_id"].unique())
        u2r = {e: r for r, e in enumerate(ents)}
        rows = df["entity_id"].map(u2r).to_numpy()
        cols = df["item_id"].map(self._i2c).to_numpy()
        n_u, n_i = len(ents), len(items)
        if n_i > 20000:
            raise ValueError(f"{self.name}: dense item-item solve infeasible for {n_i} items (>20k)")
        X = np.zeros((n_u, n_i), dtype=np.float64)
        X[rows, cols] = 1.0
        self._X = X
        self._row = u2r
        G = X.T @ X
        lam = 20.0 * X.sum() / max(n_i, 1)  # kindling's heuristic λ
        self._B = self._solve_B(G, lam)
        # per-entity owned columns for masking
        self._owned = {e: cols[rows == r] for e, r in u2r.items()}
        return self

    def recommend(self, entity_id, n: int = 10):
        r = self._row.get(entity_id)
        if r is None:
            return []
        s = self._X[r] @ self._B
        own = self._owned.get(entity_id)
        if own is not None and own.size:
            s[own] = -np.inf
        k = min(n, s.size)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [self._items[c] for c in top if np.isfinite(s[c])]


def _ease_P(G, lam):
    return np.linalg.inv(G + lam * np.eye(G.shape[0]))


class EDLAE(_LinearItemItem):
    name = "edlae"

    def __init__(self, delta: float = 0.5):
        self.delta = delta

    def _solve_B(self, G, lam):
        G2 = G.copy()
        np.fill_diagonal(G2, np.diag(G2) + lam + self.delta * np.diag(G))
        P = np.linalg.inv(G2)
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0.0)
        return B


class RLAE(_LinearItemItem):
    name = "rlae"

    def __init__(self, rho: float = 1e-4):
        self.rho = rho

    def _solve_B(self, G, lam):
        P = _ease_P(G, lam)
        gamma = 1.0 / (np.diag(P) + self.rho)
        B = -P * gamma[None, :]
        np.fill_diagonal(B, 1.0 - np.diag(P) * gamma)
        return B


class ADMMSLIM(_LinearItemItem):
    name = "admm_slim"

    def __init__(self, l1: float = 1.0, l2: float = 500.0, rho: float = 1000.0, iters: int = 30):
        self.l1, self.l2, self.rho, self.iters = l1, l2, rho, iters

    def _solve_B(self, G, lam):  # noqa: ARG002
        n = G.shape[0]
        P = np.linalg.inv(G + (self.l2 + self.rho) * np.eye(n))
        PG = P @ G
        C = np.zeros((n, n))
        Gamma = np.zeros((n, n))
        thr = self.l1 / self.rho
        for _ in range(self.iters):
            B = PG + P @ (self.rho * (C - Gamma))
            # enforce zero diagonal via the EASE closed-form correction
            B -= P * (np.diag(B) / np.diag(P))[None, :]
            # soft-threshold + non-negativity for C
            A = B + Gamma
            C = np.sign(A) * np.maximum(np.abs(A) - thr, 0.0)
            np.maximum(C, 0.0, out=C)
            Gamma += B - C
        return C
