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

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Dense-Gram feasibility cap. ~25k items ≈ 5GB Gram / ~10GB peak on this box;
# override via EASE_MAX_ITEMS to attempt the 38–50k catalogs on a bigger machine.
MAX_ITEMS = int(os.environ.get("EASE_MAX_ITEMS", "25000"))


class _LinearItemItem:
    """Shared plumbing: binarize (entity,item) → X, solve item-item B, serve."""

    name = "linear"

    def _solve_B(self, G: np.ndarray, lam: float) -> np.ndarray:
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
        # The dense n×n Gram (and its inverse) is the hard wall: n² doubles
        # cap at ~25k items (≈5GB Gram, ~10GB peak) on this box; larger needs a
        # bigger machine. X is built sparse so it never blows up on user count.
        if n_i > MAX_ITEMS:
            raise ValueError(
                f"{self.name}: dense item-item solve infeasible for {n_i:,} items "
                f"(>{MAX_ITEMS:,}; {n_i * n_i * 8 / 1e9:.0f}GB Gram)"
            )
        X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_u, n_i))
        self._X = X
        self._row = u2r
        G = np.asarray((X.T @ X).todense(), dtype=np.float64)
        lam = 20.0 * X.sum() / max(n_i, 1)  # kindling's heuristic λ
        self._B = self._solve_B(G, lam)
        # per-entity owned columns for masking — one pass (the per-user boolean
        # mask version is O(n_users · nnz) and hangs on million-user catalogs).
        owned: dict = {}
        for e, c in zip(df["entity_id"].to_numpy(), cols):
            owned.setdefault(e, []).append(c)
        self._owned = {e: np.asarray(v) for e, v in owned.items()}
        return self

    def recommend(self, entity_id, n: int = 10):
        r = self._row.get(entity_id)
        if r is None:
            return []
        s = np.asarray(self._X[r] @ self._B).ravel()
        own = self._owned.get(entity_id)
        if own is not None and own.size:
            s[own] = -np.inf
        k = min(n, s.size)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [self._items[c] for c in top if np.isfinite(s[c])]


def _ease_P(G, lam):
    return np.linalg.inv(G + lam * np.eye(G.shape[0]))


class EASE(_LinearItemItem):
    """Canonical Steck (2019) EASE — the family anchor. Unlike the registry's
    `ease` (kindling's auto base, which falls back to cooc above the 20k gate),
    this forces the dense EASE solve wherever it's feasible, so the chart can
    show EASE itself on the larger catalogs."""

    name = "ease_full"

    def _solve_B(self, G, lam):
        P = _ease_P(G, lam)
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0.0)
        return B


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

    def _solve_B(self, G, lam):
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
