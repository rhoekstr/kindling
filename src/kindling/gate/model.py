"""Gating MLP + forward/backward + softmax output over signals.

Two-layer fully-connected:
    h = relu(W1 @ x + b1)
    logits = W2 @ h + b2
    weights = softmax(logits)

where x is the context-feature vector (n_ctx,) and weights is the
signal-weight distribution (n_signals,) applied at scoring time:
    score(candidate) = weights . signal_vector(candidate)

BPR training runs against the engine's signal features directly.
See `fit.py` for the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GatingNetwork:
    """Trained two-layer MLP gating network.

    Attributes
    ----------
    W1, b1:
        Input -> hidden parameters. Shape (hidden_dim, n_ctx) and (hidden_dim,).
    W2, b2:
        Hidden -> output parameters. Shape (n_signals, hidden_dim) and (n_signals,).
    ctx_mean, ctx_std:
        Z-score normalization stats computed at fit time over the training
        entities' context features. Used to standardize the input at both
        train and inference. Prevents wild outputs when an entity's
        context is e.g. an order of magnitude more active than training.
    n_ctx:
        Context feature dimension.
    n_signals:
        Output dimension (== len(SIGNAL_ORDER)).
    """

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    ctx_mean: np.ndarray
    ctx_std: np.ndarray
    n_ctx: int
    n_signals: int

    @classmethod
    def initialize(
        cls,
        n_ctx: int,
        n_signals: int,
        hidden_dim: int,
        rng: np.random.Generator,
    ) -> "GatingNetwork":
        """Xavier-style init."""
        scale1 = np.sqrt(2.0 / max(n_ctx, 1))
        scale2 = np.sqrt(2.0 / max(hidden_dim, 1))
        return cls(
            W1=(rng.standard_normal((hidden_dim, n_ctx)) * scale1).astype(np.float32),
            b1=np.zeros(hidden_dim, dtype=np.float32),
            W2=(rng.standard_normal((n_signals, hidden_dim)) * scale2).astype(np.float32),
            b2=np.zeros(n_signals, dtype=np.float32),
            ctx_mean=np.zeros(n_ctx, dtype=np.float32),
            ctx_std=np.ones(n_ctx, dtype=np.float32),
            n_ctx=n_ctx,
            n_signals=n_signals,
        )

    def forward(self, ctx: np.ndarray) -> np.ndarray:
        """Return the softmax signal-weight vector for a single entity.

        ``ctx`` is (n_ctx,). Normalizes via stored ctx_mean/std. Output
        is (n_signals,) summing to 1.0.
        """
        h = _relu(self.W1 @ self._normalize(ctx) + self.b1)
        logits = self.W2 @ h + self.b2
        return _softmax(logits)

    def forward_batch(self, ctx: np.ndarray) -> np.ndarray:
        """Batched version. ``ctx`` is (B, n_ctx) -> returns (B, n_signals)."""
        x = (ctx - self.ctx_mean) / np.maximum(self.ctx_std, 1e-6)
        h = _relu(x @ self.W1.T + self.b1)
        logits = h @ self.W2.T + self.b2
        return _softmax_batch(logits)

    def _normalize(self, ctx: np.ndarray) -> np.ndarray:
        return (ctx - self.ctx_mean) / np.maximum(self.ctx_std, 1e-6)


# ------------------------------ helpers -------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    e = np.exp(shifted)
    return e / e.sum()


def _softmax_batch(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)
