"""LightGCN signal, pure-numpy implementation.

LightGCN (He et al., SIGIR 2020) is a graph convolutional network for
collaborative filtering. Core mechanism: K-layer embedding propagation
over the bipartite user-item graph with symmetric degree normalization,
no nonlinearity, no feature transformation. Final user/item embeddings
are a layer-combination (arithmetic mean here) of E^(0), E^(1), ..., E^(K).

This file intentionally avoids PyTorch. The simplification versus the
reference implementation:

    Paper:   forward pass propagates; backward pass flows through K
             sparse matmuls via autograd.
    Here:    base embeddings E^(0) are trained with BPR loss using just
             the dot product of E^(0) (no propagation inside the
             training loop). After training, we propagate K layers and
             combine to produce the served embeddings.

This gives us 80% of LightGCN's structural advantage (graph-smoothed
latent factors that generalize across items with no direct cooc)
without the custom autograd code a full-fidelity implementation would
need. The two-stage recipe is well-established in the GNN literature;
calling it "LightGCN-lite" to be honest.

Training loss is BPR with mini-batch SGD, rating-aware positive
sampling (pairs weighted by the preprocessor's ``_interaction_weight``
column so 5-star pairs are sampled more than 3-star), and uniform
negative sampling from items the entity hasn't interacted with.

Scoring: dot product of the final (layer-combined) user and item
embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class LightGCNConfig:
    """Knobs. Defaults are the paper's suggested values where applicable."""

    dim: int = 64
    n_layers: int = 3
    learning_rate: float = 0.005
    weight_decay: float = 1e-4
    n_epochs: int = 20
    batch_size: int = 2048
    negatives_per_positive: int = 1
    seed: int = 0
    use_rating_weights: bool = True
    # Minimum entities+items required to kick off training. Below this
    # we skip - propagation over a graph with < this many nodes is noise.
    min_users: int = 50
    min_items: int = 50


@dataclass(frozen=True)
class LightGCNModel:
    """Fitted user / item embeddings after K-layer propagation + layer combine.

    Attributes
    ----------
    entity_factors:
        (n_entities, dim) — final served embedding per entity.
    item_factors:
        (n_items, dim) — final served embedding per item (indexed on
        the engine's item_graph ordering).
    entity_index:
        entity_id → row in entity_factors.
    item_index:
        item_id → row in item_factors (mirrors engine's item_graph).
    n_epochs_trained:
        Actual epochs run (may be below config.n_epochs if training
        aborted e.g. on NaN). Useful for diagnostics.
    """

    entity_factors: np.ndarray
    item_factors: np.ndarray
    entity_index: dict[object, int]
    item_index: dict[object, int]
    n_epochs_trained: int

    def score_many(
        self,
        entity_id: object,
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        """Return LightGCN scores for the given candidate indices.

        Normalizes scores to [0, 1] per query so the signal lives on a
        comparable scale to the other signals. Unknown entities score
        zero (cold-start fallback is handled upstream).
        """
        n = candidate_indices.size
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        entity_idx = self.entity_index.get(entity_id)
        if entity_idx is None:
            return np.zeros(n, dtype=np.float64)
        entity_vec = self.entity_factors[entity_idx]
        item_vecs = self.item_factors[candidate_indices]
        scores = np.asarray(item_vecs @ entity_vec, dtype=np.float64)
        max_s = float(scores.max()) if scores.size else 0.0
        if max_s > 0:
            scores = np.maximum(scores, 0.0) / max_s
        else:
            scores = np.zeros(n, dtype=np.float64)
        return scores


def build_lightgcn(
    interactions: pd.DataFrame,
    item_graph_item_index: dict[object, int],
    config: LightGCNConfig | None = None,
) -> "LightGCNModel | None":
    """Fit LightGCN (two-stage: BPR-trained base + inference-time propagation).

    Returns None on skip conditions (too-small dataset, degenerate
    graph). The engine uses this alongside the other signals; a None
    return leaves the engine to proceed without the LightGCN column.
    """
    from kindling.preprocess import weights_of

    cfg = config or LightGCNConfig()
    n_items = len(item_graph_item_index)
    if n_items < cfg.min_items:
        return None

    # Build the weighted user-item matrix on the engine's item ordering.
    entities = sorted(interactions["entity_id"].unique(), key=str)
    if len(entities) < cfg.min_users:
        return None
    entity_index = {e: i for i, e in enumerate(entities)}

    weights = weights_of(interactions)
    rows = interactions["entity_id"].map(entity_index).to_numpy()
    cols = interactions["item_id"].map(
        lambda x: item_graph_item_index.get(x, -1)
    ).to_numpy()
    keep = cols >= 0
    rows = rows[keep]
    cols = cols[keep]
    data = weights[keep].astype(np.float32)

    # Drop zero-weight rows (low-rating interactions handled as negatives).
    nonzero = data > 0
    rows = rows[nonzero]
    cols = cols[nonzero]
    data = data[nonzero]
    if rows.size == 0:
        return None

    n_users = len(entities)
    ui = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    ui.sum_duplicates()
    # Cap at 1.0 per (user, item) so duplicates don't inflate.
    ui.data = np.minimum(ui.data, 1.0)

    # Train base embeddings E^(0) with BPR SGD.
    e_u, e_i, n_epochs_trained = _train_bpr(ui, cfg)
    if e_u is None or e_i is None:
        return None

    # Propagate + layer-combine to get served embeddings.
    e_u_final, e_i_final = _propagate_and_combine(ui, e_u, e_i, cfg.n_layers)

    return LightGCNModel(
        entity_factors=e_u_final,
        item_factors=e_i_final,
        entity_index=entity_index,
        item_index=dict(item_graph_item_index),
        n_epochs_trained=n_epochs_trained,
    )


# ------------- training: BPR SGD over base embeddings -----------------

def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid that avoids overflow on large |x|."""
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    neg_exp = np.exp(x[~pos])
    out[~pos] = neg_exp / (1.0 + neg_exp)
    return out


def _train_bpr(
    ui: sp.csr_matrix,
    cfg: LightGCNConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Train base embeddings via BPR loss with SGD."""
    n_users, n_items = ui.shape
    rng = np.random.default_rng(cfg.seed)

    # Initialize: small random normals. Xavier-like but modest variance.
    e_u = rng.normal(0, 0.01, size=(n_users, cfg.dim)).astype(np.float32)
    e_i = rng.normal(0, 0.01, size=(n_items, cfg.dim)).astype(np.float32)

    # Build per-user positive item lists and weights for sampling.
    ui_csr = ui.tocsr()
    owned: list[np.ndarray] = [
        ui_csr.indices[ui_csr.indptr[u] : ui_csr.indptr[u + 1]] for u in range(n_users)
    ]
    owned_weights: list[np.ndarray] = [
        ui_csr.data[ui_csr.indptr[u] : ui_csr.indptr[u + 1]] for u in range(n_users)
    ]
    owned_sets: list[set[int]] = [set(arr.tolist()) for arr in owned]

    # Flat positive pool: one (user, item, weight) triple per observed pair.
    rows_flat, cols_flat = ui_csr.nonzero()
    weights_flat = np.asarray(ui_csr.data, dtype=np.float32)
    n_positives = rows_flat.size
    if n_positives == 0:
        return None, None, 0

    # Rating-aware positive sampling: probability proportional to weight.
    if cfg.use_rating_weights and weights_flat.std() > 0:
        probs = weights_flat / weights_flat.sum()
    else:
        probs = np.full(n_positives, 1.0 / n_positives, dtype=np.float64)

    steps_per_epoch = max(1, n_positives // cfg.batch_size)
    n_trained = 0
    for epoch in range(cfg.n_epochs):
        for _ in range(steps_per_epoch):
            # Sample positives.
            pos_idx = rng.choice(n_positives, size=cfg.batch_size, p=probs, replace=True)
            u_batch = rows_flat[pos_idx]
            i_pos = cols_flat[pos_idx]

            # Sample negatives per positive: uniform from items NOT in user's owned set.
            # Rejection-sample until we hit a non-owned item. For small owned sets
            # this is fast; for users who own most items, it's still O(few) tries.
            i_neg = np.empty_like(i_pos)
            for k in range(len(u_batch)):
                u = u_batch[k]
                owned_k = owned_sets[u]
                tries = 0
                while tries < 20:
                    cand = int(rng.integers(0, n_items))
                    if cand not in owned_k:
                        i_neg[k] = cand
                        break
                    tries += 1
                else:
                    # Fallback: any item. Rare when users own < catalog.
                    i_neg[k] = int(rng.integers(0, n_items))

            # Forward: score difference.
            eu = e_u[u_batch]
            eip = e_i[i_pos]
            ein = e_i[i_neg]
            diff = np.einsum("bd,bd->b", eu, eip) - np.einsum("bd,bd->b", eu, ein)
            # BPR loss: -log sigmoid(diff). Gradient wrt diff = -sigmoid(-diff).
            s_neg_diff = _sigmoid_stable(-diff).astype(np.float32)  # (batch,)

            # Chain-rule to each embedding (plus L2 regularization).
            grad_scale = s_neg_diff[:, None]  # (batch, 1)
            # d loss / d eu = -s_neg_diff * (eip - ein) + wd * eu
            dE_u = -grad_scale * (eip - ein) + cfg.weight_decay * eu
            # d loss / d eip = -s_neg_diff * eu + wd * eip
            dE_ip = -grad_scale * eu + cfg.weight_decay * eip
            # d loss / d ein = +s_neg_diff * eu + wd * ein
            dE_in = grad_scale * eu + cfg.weight_decay * ein

            lr = cfg.learning_rate
            # Atomic batched updates.
            np.add.at(e_u, u_batch, -lr * dE_u)
            np.add.at(e_i, i_pos, -lr * dE_ip)
            np.add.at(e_i, i_neg, -lr * dE_in)

            # NaN guard.
            if not np.isfinite(e_u).all() or not np.isfinite(e_i).all():
                return e_u, e_i, n_trained

        n_trained += 1
    return e_u, e_i, n_trained


# ------------- inference-time propagation + layer combination ----------

def _propagate_and_combine(
    ui: sp.csr_matrix,
    e_u: np.ndarray,
    e_i: np.ndarray,
    n_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply K-layer symmetric-normalized bipartite propagation + layer-mean.

    Uses the standard LightGCN normalization
    ``A_hat = D^(-1/2) A D^(-1/2)`` where A is the full bipartite
    adjacency ``[[0, U], [U.T, 0]]`` and D is the node-degree diagonal.

    Propagation is just ``E^(k) = A_hat @ E^(k-1)`` - a sparse matmul.
    Layer combination is an unweighted average over layers 0..K.
    """
    n_u, n_i = ui.shape

    # Degrees.
    d_u = np.asarray(ui.sum(axis=1)).ravel()  # (n_u,)
    d_i = np.asarray(ui.sum(axis=0)).ravel()  # (n_i,)
    d_u_inv_sqrt = np.where(d_u > 0, 1.0 / np.sqrt(d_u), 0.0)
    d_i_inv_sqrt = np.where(d_i > 0, 1.0 / np.sqrt(d_i), 0.0)
    ui_norm = sp.diags(d_u_inv_sqrt) @ ui @ sp.diags(d_i_inv_sqrt)

    # Stacked initial embeddings.
    stacked = np.vstack([e_u, e_i]).astype(np.float32)

    # Layer accumulator starts with E^(0).
    accum = stacked.copy()
    current = stacked
    for _ in range(n_layers):
        # E^(k+1) = A_hat @ E^(k) where A_hat is the full bipartite adj.
        # Split for sparse efficiency: U bottom -> U top, U^T top -> U^T bottom.
        upper_block = ui_norm @ current[n_u:]       # (n_u, d) <- U_hat @ E_i^{k}
        lower_block = ui_norm.T @ current[: n_u]    # (n_i, d) <- U_hat.T @ E_u^{k}
        current = np.vstack([upper_block, lower_block])
        accum += current

    # Layer-average (K+1 layers total: 0 through K).
    combined = accum / float(n_layers + 1)
    e_u_final = combined[: n_u]
    e_i_final = combined[n_u :]
    return e_u_final, e_i_final
