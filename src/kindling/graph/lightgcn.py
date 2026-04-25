"""LightGCN signal, pure-numpy implementation (full end-to-end gradient).

LightGCN (He et al., SIGIR 2020) is a graph convolutional network for
collaborative filtering. Core mechanism: K-layer embedding propagation
over the bipartite user-item graph with symmetric degree normalization,
no nonlinearity, no feature transformation. Final user/item embeddings
are a layer-combination (arithmetic mean here) of E^(0), E^(1), ..., E^(K).

This file intentionally avoids PyTorch by implementing the gradient
through K propagation layers analytically:

    Forward (per batch step):
        E^(0)        = base embeddings (the trainable parameters)
        E^(k+1)      = A_hat @ E^(k)              for k = 0..K-1
        E_final      = (1/(K+1)) sum_{k=0..K} E^(k)
        score(u,i)   = E_final[u] . E_final[i]
        loss         = -log sigmoid(score(u, i_pos) - score(u, i_neg))

    Backward (analytic):
        dL/dE_final  = sparse build from BPR triples
        dL/dE^(0)    = (1/(K+1)) sum_{k=0..K} A_hat^k @ dL/dE_final

        Because A_hat is symmetric for the bipartite block adjacency
        ``[[0, U_norm], [U_norm.T, 0]]``, A_hat^T = A_hat, so the
        backward propagation has the same structure (and cost) as the
        forward propagation.

This is the full end-to-end LightGCN training objective — base
embeddings are optimized so that AFTER propagation they differentiate
positives from negatives. The earlier two-stage shortcut (BPR on raw
base, propagate only at inference) was abandoned because it
underperformed badly on sparse bipartite graphs (yelp2018 lightgcn
collapsed to NDCG 0.013 vs cooc 0.037, recall@budget 0.43 vs 0.88).
The fix recovers the train/inference alignment.

Training: full end-to-end BPR with mini-batch SGD, rating-aware
positive sampling, vectorized uniform negative sampling, sparse-on-base
L2 regularization.

Scoring: dot product of the final (propagated + layer-combined) user
and item embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class LightGCNConfig:
    """Knobs. Defaults are the paper's suggested values where applicable.

    With end-to-end propagation in the BPR loop, each step is roughly
    ``2K`` sparse matmuls of cost ``O(nnz(A) * dim)``. For a 1.2M-edge
    graph (yelp2018), K=3, dim=64 → ~500M flops/step. We compensate
    with a bigger default ``batch_size`` (8192) so steps_per_epoch
    stays manageable; the propagation cost is per-step, not per-triple,
    so larger batches are essentially free here.
    """

    dim: int = 64
    n_layers: int = 3
    learning_rate: float = 0.005
    weight_decay: float = 1e-4
    n_epochs: int = 30
    batch_size: int = 8192
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
    """End-to-end BPR with gradient through K-layer LightGCN propagation.

    Forward, per batch step:
        E^(0)        = stack(e_u, e_i)                         (n_u + n_i, dim)
        E^(k+1)[u]   = ui_norm @ E^(k)[i]                      (n_u, dim)
        E^(k+1)[i]   = ui_norm.T @ E^(k)[u]                    (n_i, dim)
        E_final      = (1/(K+1)) sum_{k=0..K} E^(k)
        s_pos        = E_final[u] . E_final[i_pos]             (B,)
        s_neg        = E_final[u] . E_final[i_neg]             (B,)
        loss         = -log sigmoid(s_pos - s_neg)

    Backward (analytic):
        dL/d_diff    = -sigmoid(-(s_pos - s_neg))              (B,)
        Sparse on E_final:
            dL/dE_final[u]      += -dL/d_diff * (E_final[ip] - E_final[in])
            dL/dE_final[ip]     += -dL/d_diff * E_final[u]
            dL/dE_final[in]     += +dL/d_diff * E_final[u]
        Through layer-mean: dL/dE^(k) = (1/(K+1)) * dL/dE_final
        Through propagation: A_hat is symmetric, so the backward
        propagation has the same structure (and cost) as forward:
            dL/dE^(0) = (1/(K+1)) * sum_{k=0..K} A_hat^k @ dL/dE_final
        which expands to repeating the bipartite matmul structure
        ``g_u <- ui_norm @ g_i; g_i <- ui_norm.T @ g_u``.

    L2 regularization is applied sparsely on the BPR-triple base rows
    only — matching the published LightGCN formulation and avoiding
    decay on nodes the gradient backprop doesn't directly select.
    """
    n_users, n_items = ui.shape
    rng = np.random.default_rng(cfg.seed)

    # Initialize: small random normals.
    e_u = rng.normal(0, 0.01, size=(n_users, cfg.dim)).astype(np.float32)
    e_i = rng.normal(0, 0.01, size=(n_items, cfg.dim)).astype(np.float32)

    # Build symmetrically-normalized bipartite adjacency once.
    d_u = np.asarray(ui.sum(axis=1)).ravel()
    d_i = np.asarray(ui.sum(axis=0)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        d_u_inv_sqrt = np.where(d_u > 0, 1.0 / np.sqrt(d_u), 0.0)
        d_i_inv_sqrt = np.where(d_i > 0, 1.0 / np.sqrt(d_i), 0.0)
    ui_norm = (sp.diags(d_u_inv_sqrt) @ ui @ sp.diags(d_i_inv_sqrt)).tocsr()
    ui_norm_t = ui_norm.T.tocsr()  # cache transpose for repeated access

    # Per-user owned sets for negative-sampling rejection.
    ui_csr = ui.tocsr()
    owned_sets: list[set[int]] = [
        set(ui_csr.indices[ui_csr.indptr[u] : ui_csr.indptr[u + 1]].tolist())
        for u in range(n_users)
    ]

    # Flat positive pool.
    rows_flat, cols_flat = ui_csr.nonzero()
    weights_flat = np.asarray(ui_csr.data, dtype=np.float32)
    n_positives = rows_flat.size
    if n_positives == 0:
        return None, None, 0

    if cfg.use_rating_weights and weights_flat.std() > 0:
        probs = weights_flat / weights_flat.sum()
    else:
        probs = np.full(n_positives, 1.0 / n_positives, dtype=np.float64)

    K = cfg.n_layers
    layer_scale = np.float32(1.0 / (K + 1))
    lr = cfg.learning_rate
    wd = cfg.weight_decay
    decay_mul = np.float32(1.0 - lr * wd)
    B = cfg.batch_size
    steps_per_epoch = max(1, n_positives // B)
    n_trained = 0

    for _epoch in range(cfg.n_epochs):
        for _step in range(steps_per_epoch):
            # ---- Sample BPR triples ----
            pos_idx = rng.choice(n_positives, size=B, p=probs, replace=True)
            u_batch = rows_flat[pos_idx]
            i_pos = cols_flat[pos_idx]

            # Vectorized negative sampling with rejection on owned set.
            # First-shot uniform; resample only the conflicts (typically <5%).
            i_neg = rng.integers(0, n_items, size=B).astype(np.int64)
            for k in range(B):
                tries = 0
                while int(i_neg[k]) in owned_sets[int(u_batch[k])] and tries < 20:
                    i_neg[k] = int(rng.integers(0, n_items))
                    tries += 1

            # ---- Forward propagation: build E_final from current e_u, e_i ----
            # Keep upper (user) and lower (item) blocks as separate arrays
            # to avoid n_u+n_i x dim vstacks every step.
            cur_u, cur_i = e_u, e_i
            acc_u, acc_i = e_u.copy(), e_i.copy()
            for _ in range(K):
                new_u = ui_norm @ cur_i           # (n_u, dim)
                new_i = ui_norm_t @ cur_u         # (n_i, dim)
                cur_u, cur_i = new_u, new_i
                acc_u += cur_u
                acc_i += cur_i
            ef_u = acc_u * layer_scale            # (n_u, dim)
            ef_i = acc_i * layer_scale            # (n_i, dim)

            # ---- Score the BPR triples on E_final ----
            eu = ef_u[u_batch]
            eip = ef_i[i_pos]
            ein = ef_i[i_neg]
            diff = (eu * eip).sum(axis=1) - (eu * ein).sum(axis=1)
            s_neg_d = _sigmoid_stable(-diff).astype(np.float32)  # (B,)
            scale = -s_neg_d[:, None]

            # ---- Build sparse dL/dE_final ----
            dL_u = np.zeros_like(e_u)
            dL_i = np.zeros_like(e_i)
            np.add.at(dL_u, u_batch, scale * (eip - ein))
            np.add.at(dL_i, i_pos, scale * eu)
            np.add.at(dL_i, i_neg, -scale * eu)

            # ---- Backward through propagation + layer-mean ----
            # dL/dE^(0) = layer_scale * sum_{k=0..K} A_hat^k @ dL/dE_final
            cur_gu, cur_gi = dL_u, dL_i
            acc_gu, acc_gi = dL_u.copy(), dL_i.copy()
            for _ in range(K):
                new_gu = ui_norm @ cur_gi
                new_gi = ui_norm_t @ cur_gu
                cur_gu, cur_gi = new_gu, new_gi
                acc_gu += cur_gu
                acc_gi += cur_gi
            dE_u = acc_gu * layer_scale
            dE_i = acc_gi * layer_scale

            # ---- Apply gradient ----
            e_u -= lr * dE_u
            e_i -= lr * dE_i

            # Sparse L2 reg on the BPR-triple base rows (matches paper).
            e_u[u_batch] *= decay_mul
            e_i[i_pos] *= decay_mul
            e_i[i_neg] *= decay_mul

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
    with np.errstate(divide="ignore", invalid="ignore"):
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
