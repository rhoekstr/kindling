"""Train the gating MLP with BPR SGD + hand-computed gradients.

Pipeline per entity:
1. Build (context, positive_item, negative_item) triples.
2. Forward: context -> MLP -> softmax weights -> score_pos & score_neg
   using the engine's normalized signal features.
3. BPR loss on score_pos - score_neg.
4. Backward: chain rule through softmax -> W2 -> relu -> W1.
5. SGD update with L2 regularization.

Uses the engine's fitted signal stack (cooc/cosine/als/persona/...)
so the gate trains against the same features it'll see at inference.
Signals are normalized to zscore before training so the gate doesn't
have to learn scale-compensation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from kindling.blend.normalize import normalize_columns
from kindling.gate.config import GatingConfig
from kindling.gate.features import CONTEXT_FEATURE_NAMES, compute_context_features
from kindling.gate.model import GatingNetwork, _relu, _softmax_batch

if TYPE_CHECKING:
    from kindling.engine import Engine


def fit_gating_network(
    engine: "Engine",
    config: GatingConfig | None = None,
) -> GatingNetwork | None:
    """Train the gate. Returns None when the dataset is too small or the
    engine's signal stack hasn't been fitted.
    """
    cfg = config or GatingConfig()
    assert engine._interactions is not None

    n_users = int(engine._interactions["entity_id"].nunique())
    if n_users < cfg.min_users:
        return None

    # Context features (static per entity).
    ctx_by_entity = compute_context_features(engine)
    if not ctx_by_entity:
        return None
    entities = list(ctx_by_entity.keys())

    n_ctx = len(CONTEXT_FEATURE_NAMES)
    from kindling.engine import SIGNAL_ORDER

    n_signals = len(SIGNAL_ORDER)

    # Stack context matrix for training-time normalization stats.
    ctx_matrix = np.stack([ctx_by_entity[e] for e in entities]).astype(np.float32)
    ctx_mean = ctx_matrix.mean(axis=0)
    ctx_std = ctx_matrix.std(axis=0)
    ctx_std = np.where(ctx_std > 1e-6, ctx_std, 1.0)

    rng = np.random.default_rng(cfg.seed)
    gate = GatingNetwork.initialize(
        n_ctx=n_ctx, n_signals=n_signals, hidden_dim=cfg.hidden_dim, rng=rng
    )
    gate.ctx_mean = ctx_mean.astype(np.float32)
    gate.ctx_std = ctx_std.astype(np.float32)

    # Build per-entity signal feature cache: for each entity + candidate item,
    # compute the 11-dim signal vector (normalized to zscore). Doing this for
    # EVERY (entity, item) pair is expensive; instead, we compute lazily per
    # batch and cache recent.
    n_positives = _count_positive_pairs(engine._owned_by_entity)
    if n_positives < cfg.min_positives:
        return None

    # Pre-compute signal vectors for each entity's (owned items + a fixed
    # set of sampled negatives) ONCE. SGD then samples from this cache
    # instead of re-running _compute_signal_features per batch sample.
    # For grocery-deep: 1500 users * ~15 candidates * 11 signals = 250k
    # floats, fits in RAM easily. Previously the uncached loop called
    # _compute_signal_features ~14 * batch_size * n_epochs times (140k+
    # calls) which was the 40-min bottleneck.
    all_items = list(engine._item_graph.item_ids)
    n_all_items = len(all_items)
    cache = _build_signal_cache(
        engine=engine,
        entities=entities,
        owned_by_entity=engine._owned_by_entity,
        history_by_entity=engine._history_by_entity,
        all_items=all_items,
        neg_per_entity=cfg.negatives_per_positive * 2,
        rng=rng,
    )
    # cache[ent_idx] = (pos_sig_array, neg_sig_array) - both (n_local, n_signals)

    # Build the flat (entity_idx, local_pos_idx) training pool.
    train_pool: list[tuple[int, int]] = []
    for ent_idx in range(len(entities)):
        n_pos, _ = cache[ent_idx][0].shape[0], cache[ent_idx][1].shape[0]
        for p in range(n_pos):
            train_pool.append((ent_idx, p))
    n_triples = len(train_pool)
    if n_triples < cfg.min_positives:
        return None
    train_pool_arr = np.array(train_pool, dtype=np.int64)

    steps_per_epoch = max(1, n_triples // cfg.batch_size)
    for _epoch in range(cfg.n_epochs):
        for _step in range(steps_per_epoch):
            sample_idx = rng.integers(0, n_triples, size=cfg.batch_size)
            batch = train_pool_arr[sample_idx]
            pos_sig = np.empty((cfg.batch_size, n_signals), dtype=np.float32)
            neg_sig = np.empty((cfg.batch_size, n_signals), dtype=np.float32)
            ctx_batch = np.empty((cfg.batch_size, n_ctx), dtype=np.float32)
            for k in range(cfg.batch_size):
                ent_idx, pos_idx = int(batch[k, 0]), int(batch[k, 1])
                pos_cache, neg_cache = cache[ent_idx]
                pos_sig[k] = pos_cache[pos_idx]
                # Uniform sample from entity's cached negatives.
                neg_idx = int(rng.integers(0, neg_cache.shape[0]))
                neg_sig[k] = neg_cache[neg_idx]
                ctx_batch[k] = ctx_by_entity[entities[ent_idx]]

            _sgd_step(
                gate=gate,
                ctx_batch=ctx_batch,
                pos_sig=pos_sig,
                neg_sig=neg_sig,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
            if not (np.isfinite(gate.W1).all() and np.isfinite(gate.W2).all()):
                return gate
    return gate


def _build_signal_cache(
    engine: "Engine",
    entities: list[object],
    owned_by_entity: dict,
    history_by_entity: dict,
    all_items: list[object],
    neg_per_entity: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-entity cache of (pos_signal_matrix, neg_signal_matrix).

    Compute signals ONCE per entity for all their owned items (pos) plus
    ``neg_per_entity`` uniformly-sampled non-owned items (neg). Normalize
    each entity's (pos + neg) pool via zscore so the gate sees the same
    per-query normalization it'll use at inference time.
    """
    from kindling.engine import (
        MAX_QUERY_BASKET_SIZE,
        _compute_signal_features,
    )
    from kindling.retrieve.protocol import Candidate

    n_items = len(all_items)
    cache: list[tuple[np.ndarray, np.ndarray]] = []
    for ent in entities:
        owned = owned_by_entity.get(ent, np.array([]))
        history = history_by_entity.get(ent, ())
        owned_list = owned.tolist() if owned.size else []
        owned_set = set(owned_list)
        # Sample negatives once.
        negs: list[object] = []
        while len(negs) < neg_per_entity:
            cand = all_items[int(rng.integers(0, n_items))]
            if cand not in owned_set:
                negs.append(cand)
                owned_set.add(cand)
                if len(owned_set) == n_items:
                    break
            if len(owned_set) >= n_items:
                break

        pool = [*owned_list, *negs]
        if not pool:
            cache.append((
                np.zeros((0, len(engine._bayesian_blend.signal_names) if engine._bayesian_blend else 11)),
                np.zeros((0, len(engine._bayesian_blend.signal_names) if engine._bayesian_blend else 11)),
            ))
            continue
        candidates = [
            Candidate(item_id=c, score=0.0, source="gate_cache") for c in pool
        ]
        features = _compute_signal_features(
            candidates=candidates,
            owned_items=owned,
            query_basket=frozenset(history[-MAX_QUERY_BASKET_SIZE:]),
            history=history[-engine.max_history_for_recommend :],
            item_graph=engine._item_graph,
            tail_index=engine._tail_index,
            path_tree=engine._path_tree,
            basket_index=engine._basket_index,
            basket_similarity=engine.basket_similarity,
            cost_graph=engine._cost_graph,
            entity_id=ent,
            item_cosine=engine._item_cosine,
            als_factors=engine._als_factors,
            persona_index=engine._persona_index,
            lightgcn=engine._lightgcn,
        )
        normalized = normalize_columns(features.matrix, mode="zscore").astype(np.float32)
        n_pos = len(owned_list)
        pos_sig = normalized[:n_pos]
        neg_sig = normalized[n_pos:]
        # Guard against empty halves.
        if neg_sig.shape[0] == 0 and pos_sig.shape[0] > 0:
            neg_sig = np.zeros((1, pos_sig.shape[1]), dtype=np.float32)
        if pos_sig.shape[0] == 0:
            pos_sig = np.zeros((0, neg_sig.shape[1]), dtype=np.float32)
        cache.append((pos_sig, neg_sig))
    return cache


# --------------- training-data helpers ---------------

def _count_positive_pairs(owned_by_entity: dict) -> int:
    return sum(int(v.size) for v in owned_by_entity.values() if v is not None)


# --------------- SGD step with manual gradients ---------------

def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    negx = x[~pos]
    e = np.exp(negx)
    out[~pos] = e / (1.0 + e)
    return out


def _sgd_step(
    gate: GatingNetwork,
    ctx_batch: np.ndarray,
    pos_sig: np.ndarray,
    neg_sig: np.ndarray,
    lr: float,
    weight_decay: float,
) -> None:
    """One BPR gradient step over the batch.

    Forward:
        x = (ctx - mean) / std         (B, n_ctx)
        h = relu(x @ W1.T + b1)        (B, hidden)
        logits = h @ W2.T + b2         (B, n_signals)
        weights = softmax(logits)      (B, n_signals)
        score_pos = sum(weights * pos_sig, axis=1)
        score_neg = sum(weights * neg_sig, axis=1)
        diff = score_pos - score_neg
        loss = -log sigmoid(diff)

    Backward: chain rule through each step. The softmax+dot-product
    derivative simplifies because weights is the same vector used on
    both pos_sig and neg_sig:
        d loss / d logits_k
          = -sigmoid(-diff) * d(diff)/d logits_k
          = -sigmoid(-diff) * weights_k * (signal_diff_k - sum(weights * signal_diff))
        where signal_diff = pos_sig - neg_sig
    """
    B = ctx_batch.shape[0]
    # Forward.
    x = (ctx_batch - gate.ctx_mean) / np.maximum(gate.ctx_std, 1e-6)
    pre_h = x @ gate.W1.T + gate.b1            # (B, hidden)
    h = _relu(pre_h)
    logits = h @ gate.W2.T + gate.b2           # (B, n_signals)
    weights = _softmax_batch(logits)           # (B, n_signals)

    signal_diff = pos_sig - neg_sig            # (B, n_signals)
    diff = (weights * signal_diff).sum(axis=1)  # (B,)

    # Gradient of loss wrt logits:
    #   d loss / d logits = -sigmoid(-diff)[:, None] * weights * (signal_diff - diff[:, None])
    s_neg = _sigmoid_stable(-diff).astype(np.float32)   # (B,)
    dlogits = -s_neg[:, None] * weights * (signal_diff - diff[:, None])  # (B, n_signals)

    # W2 gradient: (n_signals, hidden)
    dW2 = dlogits.T @ h / B + weight_decay * gate.W2
    db2 = dlogits.mean(axis=0)

    # Backprop to hidden.
    dh = dlogits @ gate.W2                     # (B, hidden)
    dpre_h = dh * (pre_h > 0.0)                # ReLU mask
    dW1 = dpre_h.T @ x / B + weight_decay * gate.W1
    db1 = dpre_h.mean(axis=0)

    gate.W2 -= (lr * dW2).astype(np.float32)
    gate.b2 -= (lr * db2).astype(np.float32)
    gate.W1 -= (lr * dW1).astype(np.float32)
    gate.b1 -= (lr * db1).astype(np.float32)
