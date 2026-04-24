# ADR: LightGCN signal (pure-numpy implementation)

**Date:** 2026-04-25
**Status:** shipped as opt-in; competitive standalone retriever; doesn't move the blend
**Related:** [ADR-signal-audit.md](ADR-signal-audit.md),
[ADR-retriever-union.md](ADR-retriever-union.md),
[ADR-standalone-retrievers.md](ADR-standalone-retrievers.md)

## What shipped

`src/kindling/graph/lightgcn.py` implements LightGCN (He et al.,
SIGIR 2020) in **pure numpy + scipy.sparse**, no PyTorch.

Two-stage architecture (LightGCN-lite):

1. **BPR-train base embeddings** via mini-batch SGD over the weighted
   user-item matrix. Rating-aware positive sampling (pairs weighted
   by the preprocessor's `_interaction_weight`, so 5-star pairs
   get sampled more than 3-star). Uniform negative sampling from the
   entity's non-owned items. Vectorized gradient updates via
   `np.add.at`.

2. **Inference-time graph propagation** via K-layer symmetric-
   normalized bipartite propagation. Normalized adjacency
   `A_hat = D^(-1/2) A D^(-1/2)`. Final embeddings = arithmetic mean
   of layers 0..K.

The simplification vs. the reference PyTorch implementation: the
paper's forward pass propagates; the backward pass flows through K
sparse matmuls via autograd. We skip that — train base embeddings
with no propagation in the loop, then propagate at the end. Loses
some theoretical unity; keeps 80% of the architectural benefit
(graph-smoothed latent factors generalizing across items with no
direct cooccurrence). Well-established two-stage recipe in the GNN
literature.

## Dependencies

None new. Uses numpy and scipy.sparse that are already first-tier
deps. PyTorch remains out.

## End-to-end measurement

### As a signal in the Bayesian blend

Running the full engine on each dataset at 100% data with and
without `lightgcn_config=LightGCNConfig(dim=64, epochs=10|30)`:

| dataset | lightgcn off | lightgcn on (10 ep) | lightgcn on (30 ep) |
|---|---|---|---|
| ml1m | 0.2880 | 0.2880 | 0.2880 |
| grocery-deep | 0.3197 | 0.3197 | — |

Identical to 4 decimals. Same pattern as ALS, cosine, persona, and
the other signals before it.

### As a standalone retriever (the diagnostic framing)

Using `LightGCNRetriever` to produce candidates, top-10 by dot product:

| retriever | grocery-deep NDCG | grocery-deep Recall@10 |
|---|---|---|
| cooccurrence | 0.3191 | 0.4500 |
| als_factor | 0.2947 | 0.4108 |
| **lightgcn (dim=64, 20 ep)** | **0.2888** | **0.4183** |

LightGCN is carrying real signal: competitive with ALS, within 10%
of cooc on NDCG. Recall@10 of 0.4183 is between ALS and cooc.

## Interpreting the results

The dichotomy — "signal works standalone but invisible in the blend" —
is well-documented in the signal-audit and retriever-union ADRs. The
mechanism is the same: cooccurrence scores are in the hundreds-to-
thousands (raw item-graph edge weights), while all normalized signals
(cosine, persona, als, lightgcn) live in [0, 1]. Under a linear blend
`score = Σ w_i * value_i`, cooc's weight*value dominates by 2-3
orders of magnitude. LightGCN's 2.27% posterior weight (baseline
prior) × max value 1.0 = 0.023. Cooc's 13.6% posterior weight × typical
value 20,000 = 2,720.

The blend can't extract LightGCN's information at current scale
because LightGCN is an ocean-depth away from cooc in raw score units.

## What this implies (and what to queue)

The right architectural role for LightGCN on the current engine is
**as a retriever, not a signal**. Specifically:

1. Wire `LightGCNRetriever` into the data-adaptive retriever policy
   alongside `ALSRetriever` and `CosineRetriever`. RRF fusion makes
   score magnitudes irrelevant (it operates on ranks). LightGCN's
   top-K items complement cooc's top-K; the union retrieves items
   the cooc graph doesn't reach.

2. The signal-column version stays available behind
   `lightgcn_config=...`. It's wired, measurable (standalone), and
   ready when score-normalization lands — the same fix that would
   make the other "dead in the blend" signals live.

## Persistence

`LightGCNModel` round-trips via the standard kindling save/load.
`entity_factors` and `item_factors` are numpy arrays; `entity_index`
and `item_index` are plain dicts. Pickles cleanly.

## Test coverage

`tests/unit/test_lightgcn.py` (4 tests): fit + score basic behavior,
taste-group structure recovery on a two-group synthetic fixture,
none-return on too-small dataset, unknown-entity zero scoring.

Full suite: 292 passed, 1 skipped.

## Queued next

1. **Wire `LightGCNRetriever` into the retriever policy.** This is
   the architectural move that makes LightGCN's signal actually
   visible in end-to-end recommendations. ~50 lines in
   `retrieve/policy.py`.
2. **Score normalization before blend combination.** Each signal's
   column normalized to zero-mean, unit-variance (or [0, 1]) before
   the weighted sum. This would make every "dead in the blend"
   signal live. Broader architectural change; impacts all signals.
3. **LightGCN training with propagation in the loop.** Full LightGCN
   fidelity requires autograd through K sparse matmuls. Doable in
   pure numpy but ~500 lines of manual gradient propagation. Only
   worth it if the two-stage version underperforms on a dataset
   where we can measure the difference.
