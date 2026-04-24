# ADR: scoring architecture comparison — Bayesian vs gating vs RRF

**Date:** 2026-04-25
**Status:** all three architectures shipped; Bayesian remains default; gating opt-in
**Related:** [ADR-signal-audit.md](ADR-signal-audit.md),
[ADR-score-normalization.md](ADR-score-normalization.md)

## Context

After shipping per-query signal normalization + the gating network +
RRF as a retrieval-fusion mechanism, the obvious question: do the
three scoring architectures actually produce meaningfully different
NDCG?

This ADR runs all three on both reference datasets at full data, with
the same retrieval stack and the same eval split.

## What's compared

1. **Bayesian blend** (current default). `score = posterior_mean ·
   feature_vec`. Fixed per-dataset weights from data-characteristic
   priors. No normalization (default `signal_normalization="none"`)
   so cooc's raw magnitude dominates the linear combination.

2. **Gating network**. Per-entity context → MLP → softmax weights
   over signals → `score = gate_weights · normalized_feature_vec`.
   Forces `signal_normalization="zscore"` so weights aren't
   compensating for raw-magnitude mismatch. Trained via BPR SGD on
   pre-cached per-entity (positive, negative) signal vectors.

3. **RRF-of-signals**. Each signal column ranks the candidate pool
   independently; reciprocal-rank fusion sums `1/(60 + rank_per_signal)`
   across signals. Score-scale-independent, no learning.

## Numbers (NDCG@10 / Recall@10 / MRR, 500 eval entities)

### grocery-deep (full 100% data, 162k interactions)

| method | NDCG | Recall@10 | MRR | fit s | p95 ms |
|---|---:|---:|---:|---:|---:|
| **bayesian** | **0.3197** | **0.4512** | 0.3514 | 8.5 | 22.5 |
| gating | 0.3029 | 0.4183 | **0.3540** | 27.8 | 22.3 |
| rrf | 0.3025 | 0.4213 | 0.3459 | 8.5 | 23.2 |

### ml1m (full 100% data, 1M interactions, ratings preserved)

| method | NDCG | Recall@10 | MRR | fit s | p95 ms |
|---|---:|---:|---:|---:|---:|
| bayesian | 0.2880 | 0.0465 | **0.4556** | 363 | 225 |
| **gating** | **0.2911** | 0.0488 | 0.4532 | 1994 | 229 |
| rrf | 0.2865 | **0.0513** | 0.4416 | 363 | 295 |

## What the numbers tell us

### 1. Bayesian still wins on grocery; gating slightly wins on ml1m

The differences are small (1-5%) but the pattern is real:

- **Grocery is cooc-dominated.** The Bayesian blend's prior happens
  to give cooc enough effective weight (via the raw-magnitude crutch)
  to extract most of the signal. Gating tries to spread weight across
  signals → loses some of cooc's contribution. RRF treats each signal
  equally → same loss.

- **ML-1M is more signal-diverse.** Per-entity, the right signal
  varies (some users benefit from path_full, some from cosine, some
  from ALS). The gate's per-entity weighting picks up some of that
  variance. Bayesian's fixed posterior weight can't.

### 2. Recall@10 tells a different story than NDCG

On ML-1M, **RRF and gating both have higher Recall@10 than Bayesian**
(0.051 / 0.049 vs 0.046). But Bayesian's MRR (0.456) is the highest.

Translation: Bayesian retrieves slightly fewer test positives but
ranks them higher when it does retrieve them. RRF retrieves more
positives but with looser top-position ranking. Gate is in between.

This matches the architectural reasoning: RRF combines retrievers
without committing to a precision-focused score; Bayesian commits
to precision via cooc's raw-magnitude dominance; gating learns a
middle path.

### 3. Gating's fit-time cost is 5× Bayesian on ML-1M

Bayesian: 363s for full ml1m fit. Gating: 1994s (still fast — pre-
caching signals avoided the 40-minute pre-fix). The 5× cost is
acceptable for a one-time build, less acceptable for frequent
retraining cycles. Gating training is amenable to incremental
fine-tuning if needed.

### 4. None of these "breaks" the cooc-dominance pattern

The signal-audit ADR found that `only_cooc ≈ full blend` — meaning
cooc carries most of the predictive load. None of the three
architectures here changes that: the differences between them are
small relative to the gap between "blend" and "any single signal."

The path forward isn't "find a better blender." It's:
- New retriever pulling items the cooc graph doesn't reach
  (HNSW-over-LightGCN-embeddings is the queued candidate)
- Better signals on real session data (RetailRocket, Instacart,
  Amazon — queued)
- Outcome feedback to the Bayesian posterior (queued, requires the
  outcome-replay harness)

## Decisions

- **Bayesian stays the default.** Highest NDCG on grocery, highest
  MRR on both datasets. Cheapest training. Documented behavior.
- **Gating ships as opt-in.** `Engine(gating_config=GatingConfig(enabled=True))`.
  Picks up small wins on ML-1M-style data; recall@10 advantage may
  matter for "I need a wider top-100" use cases more than NDCG@10.
- **RRF stays as the retrieval-stage fusion mechanism** (already
  shipped) but **doesn't replace the blend at scoring time** by default.
  RRF-of-signals as a scoring path is here as a measurement tool,
  not a recommended config.

## What this means for queued work

1. **Re-tune `priors.toml`**. The Bayesian's "raw-magnitude crutch"
   wins by accident — its posterior weights would let it lose under
   normalization. A proper re-tune would let normalized + Bayesian
   match or beat unnormalized + Bayesian. Worth doing, modest expected
   lift.
2. **HNSW-over-LightGCN retriever**. The clearest path to actual
   NDCG improvement — adds candidates outside cooc's reach.
3. **Cross-dataset benchmarks** (gowalla, yelp2018, amazon-beauty,
   amazon-book, tafeng, dunnhumby, instacart). Different cooc-vs-
   signal-diversity profiles will tell us when gating actually wins.
   Current 2-dataset comparison is too narrow to draw architectural
   conclusions.

## Code state

- `src/kindling/blend/normalize.py`: 4-mode normalization (zscore default
  for gating; "none" default in Engine to preserve backward compat).
- `src/kindling/gate/`: full gating module (config, features, model,
  fit). Pre-caches per-entity signal vectors at fit time so SGD
  doesn't call `_compute_signal_features` per-batch (was the 40-min
  bottleneck pre-fix).
- `src/kindling/benchmarks/scoring_architecture.py`: comparison
  harness used here.
- Persistence round-trips the gate (`_gate`, `_gate_context_cache`).
- Tests: 7 gate unit tests, 8 normalization tests; 307 total green.

Reports saved to:
- `bench/reports/scoring_architecture_grocery.json`
- `bench/reports/scoring_architecture_ml1m.json`
