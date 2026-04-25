# kindling — Reference

> A reference for the kindling recommender as it stands today (Apr 2026).
> Synthesizes the architectural decisions, the eleven signals, the three
> scoring architectures, and the empirical results from the ADRs in
> `bench/reports/`.
>
> **This document is the living source of truth.** Whenever the
> architecture, signals, defaults, or benchmark results change, update
> this file in the same change. ADRs in `bench/reports/` remain the
> deeper chain-of-evidence; this is the synthesis.

---

## 1. What kindling is

A hybrid Python recommender library. Three-stage funnel — **retrieve → rank
→ rerank** — with a Bayesian-blend default scorer, pluggable retrievers /
rankers, and a small but opinionated set of supplementary modules
(personas, repeat-consumption, drift, persistence). Pure Python + NumPy
+ SciPy at every layer; no PyTorch, no autograd. Rust crate is parked
because Phase-8 evidence didn't justify it (see
`ADR-phase8-rust-decision.md`).

The design sells **calibrated uncertainty** as the differentiator: every
recommendation carries a Beta credible interval from the variational
posterior, not a heuristic margin. The point of the library is "honest
defaults" over headline NDCG — picking a default that is auditable and
reproducible even when a slightly higher number is reachable.

### Three-stage funnel

```
fit(interactions)
    └── builds: item_graph, path_tree, tail_index, basket_index,
                item_cosine, als_factors, persona_index, lightgcn,
                cost_graph, repeat_module, bayesian_blend, [gate]
recommend(entity_id, n=K)
    ├── 1. RETRIEVE  ─ union of pluggable retrievers (cooc / path /
    │                  basket / cosine / ALS / persona / LightGCN /
    │                  popularity), max-score or RRF fusion, dedup,
    │                  budget-capped
    ├── 2. RANK      ─ score each candidate with K signals → blend
    │                  via Bayesian posterior mean (default), gating
    │                  network (opt-in), or RRF-of-signals (test only)
    └── 3. RE-RANK   ─ DPP-style diversity, per-position temperature,
                       calibration (Steck), lift emphasis, constraints
```

Every layer is replaceable via a Protocol. Default ranker is the blend
itself (`use_ranker=False`); LightGBM/XGBoost/CatBoost rankers exist but
are off-by-default because the matrix audit (see §6) showed the blend
already saturates the available signal-space.

---

## 2. The eleven signals

`SIGNAL_ORDER` in `engine.py`:

| # | signal | source | what it captures |
|---|---|---|---|
| 1 | `path_full` | `PathTree` | exact prefix sequences `A→B→C` |
| 2 | `path_tail` | `TailIndex` | "what comes after seeing X" |
| 3 | `path_basket` | `BasketIndex` | per-basket co-membership scoring with similarity-weighted lift |
| 4 | `cooccurrence` | `ItemGraph` | symmetric pair counts, the workhorse |
| 5 | `cost_population` | `CostGraph` | global negative-signal weight |
| 6 | `cost_entity` | `CostGraph` | per-entity removed/skipped weight |
| 7 | `cost_context` | `CostGraph` | per-context (e.g. session) cost |
| 8 | `item_item_cosine` | `ItemCosine` | TF-IDF–weighted user-vector cosine over items |
| 9 | `als_factor` | implicit ALS | dot-product of latent user/item factors |
| 10 | `persona` | HDBSCAN+UMAP clusters | per-persona overperformance ratio |
| 11 | `lightgcn` | pure-numpy LightGCN | K-layer message-passing embedding dot-product |

Each signal is computed in `_compute_signal_features` and emerges as one
column in an `(n_candidates, 11)` matrix. The blend reads those columns;
the gate weights them per-entity via softmax; RRF-of-signals ranks each
column and reciprocal-rank-fuses. See §4 for how each is combined.

### 2.1 Path family — the basket variant is the distinctive one

The three `path_*` signals decompose prefix mining into:

- **path_full**: full ordered prefix counts. Sparse on most data.
  NDCG@10 ≈ 0.05 grocery, 0.03 ml1m as a standalone retriever.
- **path_tail**: marginal "next item given last item". Mid-strength.
- **path_basket**: per-basket inverted index with **basket
  similarity** (coverage / Jaccard / IDF-weighted / exact) and
  similarity-weighted lift. This is the signal closest to "you also
  buy X with Y" reasoning, and is the one most pluggable for grocery
  / e-commerce. Cost is fit-time (basket-index build = 91s on ml1m at
  full data).

### 2.2 Cost graph — three-layer additive model

`α_pop · population_cost + entity_cost + context_cost` (PRD §3.6).
`α_pop` defaults to 0.3. Cost is signed: negative actions (`remove`,
low ratings under `mode="explicit"`) increase cost; the ranker
subtracts cost from positive scores.

### 2.3 LightGCN — pure-numpy, end-to-end gradient through K layers

`ADR-lightgcn-numpy.md`. The earlier two-stage shortcut (BPR-train base
embeddings; propagate only at inference) is **abandoned** as of Apr
2026 — it broke badly on sparse bipartite graphs (yelp2018: NDCG 0.013
vs cooc 0.037, recall@budget 0.43 vs 0.88; amazon-beauty similarly).
The base BPR optimized raw dot products, but inference served smoothed
`mean(E^(0..K))` — the gradient never told the base embeddings
"after propagation, you should still differentiate positives from
negatives," so propagation homogenized them.

The current implementation:

- **Forward (per batch step)**: stack base embeddings, propagate K
  layers `E^(k+1) = A_hat · E^(k)`, layer-mean to `E_final`, score
  BPR triples on `E_final`.
- **Backward (analytic)**: build sparse `dL/dE_final` from the BPR
  triples; propagate it back through K layers. Because A_hat is
  symmetric for the bipartite `[[0, U_norm], [U_norm.T, 0]]` block,
  backward propagation has the same matmul structure (and cost) as
  forward: `dL/dE^(0) = (1/(K+1)) · sum_{k=0..K} A_hat^k @ dL/dE_final`.
- **Sparse L2** on the BPR-triple base rows only (matches the paper).
- **Defaults**: n_epochs=30, batch_size=8192. Per-step cost is
  dominated by the 2K sparse matmuls, not the batch — bigger batches
  are essentially free.

Validated on yelp2018: recall@budget jumped 0.43 → 0.77, confirming
the model now retrieves the right neighborhoods. NDCG ~62% lift
(0.013 → 0.021) — the remaining gap to cooc is hyperparameter
territory (published LightGCN runs 1000 epochs vs our 30, plus hard-
negative mining), not architectural.

Cost: ml1m LightGCN fit went 30s → 30s+ (TBD on re-run); yelp went
74s → 660s. Acceptable for the architectural correctness gain.

### 2.4 Persona signal — HDBSCAN + UMAP, cold-start aware

`ADR-persona-signal.md`. Users are clustered in latent-cosine space,
then per-persona "overperformance" ratios become the persona signal.
Cold-start users (history < threshold) get a soft assignment from
**log1p + L2-normalized** overperformance vectors — the early
implementation hit a scale-mismatch bug where raw ratios maxed at
254 vs main vectors at 0.22. Fixed scale → ml1m persona NDCG went
0.0002 → 0.140.

---

## 3. Data ingestion

### 3.1 Centralized preprocessor

`engine._preprocess_interactions` is the single entry point that:

1. Validates schema (entity_id, item_id required; timestamp,
   session_id, action_type, rating optional) via pandera.
2. Computes `_interaction_weight` from rating: `w = max(0, (rating - 2.5) / 2.5)`
   clipped to [0, 1]. Wired through cooc / path / cost as a count
   multiplier (`ADR-rating-aware-signals.md`).
3. Infers sessions when `session_id` is missing — 2-component GMM
   on log-inter-event-deltas with `manual_fallback` (30-min default)
   when the GMM doesn't find a clear bimodal gap.
4. Caches `_owned_by_entity` and `_history_by_entity` for O(1)
   lookup at recommend time.

### 3.2 Reference datasets

| dataset | format | timestamps | basket structure | cooc-vs-diversity |
|---|---|---|---|---|
| movielens-1m | rating CSV | yes (rating-burst) | manual_fallback only | mid-diverse |
| synthetic-grocery / -deep | generator | synthesized | real session_id | cooc-dominant |
| retailrocket | impression log | yes | session-grouped | mid |
| **instacart** | Kaggle order_products | synthesized | real BASKET_ID | cooc-dominant |
| **gowalla** | SNAP check-ins | yes | session-light | mid |
| **yelp2018** | academic JSON | yes | none | mid |
| **tafeng** | Kaggle tx | yes | day = basket | cooc-dominant |
| **dunnhumby** | Complete Journey | yes | real BASKET_ID | cooc-dominant |
| **amazon-beauty / -book** | 5-core JSONL | yes | none | rating-driven |

Bold rows are loaders shipped Apr 2026; data files are user-provided
(`~/.cache/kindling/<dataset>/`). Each loader raises a structured
`<Dataset>DataNotAvailableError` with a download-source URL when files
are missing.

The cross-dataset architecture-comparison harness lives at
`kindling.benchmarks.cross_dataset` — runs Bayesian + gating + RRF on
every dataset that has data on disk; skips others gracefully.

---

## 4. Scoring architectures

`ADR-scoring-architecture.md` shipped all three; **Bayesian is the
default**.

### 4.1 Bayesian blend (default)

- Mean-field Dirichlet variational posterior over the 11-signal
  weight simplex (`blend/vi.py`).
- Likelihoods: listwise calibration (default), pairwise BT,
  multinomial, binary independent (`blend/likelihoods.py`).
- Position-Based Model with η=1.0 in the listwise likelihood.
- Priors built from data-characteristic features in
  `blend/priors.toml` — 17+ coefficients tunable as a community
  asset, not hard-coded magic numbers.
- Beta-marginal **credible intervals** (renamed from PRD's
  "confidence_interval" — kindling sells calibrated uncertainty,
  not frequentist coverage).
- Convergence diagnostics: ELBO monotonicity, posterior-predictive
  Brier, variational ESS — surface as warnings via
  `engine.posterior_summary()`.

Scoring at recommend time: `score = posterior_mean · feature_vec`.
Default `signal_normalization="none"` to preserve the calibration
the priors were tuned against (the magnitude of cooc dominates the
linear combination, by design — see §5).

### 4.2 Gating network (opt-in)

- Per-entity context features → 2-layer MLP → softmax weights over
  signals → score = `gate_weights · normalized_feature_vec`.
- Trained with manually-computed BPR gradients (no autograd):
  forward through softmax + dot-product, backward through the
  closed-form `weights · (signal_diff - sum(weights · signal_diff))`
  derivative.
- Forces `signal_normalization="zscore"` so weights aren't
  compensating for raw-magnitude mismatch.
- **Pre-caches per-entity (pos_sig, neg_sig) matrices once before
  SGD** — earlier per-batch calls to `_compute_signal_features`
  were the 40-minute training bottleneck on ml1m. After the cache
  fix: ~33 min on ml1m at full data, ~5× the Bayesian fit cost.

Wins on ML-1M's signal-diverse profile (NDCG +0.3% over Bayesian).
Loses on grocery (NDCG -1.7% vs Bayesian) where cooc dominance is
load-bearing.

### 4.3 RRF-of-signals (measurement only)

- Each signal column ranks the candidate pool independently;
  reciprocal-rank fusion sums `1/(60 + rank_per_signal)` across
  signals.
- Score-scale-independent, no learning.
- Tracks the gating result on ml1m (slightly higher recall@10),
  loses to Bayesian on grocery.

Shipped as a benchmark/measurement architecture; **not a default
scoring path**. The retrieval-stage RRF fusion is a separate
mechanism (already shipped) and is the recommended fusion at the
retrieval layer.

### 4.4 Cooc + adaptive boosting (Apr 2026)

A fourth scoring architecture: cooccurrence as the primary, plus a
cumulative stack of one-tailed z-gated boost layers. Frames the
problem as "cooc rules; refinement signals nudge, only when they
fire confidently."

```
score(c) = cooc(c) + sum_layers boost · I[ z_layer(c) > tau ]
```

Each layer (path_basket, session_cooccurrence, temporal_cooccurrence)
contributes an additive boost only when the candidate's one-tailed
z-score within the layer's *non-zero subset* exceeds `tau`. Boost
magnitude is calibrated to `boost_multiplier × median(adjacent gaps
in cooc top-20)` — physical units, ~3 rank positions per firing
layer.

**Why one-tailed:** sparse refinement signals have asymmetric
semantics. High path_basket score on `c` = "yes, this confidently
appears in baskets near recent activity." Low score = "the index has
no data on `c`," not "c is bad." Boosts only — never penalties.

**Why z over the non-zero subset:** with 80% zero candidates, z over
all candidates makes σ tiny and almost everything fires. Z over the
non-zero population gives a fair "stand out among items that
registered any signal."

**Two-stage gating (data-shape aware):**

1. **Per-layer meaningfulness gate** at fit time
   (`is_layer_meaningful`): rejects layers with too-few-nonzero,
   low fire rate (<1%, layer is silent), or high fire rate (>30%,
   z-threshold isn't being selective). path_basket is auto-skipped
   on no-session datasets; session_cooccurrence shares the
   rating-burst guard with the temporal kernel.
2. **Per-candidate z-threshold** at recommend time: only candidates
   with confident layer signal get the boost.

**Auto-calibration** (`blend/layered_calibrator.py`): at fit time,
sweep a 3×3 grid over (z ∈ {2.0, 2.5, 3.0}) × (boost ∈ {1.0, 3.0,
5.0}) using leave-3-random-items-out on a 200-user sample, pick the
config that maximizes held-out NDCG@10. Default-preference tie-break
(z=2.5, b=3.0) when cells score within 0.003 NDCG. Cost: 0.2-12s
depending on dataset size.

**Headline numbers** (full-data, 500 eval entities):

| dataset | cooc | bayesian blend | adaptive layered | vs blend |
|---|---:|---:|---:|---:|
| grocery-deep | 0.3191 | 0.3197 | **0.3213** | **+0.5%** |
| ml1m | 0.2877 | 0.2878 | **0.2894** | **+0.6%** |
| amazon-beauty | 0.0302 | 0.0201 | **0.0310** | **+54%** |
| yelp2018 | 0.0366 | 0.0363 | 0.0364 | +0.0% |

Calibrator picks per dataset:

- grocery-deep: z=2.5, b=3.0 (matches manual-sweep optimum)
- ml1m: z=2.5, b=3.0 (default-preference fallback; manual optimum
  was z=2.5, b=5.0; gap ~0.6% NDCG)
- amazon-beauty: z=3.0, b=5.0 (matches manual-sweep optimum)
- yelp: fallback to default (only 1 layer fires; nothing to
  discriminate)

**Two architectural strengths over Bayesian blend:**

1. **Robust to bad-signal-mix datasets.** amazon-beauty has 8
   events/user; signals like lightgcn / persona / cosine are all
   worse than cooc on it. Bayesian blend can't downweight them
   enough — its NDCG is **33% below cooc** (0.0201 vs 0.0302).
   Adaptive layered's z-gate ignores the noisy contributions and
   stays at cooc baseline + selective boosts → +54% vs blend.
2. **Clean fallback when no info available.** yelp has only 1
   active layer (temporal_cooc, near-redundant with cooc). The
   calibrator's fallback-to-default + all-cells-tied detection
   keeps the engine at cooc baseline; no regression vs cooc-alone
   under noise.

**Cost** (per `Engine.recommend()` call):

| architecture | p95 ms (grocery) | p95 ms (ml1m) |
|---|---:|---:|
| cooc_alone | 0.7 | 4.8 |
| layered_adaptive | 16.7 | 4-7 |
| bayesian_blend | 27.1 | 44.6 |

Layered is **faster than Bayesian blend** (no VI-posterior dot-
product at query time, just a sparse-matvec per layer + boost
arithmetic).

**Status (Apr 2026)**: shipped as a probe + auto-calibrator + 49
unit tests + **engine integration**. The user-facing API:

```python
from kindling import Engine

# Adaptive boosting (auto-calibrated)
engine = Engine(layered_scoring=True).fit(train)
recs = engine.recommend(entity_id=u, n=10)
# Inspect calibrator's pick:
print(engine.layered_config)            # LayeredConfig(z_threshold=2.5, ...)
print(engine._layered_calibration)      # CalibrationResult with grid trace

# Or pass an explicit config to skip auto-calibration
from kindling.blend.layered import LayeredConfig
engine = Engine(
    layered_scoring=True,
    layered_config=LayeredConfig(z_threshold=2.5, boost_multiplier=5.0),
).fit(train)
```

End-to-end engine validation:

| dataset | Bayesian blend | layered (auto) | delta |
|---|---:|---:|---:|
| grocery-deep | 0.3197 | 0.3198 | +0.04% (parity) |
| amazon-beauty | 0.0201 | 0.0308 | **+53.5%** |

amazon-beauty's blend-catastrophe → layered-rescue is reproducible
via the engine API directly. Calibration adds 0.2-12s to fit time
(negligible vs total).

Bayesian blend is **not deprecated**. It remains the right choice
when calibrated credible intervals matter (uncertainty surfacing,
A/B testing, downstream consumers that need posterior moments).
Layered is the right choice when ranking quality + robustness
matter more than uncertainty quantification.

### 4.5 Headline numbers — full-data, 500 eval entities

`ADR-scoring-architecture.md`:

**grocery-deep** (162k interactions):

| method | NDCG | Recall@10 | MRR | fit s |
|---|---:|---:|---:|---:|
| **bayesian** | **0.3197** | **0.4512** | 0.3514 | 8.5 |
| gating | 0.3029 | 0.4183 | 0.3540 | 27.8 |
| rrf | 0.3025 | 0.4213 | 0.3459 | 8.5 |

**ml1m** (1M interactions):

| method | NDCG | Recall@10 | MRR | fit s |
|---|---:|---:|---:|---:|
| bayesian | 0.2880 | 0.0465 | **0.4556** | 363 |
| **gating** | **0.2911** | 0.0488 | 0.4532 | 1994 |
| rrf | 0.2865 | **0.0513** | 0.4416 | 363 |

---

## 5. Standalone-signal results across 6 datasets

`bench/reports/retriever_matrix_*_v5.json` (or v4 for ml1m / yelp /
amazon-book where pure-count mode means kernel changes are no-ops).

Each signal as both retriever AND ranker (the entire pipeline
collapsed onto one column), full data, 500 eval entities, k=10.

### 5.1 temporal_cooccurrence — the headline result

The newest signal (Apr 2026) wins or ties on 5 of 6 datasets:

| dataset | events/user | cooc NDCG | **temporal_cooc NDCG** | Δ | kernel mode |
|---|---:|---:|---:|---:|---|
| grocery-deep | 108 | 0.319 | **0.322** | +0.9% | hybrid (gmm) |
| ml1m | 142 | 0.288 | **0.300** | **+4.2%** | pure-count (rating-burst) |
| yelp2018 | 39 | 0.037 | **0.038** | +2.7% | pure-count (no timestamps) |
| amazon-book | 46 | 0.026 | 0.026 | 0.0% | pure-count |
| amazon-beauty | 8 | 0.030 | **0.031** | +3.3% | hybrid (gmm) |
| gowalla | 30 | **0.035** | 0.026 | -25.7% ⚠ | hybrid (gmm) |

Two mechanisms drive the lift, working independently:

1. **Hybrid temporal kernel** (`weight = 1 + α · logistic(dt)`,
   default α=1). On real-session data, close-in-time pair contributions
   get up to 2× weight; far pairs decay back to the +1 cooc baseline.
   The +1 baseline preserves candidate-pool coverage on long-time-
   horizon datasets — replacing the prior "weight = logistic(dt)"
   formulation that dropped legitimate cross-time pairs.

2. **Per-user history cap** (default 200 most-recent events). On
   power-rater datasets like ml1m where some users have 500+
   ratings, the cap concentrates pair counts on recent co-ratings,
   acting as implicit recency truncation. This is the entire +4.2%
   lift on ml1m where the kernel itself is auto-disabled.

**Auto-detect logic** (`graph/temporal_interaction.calibrate_kernel`):

- No timestamps → `strategy="pure_count"`, kernel disabled.
- GMM bimodality LLR < 10 → `strategy="pure_count"`, kernel disabled.
- GMM midpoint < 300s → `strategy="rating_burst_detected"`, kernel
  disabled. The "session structure" is rating-burst UI ordering, not
  consumption adjacency. (ml1m midpoint 87s falls here.)
- Otherwise → `strategy="gmm"`, hybrid kernel active.

**The gowalla outlier** (-26% NDCG) is honest about a domain-specific
limitation: check-in temporal proximity reflects *geographic locality*
(visiting nearby places the same day), not *taste correlation*. The
candidate-pool defect is fixed (R@B 0.335 → 0.465 matches cooc 0.467
under the hybrid kernel), but the kernel's prior that "close-in-time
pairs are more informative" is the wrong prior here. Future work:
either per-domain α tuning or a held-out kernel-vs-no-kernel
auto-tuner at fit time.

### 5.2 Full per-dataset standalone tables

**grocery-deep:**

| signal | NDCG | Recall@10 | Recall@budget | p95 ms |
|---|---:|---:|---:|---:|
| **temporal_cooccurrence** | **0.322** | **0.753** | 1.000 | 0.6 |
| item_item_cosine | 0.320 | 0.742 | 1.000 | 0.1 |
| cooccurrence | 0.319 | 0.738 | 1.000 | 0.2 |
| persona | 0.315 | 0.751 | 1.000 | 0.2 |
| path_basket | 0.304 | 0.732 | 1.000 | 12.5 |
| path_tail | 0.181 | 0.474 | 0.996 | 0.1 |
| lightgcn | 0.101 | 0.392 | 0.912 | 0.4 |
| path_full | 0.047 | 0.178 | 0.180 | 0.05 |

**ml1m:**

| signal | NDCG | Recall@10 | Recall@budget | p95 ms |
|---|---:|---:|---:|---:|
| **temporal_cooccurrence** | **0.300** | **0.724** | 0.978 | 1.6 |
| item_item_cosine | 0.292 | 0.706 | 0.978 | 0.9 |
| cooccurrence | 0.288 | 0.712 | 0.974 | 2.6 |
| lightgcn | 0.278 | 0.708 | 0.976 | 0.4 |
| persona | 0.216 | 0.716 | 0.978 | 0.8 |
| path_tail | 0.140 | 0.522 | 0.834 | 0.6 |
| path_basket | 0.076 | 0.322 | 0.878 | 282 |
| path_full | 0.025 | 0.114 | 0.114 | 0.4 |

**yelp2018** (academic split, no timestamps):

| signal | NDCG | Recall@10 | Recall@budget |
|---|---:|---:|---:|
| **temporal_cooccurrence** | **0.038** | 0.214 | **0.890** |
| cooccurrence | 0.037 | 0.220 | 0.882 |
| persona | 0.027 | 0.176 | 0.822 |
| lightgcn | 0.021 | 0.126 | 0.768 |
| path_* | 0.000 | 0.000 | 0.000 |

**amazon-beauty** (5-core JSONL, 178k interactions, 8 events/user):

| signal | NDCG | Recall@10 | Recall@budget |
|---|---:|---:|---:|
| **temporal_cooccurrence** | **0.031** | 0.090 | **0.392** |
| cooccurrence | 0.030 | 0.090 | 0.356 |
| item_item_cosine | 0.024 | 0.070 | 0.284 |
| persona | 0.020 | 0.062 | 0.340 |
| path_basket | 0.018 | 0.050 | 0.244 |
| path_tail | 0.013 | 0.048 | 0.066 |
| lightgcn | 0.006 | 0.024 | 0.146 |
| path_full | 0.001 | 0.006 | 0.006 |

**amazon-book** (LightGCN academic split, 2.4M, no timestamps):

| signal | NDCG | Recall@10 | Recall@budget |
|---|---:|---:|---:|
| item_item_cosine | **0.048** | **0.236** | **0.846** |
| cooccurrence | 0.026 | 0.142 | 0.718 |
| temporal_cooccurrence | 0.026 | 0.148 | 0.776 |
| path_*, persona | 0.000 | 0.000 | 0.000 |

**gowalla** (SNAP raw check-ins, 5.76M, 30 events/user):

| signal | NDCG | Recall@10 | Recall@budget |
|---|---:|---:|---:|
| **cooccurrence** | **0.035** | **0.132** | **0.467** |
| temporal_cooccurrence | 0.026 | 0.087 | 0.465 |
| path_tail | 0.016 | 0.041 | 0.048 |
| item_item_cosine | 0.015 | 0.062 | 0.351 |
| path_basket | 0.011 | 0.021 | 0.093 |
| path_full | 0.000 | 0.000 | 0.000 |

### 5.3 Deactivated signals

- **`path_full`**: consistently the weakest signal (NDCG 0.000-0.047
  across all datasets). Skipped in `_compute_signal_features` while
  staying in SIGNAL_ORDER for back-compat. Bayesian posterior naturally
  drives weight to 0 over a column of zeros. The matrix harness still
  scores it on demand for diagnostic purposes.

- **`interaction_network`**: built and probed (random walks on the
  temporal graph). Lost to direct cooc on grocery (0.290 vs 0.319) and
  matched cooc on ml1m via pure-count. Walk machinery added latency
  (~70ms p95 vs 0.2ms for direct cooc lookup) without ranking value.
  Module retained at `retrieve/interaction_network.py` for research
  but not wired into the engine's blend.

- **`interaction_neighborhood`**: built with 5 pluggable centrality
  measures (betweenness, pagerank, eigenvector, degree, closeness).
  Probed only on grocery (no winner; betweenness was the *worst*
  centrality, contradicting the proposal's hypothesis). Not wired
  into the engine pending dataset-shape evaluation.

### 5.4 What this means for the architecture

The cooc-dominance pattern from `ADR-signal-audit.md` is no longer
strict: `temporal_cooccurrence` gives the first reproducible NDCG
lift on real recommendation data without changing the blend. The lift
mechanism is data-driven (history cap on dense data, hybrid kernel on
session data) and dataset-shape-aware (auto-detect rating-burst
timestamps).

**Queued work** (in priority order):
1. Per-domain α tuning or held-out kernel-vs-no-kernel auto-tuner —
   gowalla shows the kernel can be wrong even when timestamps look
   real, and a fit-time eval would catch this.
2. Global recency decay on all signals (cosine / ALS / persona /
   cost) — the +4.2% on ml1m from the implicit history cap suggests
   explicit decay would compound across signals.
3. HNSW-over-LightGCN retriever — still the candidate-expansion play
   for cases where temporal_cooc doesn't help.

### 5.1 Per-subsystem fit timings

`bench/reports/retriever_matrix_*_v3.json` `fit_timings_per_fraction`
breaks the engine fit into subsystem-level seconds:

**grocery-deep** (full): item_graph 0.02, tail 0.02, path_tree 0.21,
basket_index 0.57, cosine 0.01, als 0.13, **lightgcn 2.49**,
**persona 0.71**, bayesian_blend 5.04. Total ≈ 8.5s.

**ml1m** (full): item_graph 0.40, tail 0.23, path_tree 2.62,
**basket_index 90.98**, cosine 0.38, als 1.05, **lightgcn 30.06**,
**persona 8.09**, bayesian_blend 86.06. Total ≈ 363s.

Two 90s+ items dominate ml1m: basket_index build and the Bayesian
VI loop. Path-basket's 91s is justified by the eventual basket
similarity and lift it computes; the VI loop is the price of the
calibrated posterior. LightGCN at 30s is reasonable for pure numpy
on a 1M-interaction dataset.

---

## 6. Ranker matrix — why `use_ranker=False` by default

`ADR-retriever-ranker-matrix.md` + `ADR-lightgbm-warm-regime.md`.

LightGBM-with-LambdaRank was tested with three training distributions
(random negatives, retrieved negatives, retrieval-hit-only filter +
blend feature). All three produced *worse* NDCG than the blend on
both datasets. Root cause: degenerate feature space — cooc dominates,
the other signals are nearly redundant once normalized, and LambdaRank
has no signal-distinct directions to learn against.

Default is `use_ranker=False`. The LightGBM/XGBoost/CatBoost adapters
ship as opt-in plugins; they remain the right hook when:
- The user has dense per-item content features outside kindling's
  signal set (genre vectors, embeddings, image features, etc.).
- The retriever pool gets large enough (10k+) that re-ranking with
  a learned model becomes worthwhile vs the blend's linear pass.

---

## 6.5 Signal-pair matrix — every signal as retriever × every signal as ranker

`bench/reports/retriever_matrix_grocery_cross.json` (full data, 500 eval
entities, retrieval budget 500, k=10). Each row is "use signal R to
fetch the candidate pool, then sort by signal K" — the diagonal
matches §5's standalone numbers; off-diagonal cells answer "does
mixing R and K beat either alone?"

**grocery-deep NDCG@10 (rows = retriever, cols = ranker):**

| R \ K | cooc | path_tail | path_full | path_basket | cosine | als¹ | persona | lightgcn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cooccurrence | **0.319** | 0.181 | 0.047 | 0.304 | **0.320** | — | 0.315 | 0.265 |
| path_tail | 0.314 | 0.181 | 0.047 | 0.299 | 0.315 | — | 0.312 | 0.266 |
| path_full | 0.061 | 0.053 | 0.047 | 0.059 | 0.061 | — | 0.060 | 0.049 |
| path_basket | 0.319 | 0.181 | 0.047 | **0.304** | **0.320** | — | 0.315 | 0.265 |
| item_item_cosine | 0.319 | 0.181 | 0.047 | 0.304 | **0.320** | — | 0.315 | 0.265 |
| persona | 0.319 | 0.181 | 0.047 | 0.304 | **0.320** | — | 0.315 | 0.265 |
| lightgcn | 0.315 | 0.204 | 0.048 | 0.307 | 0.317 | — | 0.311 | 0.265 |

¹ ALS retriever was unavailable in this run (cold timing during fit
left `als_factor` empty); the standalone diagonal in `v3` reports has
the ALS standalone NDCG of 0.295.

What the matrix says:

1. **cooc-as-ranker is the load-bearer.** Every retriever paired with
   cooc-ranker lands within 1% of cooc-standalone (0.319). Cosine and
   persona rankers track cooc closely. Translation: once you have a
   reasonable candidate pool, the ranker choice barely matters in
   grocery — cooc / cosine / persona are nearly interchangeable.

2. **path_full retrieval caps the matrix.** Every (path_full, *) row
   tops out at recall@K=0.178 because path_full's retrieval budget
   is too sparse to populate the pool — 80%+ of candidates aren't
   even surfaced. The ranker can't recover what wasn't retrieved.

3. **lightgcn retrieval brings new candidates.** lightgcn-retrieve +
   path_tail-rank = NDCG 0.204, which is *higher* than path_tail-
   standalone (0.181). lightgcn pulls candidates path_tail's
   prefix-only retrieval misses, and path_tail successfully picks
   the right ones from that wider pool. This is the cleanest
   evidence in the matrix that retrieval-stage diversification
   helps even when the ranker is weaker.

4. **The cooc / cosine / persona / path_basket retrievers are
   interchangeable** — every row of those four is identical to the
   cooc row across all rankers. Their candidate pools overlap so
   heavily that the retriever choice doesn't move the needle.

The `--cross` flag on `kindling.benchmarks.retriever_matrix` produces
the full 8×8 grid; the v4 reports (`*_cross.json`) sit alongside the
v3 standalone reports.

**ml1m NDCG@10** (`bench/reports/retriever_matrix_ml1m_cross.json`):

| R \ K | cooc | path_tail | path_full | path_basket | cosine | persona | lightgcn |
|---|---:|---:|---:|---:|---:|---:|---:|
| cooccurrence | 0.288 | 0.161 | 0.024 | 0.116 | **0.293** | 0.226 | 0.277 |
| path_tail | 0.244 | 0.140 | 0.025 | 0.099 | 0.257 | 0.185 | 0.234 |
| path_full | 0.027 | 0.027 | 0.025 | 0.025 | 0.028 | 0.027 | 0.028 |
| path_basket | 0.233 | 0.117 | 0.015 | 0.076 | 0.232 | 0.163 | 0.227 |
| item_item_cosine | 0.288 | 0.160 | 0.024 | 0.092 | 0.292 | 0.226 | 0.278 |
| persona | 0.240 | 0.144 | 0.022 | 0.102 | 0.248 | 0.216 | 0.232 |
| lightgcn | 0.288 | 0.162 | 0.024 | 0.116 | **0.293** | 0.225 | 0.277 |

ml1m differs from grocery in two informative ways:

1. **The retriever choice matters on ml1m**, unlike grocery. On
   grocery every retriever paired with cooc-rank lands within 1% of
   cooc-standalone. On ml1m, persona-retrieve + cooc-rank is
   0.240 vs cooc-retrieve + cooc-rank's 0.288 — a 17% drop.
   Translation: ml1m's retrievers pull genuinely different
   candidate pools; grocery's largely overlap.

2. **The best ml1m cell beats every standalone**. The top cells:

   | retriever | ranker | NDCG | R@K | MRR |
   |---|---|---:|---:|---:|
   | **lightgcn** | **item_item_cosine** | **0.2932** | 0.710 | 0.455 |
   | cooccurrence | item_item_cosine | 0.2927 | 0.708 | 0.454 |
   | item_item_cosine | item_item_cosine | 0.2919 | 0.706 | 0.453 |
   | cooccurrence | cooccurrence | 0.2877 | 0.712 | 0.456 |

   The top NDCG is a **lightgcn-retrieve / cosine-rank pair** that
   edges out cosine-standalone by 0.4%. Small but the right shape:
   diversification at the retrieval stage + precision at the
   ranking stage, beating any single-signal pipeline. cooc-retrieve
   + cosine-rank is statistically tied (0.2927).

   This is the cleanest in-pipeline evidence yet that **separating
   retriever and ranker yields a real lift on signal-diverse data**
   — confirming the gating result in §4.4 and providing a concrete
   alternative to the gate (just pick the best `(R, K)` pair from
   the matrix).

3. **path_full retrieval is still the bottleneck** — every
   (path_full, *) cell stuck at recall@K ≈ 0.10 on ml1m, just like
   grocery. The path_full sparsity is dataset-independent.

---

## 7. Re-rank stack (post-blend)

`ADR-phase4-temperature-solver.md`. After the blend produces a
ranked candidate list, the rerank stack applies, in order:

1. **Constraint filtering** — pluggable predicates, applied at
   retrieval output to avoid wasting ranker compute (PRD departure).
2. **DPP greedy MAP** — diversity injection via cosine kernel,
   `diversity_weight=0` falls back to argmax.
3. **Per-position temperature** — beam search (default beam=10,
   chosen for NDCG dominance over greedy and DPP-with-position-quality
   in `ADR-phase4-temperature-solver.md`). Accepts scalar, array,
   profile string, or per-position dict.
4. **Calibration re-rank** (Steck 2018) — when categorical
   metadata is present.
5. **Lift emphasis** — population baseline cached per retrain.

Per-position temperature is implemented in `rerank/temperature.py`.
Reproducibility: identical (seed, data, query) → identical list.

### 7.1 Repeat-consumption module

`ADR-repeat-consumption.md`. Four patterns per (entity, item):

- `REPEAT`: stable interval (e.g. weekly purchase). Period × CV ≈ 1.0.
- `REPLENISH`: stable but with longer interval (e.g. shampoo). CV ≈ 0.5.
- `SATIATION`: declining interest after consumption.
- `ONE_SHOT`: consume once, never again.

The module decomposes period (KDE-detected) × shape (Exponential
prototype) into a per-pair pattern. The `ONE_SHOT` multiplier became
an additive log-penalty (`scores + log(max(multiplier, 1e-20))`)
because z-scored blend scores are negative — multiplicative
suppression was buggy on the negative side.

---

## 8. Persistence & power-user surface

- Versioned binary format (msgpack with schema version) for core
  state — graphs, indexes, posterior params, orthogonalization basis,
  outcome log reference.
- Pluggable components save as a manifest of qualified-name +
  config dict. Load requires `Engine.load(path, registry={...})`.
- User closures (lambda constraints) save with a warning that they
  cannot be restored.
- Power-user properties on `Engine` (read-only): `item_graph`,
  `cost_graph`, `path_tree`, `tail_index`, `basket_index`,
  `communities`, `feature_importance`, `data_density`,
  `drift_report`, `posterior_summary`, `_gate`, `_lightgcn`.

Gate state and LightGCN factors round-trip via `getattr` fallbacks
in `persist/format.py` to preserve backward compatibility with
older snapshots.

---

## 9. Lifecycle: drift, decay, pruning

- Pruning: decay-based retention (default), adaptive
  drift-informed, fixed-window. Preserved aggregates fed back into
  the VI posterior-variance calculation so prune+refit produces a
  posterior within ε of the no-prune baseline.
- Drift report: item-graph Spearman + community ARI (defaults),
  path KL + basket JS (optional). Drift threshold bootstraps off
  the first stable retrain — concerning-drift = 3× the lag-30d
  baseline.
- Outcome log: SQLite-backed append-only with `(entity_id,
  recommendation_id, item_id)` dedup. Late outcomes accepted via
  `report_outcome_correction(...)`.

---

## 10. Decisions log — short index

| ADR | decision |
|---|---|
| signal-audit | only_cooc ≈ full blend; ceiling is candidates not blender |
| score-normalization | shipped 4-mode; default `none` pending priors re-tune |
| scoring-architecture | Bayesian default; gating opt-in; RRF measurement-only |
| persona-signal | HDBSCAN + UMAP; log1p L2-norm cold-start |
| lightgcn-numpy | **end-to-end gradient through K layers (Apr 2026)** — two-stage shortcut abandoned after yelp collapse |
| repeat-consumption | additive log-penalty for ONE_SHOT |
| rating-aware-signals | rating → `_interaction_weight` flows through cooc/path/cost |
| retriever-ranker-matrix | default `use_ranker=False` |
| phase3-default-likelihood | listwise calibration default |
| phase4-temperature-solver | beam-10 default |
| phase8-rust-decision | Rust deferred; Python+NumPy meets PRD perf targets |
| phase7-cross-dataset | extended loader suite; cross_dataset benchmark harness |

---

## 11. Where the seams are (queued work)

In rough priority order:

1. **HNSW-over-LightGCN retriever** — biggest expected lift. Adds
   candidates outside the cooc graph's reach.
2. **Re-tune `priors.toml` for normalized scale** — current priors
   bake in raw-magnitude cooc dominance. Re-tuning under z-score
   normalization could let the gating architecture become the
   strict default.
3. **Cross-dataset benchmarks** at scale — gowalla / yelp /
   amazon / tafeng / dunnhumby loaders ship now; data is
   user-provided. ADR locks architectural conclusions.
4. **Outcome feedback to the Bayesian posterior** — the
   replay-determinism path is wired; the loop closure isn't.
5. **Per-stage signal override** — let users say "use cosine as
   retriever, blend as ranker" or "skip path_full for this query".
   The signal-pair matrix (§6.5) demonstrates this is not just an
   API ergonomics request: lightgcn-retrieve + cosine-rank wins on
   ml1m, so the default policy could route to that pair on
   ml1m-shaped data.

---

## 12. Public API quick-reference

```python
from kindling import Engine
from kindling.gate import GatingConfig
from kindling.lightgcn import LightGCNConfig
from kindling.personas import PersonaConfig

engine = Engine(
    use_ranker=False,                   # default; ship the blend
    signal_normalization="none",        # default; "zscore" for gate
    gating_config=None,                 # opt-in gating
    persona_config=PersonaConfig(...),
    lightgcn_config=LightGCNConfig(dim=64, n_epochs=10),
    basket_similarity="coverage",
    max_history_for_recommend=200,
)
engine.fit(interactions_df)
recs = engine.recommend(entity_id=u, n=10)
for rec in recs:
    print(rec.item_id, rec.score, rec.credible_interval, rec.explanation)
engine.save("model.bin")
loaded = Engine.load("model.bin", registry={...})
```

CLI benchmarks:

```sh
# Three-architecture comparison (Bayesian / gating / RRF) on a single dataset
python -m kindling.benchmarks.scoring_architecture --dataset movielens-1m

# Per-fraction signal matrix; --cross adds full 8x8 retriever x ranker grid
python -m kindling.benchmarks.retriever_matrix \
    --dataset synthetic-grocery-deep --fractions 1.0 --cross \
    --output bench/reports/retriever_matrix_grocery_cross.json

# All datasets in ~/.cache/kindling/<name>/, three architectures each;
# missing datasets are skipped gracefully with structured reasons.
python -m kindling.benchmarks.cross_dataset \
    --output bench/reports/cross_dataset_architecture.json
```

---

*Last updated: 2026-04-24. For the full chain of evidence behind any
decision here, the corresponding ADR in `bench/reports/` is the
source of truth.*
