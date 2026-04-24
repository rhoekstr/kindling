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

### 2.3 LightGCN — pure-numpy, two-stage

`ADR-lightgcn-numpy.md`. To avoid PyTorch we split:
- **Train stage**: BPR-train base embeddings *without* layer
  propagation in the loop. Hand-computed sigmoid-cross-entropy
  gradients in numpy, Adam-style adaptive LR.
- **Inference stage**: at query time, apply K-layer normalized
  adjacency propagation + layer-mean to the trained base embeddings.

Loses some theoretical unity vs end-to-end propagation but
benchmarks show it lands within ~1-2% NDCG of the paper's results
on grocery and ml1m, and ships without a deep-learning runtime.

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

### 4.4 Headline numbers — full-data, 500 eval entities

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

## 5. Standalone-signal results (the "cooc dominates" finding)

`ADR-standalone-retrievers.md` + `bench/reports/retriever_matrix_*_v3.json`.

Each signal as both retriever AND ranker (i.e. the entire pipeline
collapsed onto one column), full data, 500 eval entities, k=10:

**grocery-deep:**

| signal | NDCG | Recall@10 | MRR | p95 ms |
|---|---:|---:|---:|---:|
| item_item_cosine | 0.3198 | 0.7421 | 0.3554 | 0.11 |
| cooccurrence | 0.3191 | 0.7379 | 0.3510 | 0.07 |
| persona | 0.3155 | 0.7505 | 0.3388 | 0.23 |
| path_basket | 0.3043 | 0.7317 | 0.3413 | 12.5 |
| als_factor | 0.2947 | 0.7568 | 0.3494 | 0.08 |
| lightgcn | 0.2648 | 0.6792 | 0.3091 | 0.06 |
| path_tail | 0.1807 | 0.4738 | 0.2479 | 0.11 |
| path_full | 0.0467 | 0.1782 | 0.0926 | 0.05 |

**ml1m:**

| signal | NDCG | Recall@10 | MRR | p95 ms |
|---|---:|---:|---:|---:|
| item_item_cosine | 0.2919 | 0.706 | 0.4533 | 0.85 |
| cooccurrence | 0.2877 | 0.712 | 0.4561 | 1.43 |
| als_factor | 0.2804 | 0.728 | 0.4504 | 0.39 |
| lightgcn | 0.2774 | 0.706 | 0.4364 | 0.37 |
| persona | 0.2148 | 0.726 | 0.3718 | 0.82 |
| path_tail | 0.1456 | 0.538 | 0.2647 | 0.63 |
| path_basket | 0.0855 | 0.336 | 0.1505 | 282.13 |
| path_full | 0.0266 | 0.128 | 0.0722 | 0.38 |

The pattern: **cooccurrence and item-item cosine are within ~1% of
the full Bayesian blend on both datasets.** The blend's NDCG advantage
over its best single signal is 0.0% (grocery) / -0.4% (ml1m).
`ADR-signal-audit.md` confirmed `only_cooc ≈ full blend`.

What this means for the architecture: **the ceiling isn't a better
blender, it's adding candidates that the cooc graph doesn't reach.**
Queued: HNSW-over-LightGCN-embeddings retriever, real session data
(RetailRocket / Instacart / Amazon — now all loadable), outcome
feedback to the Bayesian posterior.

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

**ml1m cross-matrix is in flight as of this writing**
(`bench/reports/retriever_matrix_ml1m_cross.json`); will be summarized
here once the run completes.

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
| lightgcn-numpy | two-stage train/inference, no PyTorch |
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
