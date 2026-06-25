# PRD: Force Projection Recommender (FPR) — Ablation Benchmark

**System name:** Force Projection Recommender
**Status:** Draft for experimental run
**Owner:** R. Hoekstra (Awry Labs / kindling)
**Type:** Offline experiment + benchmark (not a production launch)
**Scope:** Language- and implementation-agnostic. All stages are specified as data contracts and operations, not in any particular language, library, or framework.

---

## 1. Summary

The **Force Projection Recommender** is a hypothesized recommender pipeline that builds a cooccurrence graph over items, normalizes its weights, projects it into a low-dimensional space, densifies the long tail using metadata-similarity-derived edges, and serves recommendations by proximity in the projected space. Each of those is a *design decision*, and we do not yet know which decisions earn their cost.

This experiment isolates each decision as a comparison arm against simpler alternatives, measures marginal contribution under a strict temporal evaluation, and reports accuracy, beyond-accuracy quality, stability, and compute cost so that the final pipeline includes only the steps that pay for themselves. Decision rules are **pre-registered** before any arm is scored, to prevent fitting the design to the metric.

The default prior going in: the simplest credible baseline (item-similarity retrieval on popularity-normalized cooccurrence) is strong, and several of the more elaborate steps must clear a meaningful bar to justify inclusion.

---

## 2. Problem Statement

We are designing a hybrid recommender that treats cooccurrence as the primary signal and metadata as a calibrated cold-start patch. The candidate pipeline has at least six independent design decisions, each plausibly helpful and each adding build cost, serving cost, non-determinism, or failure modes. Building the full pipeline and shipping it is the wrong move: we would not know which parts help, which are inert, and which actively hurt long-tail coverage or popularity balance. We need attribution per decision before committing to architecture.

---

## 3. Goals

1. **Attribute marginal value per decision.** For each design choice, quantify its effect on recommendation quality relative to the next-simpler alternative, with confidence intervals and significance.
2. **Establish the simplest sufficient pipeline.** Identify the minimal set of steps that captures most of the achievable quality, and the point of diminishing returns.
3. **Surface decisions that help globally but hurt a segment** (or vice versa) — e.g., a step that lifts head-item accuracy while collapsing tail coverage.
4. **Price every decision.** Report build cost, serving latency, and operational complexity (determinism, incremental-update support) as first-class outcomes alongside accuracy.
5. **Produce a defensible recommendation** for the production pipeline, grounded in pre-registered decision criteria rather than post-hoc metric selection.

---

## 4. Non-Goals

1. **Production serving infrastructure.** This is an offline benchmark. No online A/B, no live traffic, no latency SLA beyond comparative measurement.
2. **Online/interactive metrics.** We do not measure live engagement, click-through, or revenue. Offline rank metrics are explicitly a proxy and their transfer to online behavior is out of scope here.
3. **Recommender UX, explanation surfaces, or diversity-injection policy.** We measure properties that would inform those, but designing them is separate.
4. **Hyperparameter exhaustion.** Each method gets a reasonable, documented tuning budget. We are comparing *decisions*, not finding each method's global optimum. (Tuning protocol is fixed and equal across arms; see §8.)
5. **Sequence/temporal-order modeling.** Cooccurrence is treated as set co-membership within a defined window. Directed/sequential models are a separate investigation.

---

## 5. Reference Pipeline (Parameterized)

The pipeline under test, with each stage's decision variable named. The experiment varies these; it does not assume any particular setting is correct.

| # | Stage | Operation (abstract) | Decision variable |
|---|-------|----------------------|-------------------|
| 1 | Cooccurrence graph | Build weighted undirected item–item graph from co-consumption within window `W` | `cooc_window` |
| 2 | Weight normalization | Transform raw co-counts to tame popularity/heavy tails | `weight_transform` |
| 3 | Projection | Map normalized graph to point coordinates in `d` dimensions | `projection_method`, `dim` |
| 4 | Metadata embedding | Independent content vector per item | (held fixed; input) |
| 5 | Overlap → cooc effect | Estimate relationship: observed cooc strength as a function of metadata similarity | `imputation_model` |
| 6 | Edge imputation | For low-occurrence items with high metadata overlap, synthesize cooc edges weighted by estimated effect | `imputation_mode` |
| 7 | Re-projection | Re-run stage 3 on the augmented graph | (inherits `projection_method`) |
| 8 | User position | Reduce a user's seen-item set to a query representation | `user_model` |
| 9 | Ranking | Score unseen items against the user query representation | `scoring_model`, `debias_mode`, `cold_path` |

**Stage contracts (so this is implementable in any language):**

- **Cooc graph:** input is an interaction log `(user, item, timestamp)`; output is a sparse symmetric weighted adjacency over items.
- **Weight transform:** pure function `raw_weight, item_marginals → normalized_weight`.
- **Projection:** input sparse weighted graph; output `item → vector(d)`. Must accept a random seed and emit it.
- **Imputation model:** input is `(metadata_similarity, observed_cooc)` pairs over item pairs that *do* co-occur; output is a function `metadata_similarity → predicted_cooc` plus a scalar predictive-confidence measure.
- **Edge imputation:** input is the predicted-cooc function, the metadata-similarity matrix, and per-item occurrence counts; output is a set of synthetic edges with weights and a `provenance = synthetic` flag.
- **User model:** input is the user's seen-item vectors; output is one or more query points or the seen set itself.
- **Scoring:** input is user query representation + candidate vectors; output is a ranked candidate list.

---

## 6. Decision Variables and Arms

Each variable below is a comparison axis. The level marked **(baseline)** is the simpler/cheaper default that more elaborate levels must beat.

### 6.1 `weight_transform` (Stage 2)
- **raw** — untransformed co-counts. *(floor — included to show the tail problem is real.)*
- **log** — log of co-counts. **(baseline)**
- **ppmi** — positive pointwise mutual information (popularity-normalized).
- Optional: **jaccard / cosine** over co-counts.

Hypothesis H1: popularity-normalized weights (`ppmi`) beat `log`, which beats `raw`, on tail-segment metrics, because raw and log leave blockbuster-co-occurs-with-everything bias intact.

### 6.2 `projection_method` + `dim` (Stage 3)
- **none** — operate directly on the sparse normalized graph / item-similarity (no projection). **(baseline)**
- **force_directed** — spring-electrical layout (e.g., multilevel force projection).
- **neighbor_embedding** — negative-sampling neighbor embedding (the UMAP/node2vec family).
- **spectral** — eigenmap embedding.
- `dim` ∈ {2, low-tens, low-hundreds} where the method supports it.

Hypothesis H2: a projection is only worth its cost if it beats `none` on the primary metric *under the same ANN/index conditions*; and any benefit is dimension-sensitive — 2D underperforms for retrieval (crowding), very high `d` underperforms (distance concentration), with an interior optimum.

### 6.3 `imputation_mode` (Stages 5–6)
- **off** — no synthetic edges. **(baseline)**
- **naive** — add edges wherever metadata similarity exceeds a threshold, unweighted.
- **calibrated** — add edges weighted by the estimated metadata→cooc effect (Stage 5 regression).
- **confidence_weighted** — calibrated, but synthetic edge weights additionally scaled by the regression's predictive confidence, and kept numerically distinct from observed edges.

Hypothesis H3: imputation helps tail/cold items but the benefit is gated by how well metadata predicts cooc (Stage 5 effect size); `naive` risks collapsing the tail into content-only filtering and will underperform `confidence_weighted` on serendipity/novelty while possibly tying on raw accuracy.

### 6.4 `scoring_model` + `user_model` (Stages 8–9)
- **centroid** — mean of seen-item vectors; rank by distance from that single point. **(the proposed design — treated as a comparison arm, not the assumed winner.)**
- **nearest_seen** — rank candidate by aggregate similarity (max or softmax-weighted top-`m`) to the user's seen items (item-based CF in the space). **(baseline)**
- **mixture** — cluster the seen set; score against nearest cluster centroid / mixture.

Hypothesis H4: `nearest_seen` and `mixture` beat `centroid` for multi-interest users, because a single centroid lands between interest clusters and ranks bland in-between items. This is also the arm expected to interact most strongly with `projection_method` (a projection preserves *local* neighborhood order but distorts *global* distance, so a global-distance query like `centroid` is the worst case for any projection).

### 6.5 `debias_mode` (Stage 9)
- **off** — no popularity correction. **(baseline)**
- **ipw** — inverse-propensity weighting on seen items when forming the user representation.
- **density_penalty** — down-weight candidates by local density in the space.

Hypothesis H5: without debiasing, the user representation drifts to the popular core and recommendations amplify popularity (lower coverage, higher Gini), even when top-K accuracy looks healthy.

### 6.6 `cold_path` (Stage 9)
- **off** — cold items ranked only through whatever cooc/synthetic edges exist. **(baseline)**
- **metadata_fallback** — items below an occurrence threshold are ranked directly in metadata space against the user's seen-item metadata profile.

Hypothesis H6: without a fallback, items with neither cooc nor a strong metadata twin are structurally unrecommendable; `metadata_fallback` lifts pure-cold-item recall at little cost to head accuracy.

### 6.7 Configuration Surface — Full Tunable Parameter Inventory

**Principle:** every constant in the pipeline is externalized as a named, bounded, logged parameter. Nothing is hardcoded. A single run is fully described by one config object; that object is persisted with the run's results so any arm is exactly reproducible. Parameters fall into two classes: **swept** (varied across arms — the decision variables in §6.1–6.6) and **fixed-but-tunable** (held constant within the experiment but exposed so they can be tuned per §8.4 or changed without code edits). The harness validates every parameter against its range at load time and rejects out-of-bounds configs.

The decision variables above are the *axes we sweep*; the tables below are the *complete knob set*, including the lower-level parameters each method exposes.

#### Stage 1 — Cooccurrence graph
| Parameter | Type | Default | Range / Levels | Note |
|---|---|---|---|---|
| `cooc.event_def` | enum | `session` | `session` / `time_window` / `co_owned` | What counts as a co-event |
| `cooc.window` | duration | 1 session | ≥ 0 | Window length when `time_window` |
| `cooc.min_user_interactions` | int | 2 | ≥ 1 | Drop sparse users from graph construction |
| `cooc.min_item_occurrences` | int | 1 | ≥ 1 | Item inclusion floor (distinct from cold threshold) |
| `cooc.min_edge_cocount` | int | 1 | ≥ 1 | Prune edges below this raw co-count |
| `cooc.directed` | bool | false | — | Held false this experiment (see §4 non-goals) |

#### Stage 2 — Cooc weight normalization
| Parameter | Type | Default | Range / Levels | Note |
|---|---|---|---|---|
| `norm.transform` | enum (**swept** §6.1) | `log` | `raw` / `log` / `ppmi` / `jaccard` / `cosine` | Primary normalization choice |
| `norm.log_offset` | float | 1.0 | > 0 | `log(offset + x)` to handle zeros/ones |
| `norm.log_base` | float | e | > 1 | — |
| `norm.ppmi_shift_k` | float | 1.0 | ≥ 1 | Shifted-PMI; `k>1` discounts rare pairs |
| `norm.ppmi_context_smoothing` | float | 0.75 | (0, 1] | Marginal-distribution smoothing exponent |
| `norm.outlier_clip_pctile` | float | 99.0 | (0, 100] | Winsorize weights above this percentile — the explicit high-outlier control |
| `norm.weight_min` / `norm.weight_max` | float | none | — | Hard clamp after transform |
| `norm.symmetrize` | enum | `mean` | `mean` / `max` / `min` | Reconcile asymmetric co-counts |

#### Stage 3 — Projection (method-specific blocks active per `projection.method`)
| Parameter | Type | Default | Range / Levels | Note |
|---|---|---|---|---|
| `projection.method` | enum (**swept** §6.2) | `none` | `none` / `force_directed` / `neighbor_embedding` / `spectral` | — |
| `projection.dim` | int (**swept** §6.2) | 32 | {2, low-tens, low-hundreds} | Embedding dimensionality |
| `projection.seed` | int | logged | any | Required; emitted with results |
| `projection.init` | enum | `spectral` | `random` / `spectral` | Warm-start initialization |
| `projection.align_to_previous` | bool | true | — | Procrustes-align to prior build for churn control |
| **force_directed** | | | | active when method = force_directed |
| `fd.attraction_strength` | float | 1.0 | > 0 | Spring constant along edges |
| `fd.repulsion_strength` | float | 1.0 | > 0 | Electrical/Coulomb constant |
| `fd.ideal_edge_length` | float | auto | > 0 | Optimal distance `K` |
| `fd.gravity` | float | 0.1 | ≥ 0 | Centering force; prevents disconnected drift |
| `fd.barnes_hut_theta` | float | 0.9 | (0, 2] | Repulsion approximation accuracy vs speed |
| `fd.coarsening_levels` | int | auto | ≥ 0 | Multilevel depth |
| `fd.cooling_initial_temp` | float | auto | > 0 | — |
| `fd.cooling_decay` | float | 0.95 | (0, 1) | Step-size decay per iteration |
| `fd.max_iterations` | int | 500 | ≥ 1 | Hard cap |
| `fd.convergence_tol` | float | 1e-4 | > 0 | Early stop on energy delta |
| **neighbor_embedding** | | | | active when method = neighbor_embedding |
| `ne.n_neighbors` | int | 15 | ≥ 2 | Local vs global balance |
| `ne.min_dist` | float | 0.1 | [0, 1) | Cluster tightness — interacts with serendipity (§6.3) |
| `ne.negative_sample_rate` | int | 5 | ≥ 1 | Repulsion strength |
| `ne.learning_rate` | float | 1.0 | > 0 | — |
| `ne.epochs` | int | auto | ≥ 1 | — |
| `ne.set_op_mix_ratio` | float | 1.0 | [0, 1] | Union↔intersection of local graphs |
| **spectral** | | | | active when method = spectral |
| `sp.laplacian` | enum | `sym` | `sym` / `rw` / `unnormalized` | — |
| `sp.n_components` | int | = `dim` | ≥ 1 | Eigenvectors retained |

#### Stage 5 — Overlap → cooc effect model
| Parameter | Type | Default | Range / Levels | Note |
|---|---|---|---|---|
| `effect.model_form` | enum | `monotone` | `linear` / `monotone` / `nonparametric` | Functional form of metadata-sim → cooc |
| `effect.regularization` | float | auto | ≥ 0 | — |
| `effect.fit_sample` | enum | `co_occurring_pairs` | — | Pairs the effect is estimated on |
| `effect.confidence_metric` | enum | `cv_r2` | `cv_r2` / `holdout_corr` | Scalar trust used downstream |

#### Stage 6 — Edge imputation thresholds (the "when to apply embedding adjustments" controls)
| Parameter | Type | Default | Range / Levels | Note |
|---|---|---|---|---|
| `impute.mode` | enum (**swept** §6.3) | `off` | `off` / `naive` / `calibrated` / `confidence_weighted` | — |
| `impute.cold_occurrence_threshold` | int | tunable | ≥ 0 | Item is imputation-eligible only below this occurrence count |
| `impute.metadata_sim_threshold` | float | tunable | [0, 1] | Minimum metadata similarity to create a synthetic edge |
| `impute.confidence_floor` | float | tunable | [0, 1] | If Stage-5 confidence is below this, **impute nothing** (the global gate) |
| `impute.max_synthetic_edges_per_item` | int | tunable | ≥ 0 | Top-N degree cap; prevents over-densification into content-only filtering |
| `impute.weight_scale` | float | 1.0 | ≥ 0 | Global multiplier on the calibrated effect |
| `impute.synthetic_weight_cap_ratio` | float | tunable | (0, 1] | Cap synthetic weight relative to max observed weight |
| `impute.provenance_flag` | bool | true | — | Tag synthetic edges for separate evaluation (§9 segmentation) |

#### Stage 8 — User representation
| Parameter | Type | Default | Range / Levels | Note |
|---|---|---|---|---|
| `user.model` | enum (**swept** §6.4) | `nearest_seen` | `centroid` / `nearest_seen` / `mixture` | — |
| `user.top_m` | int | 10 | ≥ 1 | Neighbors aggregated when `nearest_seen` |
| `user.softmax_temp` | float | 1.0 | > 0 | Sharpness of similarity weighting |
| `user.recency_decay` | float | 0.0 | ≥ 0 | Down-weight older seen items |
| `user.n_clusters` | int/auto | auto | ≥ 1 | Mixture components when `mixture` |
| `user.min_seen` | int | 1 | ≥ 1 | Below this, route to cold/popularity path |

#### Stage 9 — Scoring, debias, cold path
| Parameter | Type | Default | Range / Levels | Note |
|---|---|---|---|---|
| `score.K` | int | tunable | ≥ 1 | Recommendation list length (eval at multiple K) |
| `score.distance_metric` | enum | `cosine` | `cosine` / `euclidean` / `dot` | In projected space |
| `score.debias_mode` | enum (**swept** §6.5) | `off` | `off` / `ipw` / `density_penalty` | — |
| `score.ipw_propensity_clip` | float | 0.05 | (0, 1) | Clip extreme propensities |
| `score.density_penalty_strength` | float | 1.0 | ≥ 0 | — |
| `score.cold_path` | enum (**swept** §6.6) | `off` | `off` / `metadata_fallback` | — |
| `score.cold_fallback_threshold` | int | tunable | ≥ 0 | Occurrence count below which fallback ranks the item |
| `score.cold_blend_weight` | float | tunable | [0, 1] | Blend of cooc-space score vs metadata-fallback score |

#### Harness / evaluation parameters (fixed within a run, exposed for the protocol)
| Parameter | Type | Default | Range / Levels | Note |
|---|---|---|---|---|
| `eval.temporal_cutoffs` | list | tunable | ≥ 3 | Rolling holdout points (§8) |
| `eval.seeds` | int `R` | tunable | ≥ 1 | Re-runs for stochastic arms (churn/variance) |
| `eval.ann_index` | enum | `hnsw` | `hnsw` / `ivf_pq` / `exact` | Primary metric scored under this (§9.1) |
| `eval.ann_params` | block | — | e.g. HNSW `M`,`ef_construction`,`ef_search`; IVF `nlist`,`nprobe`,`pq_m` | Quantization is part of the measured condition |
| `eval.tuning_budget` | int | equal/arm | ≥ 0 | Identical across arms (§8.4) |

**Config object sketch (language-agnostic, e.g. a serialized map):**
```
run_id, seed
cooc:   { event_def, window, min_user_interactions, min_item_occurrences, min_edge_cocount, directed }
norm:   { transform, log_offset, log_base, ppmi_shift_k, ppmi_context_smoothing, outlier_clip_pctile, weight_min, weight_max, symmetrize }
projection: { method, dim, seed, init, align_to_previous, fd:{...}, ne:{...}, sp:{...} }
effect: { model_form, regularization, fit_sample, confidence_metric }
impute: { mode, cold_occurrence_threshold, metadata_sim_threshold, confidence_floor,
          max_synthetic_edges_per_item, weight_scale, synthetic_weight_cap_ratio, provenance_flag }
user:   { model, top_m, softmax_temp, recency_decay, n_clusters, min_seen }
score:  { K, distance_metric, debias_mode, ipw_propensity_clip, density_penalty_strength,
          cold_path, cold_fallback_threshold, cold_blend_weight }
eval:   { temporal_cutoffs, seeds, ann_index, ann_params, tuning_budget }
```
The set of parameters that *change* between two configs is, by definition, the experimental difference being measured — so diffing two run configs yields the exact attribution for any pairwise comparison.

---

## 7. Experimental Design

### 7.1 Strategy: staged ablation, not full factorial
A full grid over all six variables is combinatorially large and most cells are uninteresting. Instead:

1. **Establish baselines** (§7.2).
2. **One-factor-at-a-time (OFAT)** from a fixed reference configuration: vary each decision variable alone, holding the rest at baseline. This gives clean marginal-effect estimates.
3. **Targeted interaction checks** for the pairs with a prior reason to interact:
   - `projection_method` × `scoring_model` (global-distance scoring vs projection distortion).
   - `imputation_mode` × `weight_transform` (synthetic edges interact with how observed weights are scaled).
   - `imputation_mode` × `cold_path` (both target the tail; confirm they are complementary, not redundant).
4. **Best-of-breed composite:** assemble the winning level of each variable and confirm it beats the best single baseline by the pre-registered margin. (Guards against the case where individually-good decisions do not stack.)

Rationale for OFAT-plus-interactions over full factorial: it is interpretable, cheap, and sufficient to attribute marginal value, which is the goal. The cost is missing higher-order interactions; we accept that and note it as a limitation.

### 7.2 Baselines (the bar to clear)
- **B0 — Popularity:** non-personalized most-popular-unseen. The floor; any personalized arm must beat it decisively.
- **B1 — Item-kNN on normalized cooc:** no projection, no metadata, `ppmi` weights, `nearest_seen` scoring. The strong, boring baseline.
- **B2 — Metadata-embedding kNN:** content-only retrieval, ignores cooc entirely.
- **B3 — Latent-factor model:** implicit-feedback matrix factorization over the interaction matrix.

A complex arm "wins" only if it beats **B1** (not just B0) by the pre-registered margin.

---

## 8. Data, Splits, and Tuning Protocol

1. **Temporal holdout, not random split.** Train on interactions before cutoff `T`; evaluate on interactions after `T`. Random splits leak future cooccurrence into training and inflate every arm. This is non-negotiable and applied identically to all arms.
2. **Multiple cutoffs.** Use ≥3 rolling cutoffs to get variance across time, not a single lucky split.
3. **Cold-item construction.** Explicitly designate an evaluation subset of items with zero or near-zero training cooc but real post-cutoff interactions, so the `imputation_mode` and `cold_path` arms have something legitimate to be measured on. Without this, their value is invisible.
4. **Tuning budget.** Each method gets an equal, documented hyperparameter budget tuned on a validation slice *inside* the training period (never on the holdout). Report the chosen settings. Equal budget across arms is required for fair comparison.
5. **Determinism.** Every stochastic stage (notably any projection) takes and logs a seed. Stochastic arms are run with ≥`R` seeds and reported with across-seed variance.

---

## 9. Metrics

All metrics reported **globally and segmented** by (a) item popularity tier — head / torso / tail / cold — and (b) user activity level. Aggregate-only reporting will hide the exact effects this experiment exists to find.

### 9.1 Primary (pre-registered single decision metric)
- **nDCG@K** on the temporal holdout, evaluated **under the production-representative ANN/index conditions** (i.e., approximate retrieval with the intended quantization, not exact vectors). Rationale: ranking is what matters, and approximate-index recall — not exact-distance fidelity — is the real ceiling, so the metric must reflect it.

### 9.2 Secondary accuracy
- Recall@K, MAP@K, at multiple K.

### 9.3 Beyond-accuracy
- **Catalog coverage** — fraction of items ever recommended.
- **Popularity bias** — Gini of recommendation frequency; mean popularity of recommended items.
- **Novelty / serendipity** — self-information of recommendations; surprise relative to a popularity baseline.
- **Cold-item recall** — recall restricted to the designated cold subset (the headline metric for imputation/cold-path arms).

### 9.4 Stability
- **Rank churn** — Jaccard overlap of top-K for the same user across re-runs with different seeds (and across adjacent rebuilds). Directly prices the non-determinism of projection-based arms; a high-accuracy arm with high churn is an operational liability.

### 9.5 Cost (first-class, not a footnote)
- Build/preprocessing time and memory per arm.
- Serving query latency (p50/p95) under the chosen index.
- Incremental-update support: can a new item be added without a full rebuild? (categorical: yes / warm-start / full-rebuild-only.)

---

## 10. Statistical Methodology

1. **Uncertainty:** bootstrap over users (and over temporal cutoffs) for confidence intervals on every reported metric. Report effect sizes, not just p-values.
2. **Significance:** paired tests across the per-user metric distribution between an arm and its baseline.
3. **Multiple comparisons:** with many arms × metrics, apply a correction (e.g., Holm) to the primary-metric comparisons. State the family of tests up front.
4. **Pre-registration:** §9.1 primary metric, §11 decision rule, and the arm list in §6 are fixed before scoring. Any post-hoc metric is labeled exploratory and cannot drive the inclusion decision.

---

## 11. Decision Criteria (Pre-Registered)

A decision variable's elaborate level is adopted for the recommended pipeline **only if all hold**:

1. It beats the next-simpler level (and B1) on the **primary metric** by a margin of at least **Δ** (set before scoring), with corrected significance.
2. It does **not** regress any **guardrail metric** beyond its threshold: catalog coverage, popularity-bias Gini, and cold-item recall each have a pre-set maximum tolerated degradation.
3. Its **cost** (build + serving + operational complexity) is within the pre-set budget, or the accuracy gain exceeds a documented cost-justification threshold.

If an elaborate level wins accuracy but trips a guardrail or cost ceiling, it is **rejected** and the rejection is recorded with the trade-off that killed it. Ties go to the simpler/cheaper/more-deterministic arm.

---

## 12. Deliverables

1. **Reproducible experiment harness** implementing the stage contracts in §5, configurable over the arms in §6, runnable end-to-end from interaction log to scored metrics. Language/implementation is unconstrained; the contracts are the interface.
2. **Results table:** every arm × every metric × every segment, with CIs.
3. **Marginal-effect summary:** per decision variable, the measured lift and its cost, with the adopt/reject call and the reason.
4. **Recommended pipeline:** the composite of adopted levels, with the §7.1 step-4 confirmation that it beats the best single baseline.
5. **Limitations note:** offline-to-online transfer caveats, OFAT's missed higher-order interactions, and any segment where results were inconclusive.

---

## 13. Risks and Mitigations

| Risk | Effect | Mitigation |
|------|--------|------------|
| Temporal split done wrong | All arms inflated; comparisons still biased toward leakage-friendly methods | Strict pre-cutoff training; audit for any future information in features (incl. metadata derived post-`T`) |
| Offline metric ≠ online behavior | Adopt a pipeline that wins offline and loses live | Treat result as a *ranking of decisions*, not an absolute promise; flag as input to a later online test |
| Exact-vector eval hides ANN degradation | Projection arms look better than they serve | Evaluate primary metric under the real index/quantization (§9.1) |
| Cold subset too small | Imputation/cold-path effects statistically invisible | Size the cold subset deliberately during split construction; report its n and power |
| Unequal tuning effort | Favors whichever arm got more love | Fixed, equal, documented tuning budget per arm (§8.4) |
| Metric shopping | Post-hoc story-fitting | Pre-registered primary metric + decision rule (§10–11) |

---

## 14. Open Questions

- **[data]** What defines a co-consumption event and the window `W`? (Same user ever, same session, fixed time window?) This shapes the entire graph and should be fixed before Stage 1.
- **[data]** Is the interaction signal implicit (views/plays) or explicit (ratings)? Determines the latent-factor baseline (B3) variant and the propensity model for `ipw`.
- **[method]** What is the cold-occurrence threshold separating "tail" from "cold," and does it match how the cold evaluation subset is constructed?
- **[method]** Set the pre-registered values: primary-metric margin **Δ**, guardrail degradation tolerances, cost budget, K values, seed count `R`, number of temporal cutoffs.
- **[scope]** Is bipartite (user–item) structure in play, or is this purely item–item cooc? If bipartite, B3 and the scoring arms may need a shared user/item space, which adds an arm.
