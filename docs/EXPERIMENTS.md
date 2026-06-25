# Experiment Addendum — what was tried, what won, what was retired, and why

> The curated record of kindling's experimental program. It exists so the
> knowledge survives the code: as the consolidation deletes the rejected
> machinery, **this document is what replaces it.** Negative results are
> kept deliberately — they are the fence posts that define where the
> shallow-model philosophy stops paying.
>
> **Reading order.** Part I (methodology) first — it explains why later
> verdicts look the way they do. Part II is the *positive* record (how the
> production engine got its shape). Part III is the *negative* record (the
> rejected experiments). Part IV is the open/closed-as-bounded fronts.
> Part V is the evidence map — every claim here traces to an ADR in
> `bench/reports/ADR-*.md` and/or a frozen result artifact.
>
> **Relationship to other docs.** `REFERENCE.md` is the living architecture
> + synthesis; `bench/reports/` is the deep chain-of-evidence (ADRs +
> frozen metric files), retained in full. This addendum is the curated
> index over both. After consolidation it is the canonical experiment
> record; `REFERENCE.md` slims to the shipped architecture.
>
> **A naming caveat to avoid confusion.** Two `§`-numbering systems exist.
> `kindling_PRD_v08.docx` (the original design vision) numbers sections
> one way; the commit-message tags (`§4.4`, `§7.2`, …) and `REFERENCE.md`
> number them *differently*. Throughout this document, `§N` refers to
> **`REFERENCE.md`**, which the commits track. The docx is historical
> vision, superseded by `REFERENCE.md` for operative truth.

---

## Part I — Methodology (read first)

The verdicts below are only legible against the protocol that produced
them. Three methodological moves matter more than any single result.

### I.1 Canonical (academic) protocol

Chronological 90/10 split; eval users strided-500 by sorted `entity_id`;
**full-catalog ranking** (no sampled negatives); k=10. Builder:
`benchmarks/parity._build_eval_set`. Full-ranking is deliberate and has a
consequence: classic papers (NCF, SASRec) rank against ~100 sampled
negatives and report HR@10 ≈ 0.5; those deflate **~10×** under full
ranking (Krichene & Rendle 2020). Full-ranking SOTA on amazon-beauty is
NDCG@10 ≈ 0.03–0.05 — *the band kindling operates in*. Numbers here are
not comparable to sampled-negative leaderboards.

### I.2 The realistic-protocol tier (why some "dead" layers were never testable)

5-core preprocessing **deletes the cold population** — so academic
benchmarks cannot reward (or even test) content / LLM / cold-start
machinery. A layer flat on a 5-core aggregate is *unproven, not
disproven*. The realistic tier fixes this: **no k-core filtering**,
chronological global split, **segment-sliced reporting** (recovery
bucketed by item warmth: 0 / 1–4 / 5–19 / 20+ train interactions).
Members: `steam` (Kang & McAuley crawl — 7.8M reviews, 2.3M users, 14k
train items, 15% of test events on cold items) and `amazon-book-chrono`.
Several content/cold verdicts flipped between tiers; both are reported.

### I.3 The gap-decomposition diagnostic (run before believing any conclusion)

`benchmarks/gap_decomposition.py` brackets the system between a
**popularity floor**, an **oracle ceiling on the same candidate pool**,
and the **retrieval ceiling** (pool recall). It is the single most
important diagnostic in the project — it is what exposed that the
pre-pivot engine was "a popularity ranker in costume."

| | ml1m | amazon-beauty |
|---|---:|---:|
| popularity floor | 0.2492 | 0.0063 |
| raw-cooc base (pre-pivot) | 0.2561 | 0.0290 |
| oracle on same pool | 0.8845 | 0.2617 |
| pool recall@500 (median) | 0.56 | **0.00** |

Reading: ml1m had 3.5× scoring headroom *inside* the existing pool
(ranking-bound); beauty was broken in retrieval (half its users had zero
held-out items in the pool). The two reference datasets fail for
**different reasons** — a fact that governs every "what should we build
next" decision.

---

## Part II — Architectural decisions (the positive record)

How the production engine got its shape. The build ran in numbered
phases (the original PRD plan), then a 2026-06 diagnostic-driven pivot
replaced the base scorer.

### II.1 The phased build (Phases 1–8)

| Phase | What shipped | ADR / evidence |
|---|---|---|
| 2 | Path mechanisms, session inference, block decorrelation | — |
| 3 | **Bayesian blend** (Dirichlet VI, credible intervals); default-likelihood selection | `ADR-phase3-default-likelihood`, `ADR-scoring-architecture` |
| 4 | DPP diversity, **temperature solver** (per-position), lift emphasis, Steck calibration | `ADR-phase4-temperature-solver` |
| 5 | Cost graph (negative signals), outcome logging/replay | — |
| 6 | Lifecycle: pruning, drift detection, preserved aggregates | — |
| 7 | **Cross-dataset critical-path consolidation** — froze the fixed cross-dataset defaults that per-fit tuning kept losing to | `ADR-phase7-cross-dataset` |
| 8 | **Rust decision**: ship pure-Python for v1.0; Rust core (`kindling_core`) follows as v2 | `ADR-phase8-rust-decision`, `ADR-phase8-rust-evidence` |

Supporting positive decisions:
- **Rating-aware positive signals + centralized preprocessor** — ratings
  become preference *intensity* where real, not just presence
  (`ADR-rating-aware-signals`).
- **Pair-index + distinctiveness-weighted basket signal** — the path-family
  basket channel weights co-occurrence by how *distinctive* a pair is, not
  raw frequency (`ADR-pair-index-distinctiveness`).
- **Repeat-consumption module** — architecture shipped; replenishment
  datasets only (`ADR-repeat-consumption`; calibration still open).

### II.2 The 2026-06 pivot (the turning point)

The gap-decomposition diagnostic (I.3) showed **raw co-occurrence scoring
degenerates toward popularity ranking**: for popular items `cooc[i,c]` is
large for every `i`, so `Σ_{i∈owned} cooc[i,c] ≈ popularity`. The old
base beat a pure popularity ranker by only **+2.7%** on ml1m. This
triggered the replacement of the base scorer and the move to a fused,
z-normalized channel model.

**The fused scoring model that resulted** (the shipped design):

```
score = z(base) + 0.5·z(trend) + 0.25·z(last_item) + 0.25·z(transitions)
        + 1.0·z(user_cf)            [sparse-history datasets only]
```

| Channel | Why it survived (number + dataset) | Activation gate | ADR/§ |
|---|---|---|---|
| **EASE base** | inverse-Gram subtracts the popularity redundancy raw cooc couldn't; rating-weighted +1.8% ml1m / +5.2% beauty | `n_items ≤ 20k`; rating-weighted when `detect_rating_signal=ratings` | §2.1, §4.5 |
| **Wilson cooc base** | removes popularity cheaply; **book +68.5%** NDCG vs raw; ~90% of LightGCN at 86s CPU | `n_items > 20k` | §3.4 |
| **Trend** (0.5) | trend@1.0 0.298 > full blend 0.293 > pop — the one rerank term that beats the base | timestamps present | `ml1m_rerank.json` |
| **Last-item EASE row** (0.25) | ml1m +1.3% NDCG / +1.7% MRR; beauty recall +4.6%; reads structure not order, so *not* burst-gated | EASE present | §4.5 |
| **Transitions** (0.25) | beauty +2.9% NDCG / +7% recall; **hurts ml1m at every weight** | timestamps AND **not** rating-burst | §4.5 |
| **user_cf** (1.0) | beauty +4.5% NDCG; **−1.6% ml1m** (EASE already encodes the neighborhood on dense data) | median history ≤ 20 | §2.2 |
| **Cold slots + open-catalog** (structural) | steam cold-0 recovery 0% → **8.5%**, warm ranking untouched | `cold_slots>0` + metadata | §4.8 |
| **Popularity fallback + EB shrinkage** | ties/beats popularity at all cold buckets; ml1m 1-seed 0.094→0.105 | new/anonymous user; thin seeds | §7.4 |

Progression (ml1m): 0.2561 raw cooc → ~0.269 EASE → 0.2841 +trend →
0.2879 +last-item → **0.2931** rating-weighted. (beauty): 0.0290 →
0.0343 with +EASE/trend/transitions/rating-weight/user_cf.

**Re-captured for the record** (cumulative ablation, frozen to
`bench/reports/channel_ablation_{movielens-1m,amazon-beauty}.json` — this
progression previously lived only in REFERENCE prose / runner stdout):
- **ml1m** reproduces cleanly: 0.2465 raw cooc → 0.2665 EASE → 0.2841
  +trend → 0.2875 +last-item → (+transitions no-op, burst-gated) →
  **0.2928** +rating-weight (= **+1.8%**, confirming the rating-weight
  claim) → (+user_cf no-op, dense-gated). Matches REFERENCE almost
  exactly and confirms the auto-gates fire as documented.
- **beauty** is messier and worth stating honestly: 0.0263 raw cooc →
  0.0289 EASE → 0.0309 +trend → 0.0306 +last-item → 0.0312 +transitions
  → 0.0304 +rating-weight → **0.0328** +user_cf. Under the default
  cumulative path, rating-weight is **neutral-to-slightly-negative** here
  (not the isolated +5.2% REFERENCE reports) and **user_cf supplies the
  main late lift**; the endpoint reproduces at 0.0328, ~0.0015 below
  REFERENCE's 0.0343 (a config/user_cf-tuning gap flagged for the Phase-2
  default audit). The qualitative claim holds — the fused channels lift
  beauty **+24.7%** over the raw-cooc base — but the per-channel beauty
  deltas are order-dependent and should not be read as independent.

### II.3 The headline external result

Against popularity, item-item kNN, and implicit ALS across four datasets
(`warming_*`, `user_warmth_*`): kindling is the **strongest personalized
model at full data on all four** and **beats ALS everywhere**. On
cold-heavy realistic-tier catalogs it wins the cold-*user* buckets —
steam 1-4 items **0.053 vs ALS 0.019 (+180%)**, beating even popularity.
The honest boundary: in the genuinely data-starved *global* regime (≤~10%
data, or ml1m's tiny cold-user bucket) **popularity wins** — a universal
cold-start truth, not a kindling weakness. The defensible claim is
"best personalized model always; best on cold *users* where they matter,"
**not** "better on all cold data."

---

## Part III — Rejected experiments (the fence posts)

The summary table, then the detail. Each negative is kept because it
fences off a direction that looks promising but measurably isn't.

| Experiment | Verdict | Killer number | Evidence |
|---|---|---|---|
| Personas / clustering | DEAD as signal & router | zero LOO; only 28/500 beauty users route through a coherent persona, flat at every weight | `ADR-persona-signal`, `persona_method_*`, `clustering_coherence_*` |
| ALS as blend signal | DEMOTED | binary ALS ≈ weighted SVD on cooc; zero added info | `ADR-signal-audit`, §4.2 |
| LightGCN as blend signal | DEAD as signal | identical to 4 decimals in-blend (scale mismatch); viable only as retriever | `ADR-lightgcn-numpy` |
| graph-MF (base/boost) | SHELVED | base −13–15% NDCG; boost a wash | `graph_mf_*`, §4.3 |
| Learned gate MLP | DEAD | small ml1m win, loses grocery, 5× fit cost; feature space ~1-D | `ADR-scoring-architecture`, `scoring_architecture_*` |
| LambdaRank / GBM reranker | REJECTED | +7% ml1m eval-half but **−26% beauty**; deployable version craters −32%/−11% | `ADR-lightgbm-warm-regime`, `rerank_deploy_*` |
| Per-fit calibration | REJECTED (kept as diagnostic) | internal holdout **inverts** test ranking for trend_α (0.0310→0.0203 beauty) | §4.4 |
| Score-normalization as default | REJECTED (opt-in only) | zscore −60% ml1m / −17% grocery (exposed prior miscalibration) | `ADR-score-normalization` |
| Profile-wide recency decay | REJECTED | hurts every half-life ml1m (0.2841→0.2594); full-history sums carry signal | §4.5 |
| RRF as final ranker | REJECTED | loses to z-blend on both datasets | §4.5 |
| Content channel (warm blending) | DEAD on warm | dilutes warm ranking (ml1m 0.2841→0.2755 @α=0.5); flat-to-neg even at 100% coverage | §4.6 |
| LLM enrichment (keywords, aisle prompt) | PROBE-GATED → no value in tested regimes | separation gate FALSE; aisle labels = shuffled labels (R² −0.012); EASE+aisle zero lift | §4.7, `enrichment_probe_*`, `aisle_*` |
| Embedding imputation | DEAD (didn't transfer) | standalone +cold, but through engine *lowers* NDCG 0.1728→0.1686 (EASE scale mismatch) | §4.9, `engine_impute_40k.txt` |
| Edge grafting (metadata→cooc) | DEAD on thin meta | book every graft arm −25%; marginal-alive on steam tags (AUC ~0.59) | `graft_*` |
| Force-projection (FPR) | DEAD | 0.62× B1 on ml1m, below popularity floor; 30s build vs 0.6s | `fpr_probe_ml1m.json`, `force-projection-…-prd.md` |

### Detail on the load-bearing negatives

**Personas — the longest arc.** HDBSCAN-over-factors → Louvain on the
projected user-user graph (weight transforms, γ-resolution, user-trimming)
→ hand-rolled DC-SBM → post-hoc coherence scoring → persona-vs-cooc
differentiation. Survivors: coherence filtering is essential *when*
personas run; DC-SBM finds the most differentiated communities; HDBSCAN is
degenerate on binary-rating embeddings. But the final measurement on the
fused base is flat: **the fused base extracts the signal personas used to
carry.** All persona work lived within a few % of the popularity floor —
noise-band work on a broken base (§4.1).

**The learned-scoring family** (gate MLP, LambdaRank, ridge/GBM reranker).
The recurring failure: the feature space after the fused base is
effectively 1-D, so trees overfit and linear learned weights **invert**
between the internal (train-only) holdout and the test slice — the same
mechanism as per-fit calibration (§4.4). The eval-label *ceiling* is real
(+7% ml1m, +6% beauty) but **undeployable** on these protocols. This is
the empirical basis for the locked non-goal: **the activation detector is
deterministic because learned calibration provably doesn't deploy here.**

**The content cold-start program** (channel → enrichment → imputation →
grafting), closed across four independent attempts. Content reconstructs
only ~10% of cooc structure (the durable mapping-R² ceiling, §4.9); warm
protocols are interaction-saturated so even perfect content adds nothing
(§4.6); the regime where it *could* pay (>20k items AND content-coherent
AND warm-dominated) is occupied by **no cached dataset** (§7.6 screen).
The shipped cold-start answer is the structural `cold_slots` mechanism,
not a learned content ranker.

---

## Part IV — Open fronts (closed-as-bounded, and genuinely open)

From `REFERENCE.md §7`. Most are *closed-as-bounded*: large real oracle
headroom exists but requires a mechanism outside the shallow,
no-training, wheel-that-imports philosophy.

| Front | Status | Finding |
|---|---|---|
| §7.1 Cold-extension coverage policy | demand-aware *selection* **refuted** | book is metadata-coverage-limited (~18% ceiling), not selection-limited; salesRank already best; recency strong where it applies (steam 90%) but cap doesn't bite there |
| §7.2 Oracle headroom | **CLOSED, unified** | ml1m is ranking-bound (oracle 0.93 vs 0.29) but the gap is *sequential* — learned rerank +7% doesn't generalize/deploy; beauty is retrieval-bound (median pool recall 0) — union reaches the items (recall 0.28→0.38) but no fixed-weight blend ranks them. Both need a discriminative/sequential ranker fixed weights can't express and learned weights can't deploy. |
| §7.3 EASE beyond the 20k gate | **CLOSED, negative** | low-rank EASE decelerates far below wilson (r=512: 0.0358 vs wilson 0.0482) and needs r≈4000+ for parity (hours); wilson already removes popularity cheaply. Gate stays at 20k. |
| §7.4 Cold-user serving | **BUILT** | `recommend_for_items` serves brand-new/anonymous users from ad-hoc seeds with no per-user training; 0-seed → popularity; EB popularity-shrinkage lifts the thin-seed dip above the popularity floor on popularity-heavy catalogs |
| §7.5 More realistic-tier datasets | **GENUINELY OPEN** | RetailRocket (clickstream, hashed metadata), H&M (rich metadata + churn) would exercise content mechanics under churn |
| §7.6 Cooc-base imputation | **CLOSED, regime absent** | content-coherence lives at *small* scale; scaling past 20k loses it and gains cold-domination. No cached dataset hits the sweet spot. Reopen only with a screened-in download (Amazon Video Games is the untested candidate). |

---

## Part V — Evidence map & retention manifest

**Retention rule for the consolidation:** `bench/reports/` (ADRs + frozen
JSON/txt metric files) is **writeup, not code** — it is retained in full
when the runner scripts (`bench/run_*.py`, `src/kindling/benchmarks/*`)
are deleted. Every claim in this document must trace to a retained
artifact below before the corresponding code is removed.

| Topic | ADR | Frozen result artifact(s) |
|---|---|---|
| Baselines vs industry | `ADR-baselines-comparison` | `baselines_comparison*.json` |
| Signal audit / which signals earn compute | `ADR-signal-audit`, `ADR-signals-and-growth` | `signal_ablation_*.json` |
| Scoring architecture (Bayesian/gate/RRF) | `ADR-scoring-architecture` | `scoring_architecture_*.json` |
| Likelihood default | `ADR-phase3-default-likelihood` | `likelihood_suite_*.json` |
| Temperature solver | `ADR-phase4-temperature-solver` | `temperature_suite_*.json` |
| Persona signal & methods | `ADR-persona-signal` | `persona_method_*`, `clustering_coherence_*`, `coherence_percentile_*`, `probe_persona_*` |
| LightGCN | `ADR-lightgcn-numpy` | (academic split via `run_lightgcn_academic`) |
| graph-MF | — | `graph_mf_*.json` |
| Repeat consumption | `ADR-repeat-consumption` | `growth_*_persona/ratings` |
| Score normalization | `ADR-score-normalization` | (in ablation/parity) |
| Retriever standalone / matrix / union | `ADR-standalone-retrievers`, `ADR-retriever-ranker-matrix`, `ADR-retriever-union` | `retriever_standalone_*`, `retriever_matrix_*`, `retriever_union_*` |
| Channel progression (EASE→+trend→+last-item→+rating-weight→+user_cf) | — | **`channel_ablation_{movielens-1m,amazon-beauty}.json`** (re-captured this run) |
| Rating-aware signals (intensity) | `ADR-rating-aware-signals` (NB: measures the *pre-pivot* cooc engine — a wash there; the post-pivot EASE +1.8% ml1m is in `channel_ablation_movielens-1m.json`) | `channel_ablation_*.json` |
| Pair-index / distinctiveness basket signal | `ADR-pair-index-distinctiveness` | (path-family signal; see ADR) |
| Phase-7 cross-dataset consolidation | `ADR-phase7-cross-dataset` | `consolidated/*` |
| Cold-slot recovery (the §4.8 cold-start answer) | — | steam (0%→8.5%) and book (0%→0.5%) recovery in `REFERENCE §3.3/§4.8`. *Note: re-freezing the book run for this consolidation OOM-killed on the available 24 GB box — the full-extension book fit peaks ~18 GB and does not fit alongside the OS; the warm-only book NDCG is in `book_chrono_warm.json`. The cold-slot recovery numbers remain REFERENCE-cited.* |
| Cooc smoothing / wilson base | — | `cooc_smoothing*.json` |
| Gap decomposition | — | `gap_decomp_current.json/.txt`, `consolidated/gap_decomposition_*` |
| ml1m ranking probe | — | `ml1m_rerank.json/.txt`, `rerank_*`, `rerank_deploy_*` |
| beauty retrieval probe | — | `beauty_retrieval.json/.txt`, `beauty_rerank.*` |
| EASE-beyond-gate | — | `ease_large.json`, `ease_large_*.txt` |
| Warming & cold-user | `ADR-growth-curves` (+ `REFERENCE §3.5`) | `warming_*.json/.txt`, `user_warmth_*.json/.txt` |
| Onboarding / cold-user serving | — | `onboarding_*.json/.txt` |
| Content cold-start: imputation/enrich/graft/screen | — | `engine_impute_40k.txt`, `meta_cooc_map.json`, `enrichment_probe_*`, `aisle_*`, `graft_*`, `screen_ml-25m-*`, `cold_coverage_*` |
| Force-projection | — | `fpr_probe_ml1m.json` |
| Perf (v1 vs v2, Rust) | `ADR-phase8-rust-evidence` | `perf/v1_*`, `perf/v2_*` |

**Provenance honesty — claims that are REFERENCE synthesis, not a
separately frozen sweep** (a 2026-06 provenance audit surfaced these; they
are kept because the *verdict* is sound and corroborated, but the raw
sweep was never written to `bench/reports/`, so they are cited as
synthesis, not as a frozen artifact):
- The §3.4 published-baseline table (cooc-base academic 0.0369/0.0285 vs
  NGCF/LightGCN/Mult-VAE) and the LightGCN undertrained epoch-sweep — the
  `run_lightgcn_academic` / `run_book_academic` stdout was not saved
  (compute-bounded: ~10h to converge LightGCN on CPU). See `REFERENCE §3.4`.
- The per-fit-calibration inversion (§4.4) — the grid lands in the runtime
  `profile`, never in `bench/reports/`.
- The recency-decay and content-α warm-dilution sweeps (§4.5–4.6).
- The persona "28/500 coherent-route" headcount (§4.1) — the DEAD verdict
  is corroborated by `persona_method_*` / `signal_ablation_*_persona`; the
  exact count is REFERENCE prose.

**Historical PRDs** (archived under `docs/archive/`, not operative):
`kindling_PRD_v08.docx` (original vision) and
`force-projection-recommender-benchmark-prd.md` (FPR, result negative).
