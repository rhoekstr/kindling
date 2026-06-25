# kindling — The Production System

> A clean description of what kindling *is* after the v0.2 consolidation —
> for someone who never saw the experiment history. What it includes, what
> value it adds (and where it adds none), what's noteworthy, and the
> measured numbers. The full experiment record (what was tried and why,
> including the negatives) is the addendum in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## 1. What it is

kindling is a **hybrid recommender with no training loop**. It produces one
fused score per (user, item) from a closed-form base (EASE or
wilson-normalized co-occurrence) plus a small set of counting-statistic
channels (trend, last-item, transitions, user-CF), each **activated by a
measurable property of the dataset**. The numerics run in a pure-Rust core
(`kindling_core`); the Python shell does orchestration, profiling, and data
plumbing.

Two design commitments, both learned the hard way:
1. **A wheel that imports is a wheel that works** — numpy/pandas/scipy only,
   no PyTorch / BLAS / optional-extra hell.
2. **Closed-form shallow models, gated per dataset, beat speculative
   complexity** — every channel is closed-form or a count; every channel is
   gated; every gate exists because the ungated version measurably hurt
   somewhere.

It fits in seconds-to-minutes on CPU with no GPU, and serves in
sub-millisecond time.

## 2. What it includes

| Component | What it does |
|---|---|
| **`Engine`** (`engine`) | fit + `recommend` + `recommend_for_items`; the single public entry point |
| **Base scorer** | rating-weighted **EASE** (catalogs ≤ 20k items) or **wilson-cooc** (above), auto-selected |
| **Channels** | trend (timestamps), last-item (EASE structure), transitions (sequential, non-burst), user-CF (sparse history) — z-normalized, additive |
| **Activation detection** | `engine.activation_plan` — the inspectable regime → layer-decisions record (§4) |
| **New-user / anonymous serving** | `recommend_for_items` scores any seed set with no per-user training; popularity fallback at zero seeds; empirical-Bayes shrinkage for thin seeds |
| **Cold-item serving** | reserved `cold_slots` + open-catalog: metadata-only items ranked by content similarity + release recency |
| **Rust core** | EASE Cholesky (faer), cooccurrence, directional cooc, layered scoring, retrieval |
| **Loaders** | movielens, amazon (+chrono), steam, instacart, dunnhumby, tafeng, gowalla, yelp, retailrocket, synthetic |
| **Verification harness** | `bench/verify.py` (4-dataset regression gate) + `run_gap_decomp.py` (the floor/oracle/pool-recall diagnostic) |

The core is **40 Python modules** (down from ~130 pre-consolidation): the
validated stack and a minimal CI harness, nothing else.

## 3. Value-add — and where it has none

**The defensible, two-sided claim** (from the warming & cold-user benchmark
vs popularity, item-item kNN, and implicit ALS across four datasets;
`bench/reports/warming_*`, `user_warmth_*`):

**Where it adds value:**
- **Strongest personalized model on all four datasets** at full data, and it
  **beats trained MF (ALS) everywhere**.
- **Cold *users* on cold-heavy catalogs** — on steam it leads every history
  bucket *including the coldest*, beating even the popularity prior (1–4
  items: **0.053 vs ALS 0.019, +180%**); on beauty its margin is largest on
  the shortest histories.
- **No-training serving of brand-new / anonymous users** — a capability
  trained MF structurally cannot provide (it can't score users absent from
  training).
- **Self-explaining configuration** — the engine states which layers it
  activated and why (§4).

**Where it adds none — stated plainly:**
- **Data-starved *global* regimes**: with very little total data (≤~10%), or
  on a dataset with essentially no cold users (dense ml1m), the
  non-personalized **popularity prior wins**. Personalization needs
  co-occurrence structure before it beats popularity — a universal
  cold-start truth, not a kindling-specific weakness.
- **Ranking headroom is sequential and out of scope**: on ml1m the oracle on
  the retrieved pool is 0.93 vs the engine's 0.29 — but the gap is *which
  item comes next*, which needs a trained sequence model (out of the
  no-training philosophy). Shallow and learned re-rankers were evaluated and
  did not deploy (EXPERIMENTS.md §7.2).
- **Retrieval headroom on retrieval-bound catalogs**: on beauty half of
  users' held-out items never reach the candidate pool; multi-source
  generation reaches them but no fixed-weight blend ranks them, and learned
  weights don't deploy. Closed as philosophy-bounded.
- **Content cold-start banks no production value**: the content channel,
  LLM enrichment, embedding imputation, and edge grafting were each tried
  and retired — content reconstructs only ~10% of co-occurrence structure
  and adds nothing on warm protocols (EXPERIMENTS.md §4.6–4.9). The shipped
  cold-item answer is the *structural* `cold_slots` mechanism, not a learned
  content ranker.

## 4. Noteworthy / novel

- **Deterministic regime-based activation detection.** Rather than a learned
  gating network (which was built and *failed to deploy* — the internal
  holdout inverts the test ranking), activation is a deterministic regime
  classifier. The data *proved* fixed cross-dataset gates beat per-fit
  calibration, so the engine can **state and defend its own
  configuration** via `activation_plan`. "Intelligent activation" here means
  evidence-grounded determinism, not a model.
- **"Raw co-occurrence is popularity in a costume."** The pivot that defines
  the engine: a gap-decomposition diagnostic showed the old cooc base beat a
  popularity ranker by only +2.7% on ml1m, because `Σ cooc[i,c] ≈ item
  popularity`. EASE's inverse-Gram (and wilson's cheap normalization)
  subtract exactly that redundancy.
- **The realistic-protocol tier.** 5-core preprocessing *deletes* the cold
  population, so academic benchmarks can't even test cold-start machinery. A
  second methodology (no k-core, segment-sliced by item warmth) was added to
  measure it honestly — and several verdicts flipped between the two tiers.
- **Closed-form, no-training, no-GPU** — competitive with (and on cold users,
  better than) trained MF, while fitting in seconds and serving sub-ms; the
  shallow-baselines-rival-GNNs result, reproduced on its own engine (~90% of
  LightGCN at ~400× less compute on amazon-book).
- **The negatives are kept.** ~10 rigorously-retired directions are preserved
  as fence posts in EXPERIMENTS.md — the boundary of where this philosophy
  stops paying.

## 5. Performance statistics

**Accuracy — full-ranking NDCG@10, engine defaults** (measured this run via
`bench/verify.py`; chronological split, strided-500 eval, full-catalog
ranking, k=10):

| dataset | NDCG@10 | Recall@10 | MRR | HR@10 | fit | base |
|---|---:|---:|---:|---:|---:|---|
| movielens-1m | **0.2928** | 0.0611 | 0.4734 | 0.756 | ~6s | rating-weighted EASE |
| amazon-beauty | **0.0328** | 0.0425 | 0.0432 | 0.094 | ~15s | EASE (λ=250) |
| steam (realistic) | **0.0660** | 0.1200 | 0.0608 | 0.164 | ~250s | EASE + cold_slots |
| amazon-book-chrono | 0.0318† | 0.0443 | 0.0426 | 0.080 | ~25min | wilson cooc |

†amazon-book-chrono is **REFERENCE-cited, not re-measured this run**: its
full-extension fit peaks ~18 GB and OOM-killed on the available 24 GB dev box
(the other three were re-measured). It is local/manual validation, not a CI
gate.

*(Calibrate to full-ranking literature: sampled-negative papers report
HR@10 ≈ 0.5; those deflate ~10× under full ranking. Full-ranking SOTA on
beauty is NDCG@10 ≈ 0.03–0.05 — the band kindling is in.)*

**vs standard algorithms** (full data, NDCG@10; `bench/reports/warming_*`):

| dataset | kindling | popularity | item-kNN | ALS |
|---|---:|---:|---:|---:|
| movielens-1m | **0.309** | 0.259 | 0.257 | 0.247 |
| amazon-beauty | **0.032** | 0.006 | 0.026 | 0.022 |
| steam | **0.050** | 0.039 | 0.027 | 0.016 |
| amazon-book | **0.043** | 0.002 | 0.040 | 0.014 |

**Cold-user slice** (steam, NDCG@10 by history length; `user_warmth_steam`):
1–4 items **0.053** (pop 0.040 / kNN 0.034 / ALS 0.019) — kindling leads the
coldest bucket, beating even popularity.

**Cost:** no GPU, no training loop. Fit is one dense Cholesky (EASE) or an
O(edges) cooc pass — ml1m ~4–6s, beauty ~15s, steam ~250s. Serve latency
(ml1m, full-catalog scoring, no caching): **p50 0.5 ms, p95 1.1 ms** per
recommendation. ruff-clean (739→0), 121 tests green.
