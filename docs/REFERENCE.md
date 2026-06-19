# kindling — Reference

> A reference for the kindling recommender as it stands today (June 2026).
> Synthesizes the v2 architecture (Rust core + Python shell), the fused
> multi-channel base scorer, the auto-detection gates, and the empirical
> record — including the negative results, which are half the value.
>
> **This document is the living source of truth.** Whenever the
> architecture, signals, defaults, or benchmark results change, update
> this file in the same change. `bench/reports/` remains the deeper
> chain-of-evidence; this is the synthesis.

---

## 1. What kindling is

A hybrid recommender: a **Rust core** (`kindling_core`, PyO3) holding all
numeric kernels, with a thin **Python shell** (`kindling.engine_v2`) for
orchestration, profiling, and dataset plumbing. No PyTorch, no autograd,
no BLAS/LAPACK system deps — the linear algebra that matters (the EASE
inversion) runs on `faer`, pure Rust. This is deliberate: the v1 era was
dominated by silent dependency breakage (umap×numpy ABI, sklearn drift,
optional-extra hell), and v2's first design goal is that **a wheel that
imports is a wheel that works**.

The second design goal, learned the hard way (§4): **closed-form shallow
models, empirically gated per dataset, beat speculative complexity.**
Every channel in the scorer is either closed-form or a counting
statistic; every channel is gated by a measurable property of the
dataset; and every gate exists because the ungated version measurably
hurt somewhere.

## 2. The scoring model

One fused base score per (user, candidate), built from independent
z-normalized channels:

```
score = z(base) + 0.5·z(trend) + 0.25·z(last_item) + 0.25·z(transitions)
        + 1.0·z(user_cf)                     [sparse-history datasets only]
        [+ content_alpha·coldness·z(content)]          (opt-in, default off)

base        = EASE row-sum   (catalog ≤ ease_max_items, default 20k;
              rating-weighted when detect_rating_signal says "ratings")
            | raw cooc sum   (above the gate; "cooc_fused" path)
trend       = recent-window item popularity        [needs timestamps]
last_item   = EASE row of the user's newest item   [needs EASE]
transitions = directional cooc rows of last-5 items, decay 0.7
              [needs timestamps AND NOT rating-burst]
user_cf     = k-NN user neighbors' items (Otsuka-Ochiai over user sets)
              [median history ≤ user_cf_history_gate (20) — sparse data only]
content     = IDF/L2 item-feature cosine, cold-gated per item
              [needs item_metadata; pays only on cold-start protocols]
```

Retrieval and scoring are the same pass: the candidate pool is the
top-`retrieval_budget` (500) of the *blended* full-catalog vector. On the
sparse cooc path this is genuine retrieval fusion — trend/transition
channels promote items the cooc retriever alone could never surface.
Boost layers (path/session/temporal z-gated boosts) then apply within
the pool as before (size-gated off above 100k items: each layer is a
full duplicate of the item-item CSR).

**Open-catalog mode** (`open_catalog=True`, default): metadata-only
items — present in `item_metadata` but absent from train — join the
catalog as scoreable candidates. Interaction channels see zeros for
them; the content channel + the cold machinery below are their only
path in. **Reserved cold slots** (`cold_slots`, default 0): the last
N of the top-K are reserved for cold items (< 5 train interactions),
ranked by `z(content-similarity-to-user) + cold_recency_beta ·
exp(−days_since_release/180)`. The release-date column is
schema-inferred (first majority-parseable datetime column in the
metadata). This is structural, not score-blending: warm ranking is
untouched, so cold exposure costs nothing measurable (steam: warm
recovery 80/753 with or without slots).

### 2.1 EASE (the base that matters)

Steck, WWW 2019. `B = −P/diag(P)`, `P = (XᵀX + λI)⁻¹`, zero diagonal.
One dense Cholesky inversion (`faer`), f64 math / f32 storage. λ
defaults to the heuristic `20 · nnz / n_items`; explicit `ease_lambda`
pins it (amazon-beauty measures slightly better at 250 than its
auto ≈ 299).

X is binarized on implicit data; when `detect_rating_signal` finds
true ratings, X carries them (max per user-item pair) — **rating-
weighted EASE**: ml1m +1.8% NDCG, and on beauty (1–5 star reviews)
+5.2%. Preference *intensity* was the last signal the binarized Gram
discarded.

Why it replaced raw cooc as the base: **raw co-occurrence scoring
degenerates toward popularity ranking** — for popular items, `cooc[i,c]`
is large for every i, so `Σ_{i∈owned} cooc[i,c]` ≈ item popularity. The
gap-decomposition diagnostic (§3) showed the old base beat a pure
popularity ranker by only +2.7% on ml1m. EASE's inverse-Gram subtracts
exactly that redundant structure: it learns which co-occurrences are
informative rather than merely frequent.

### 2.2 The gates (auto-detection doing real work)

| gate | controls | why it exists |
|---|---|---|
| `n_items ≤ ease_max_items` | EASE vs cooc base | O(n³) inversion feasibility (~20k items ≈ minutes / few GB) |
| `detect_rating_signal` = ratings | rating-weighted Gram | intensity helps only when ratings are real (binary data unchanged) |
| timestamps present | trend, transitions | nothing to compute otherwise (amazon-book academic split correctly no-ops) |
| `rating_burst_detected` | transitions OFF | within-burst order is noise — ml1m measurably hurt by transitions at every weight |
| median history ≤ `user_cf_history_gate` (20) | user_cf channel | helps sparse-history data (+4.5% beauty NDCG); −1.6% on dense ml1m where EASE already encodes the neighborhood |
| `n_items ≤ 100k` | boost-layer adjacency builds | each layer duplicates the full item-item CSR — OOM territory on book-scale catalogs |
| `item_metadata` present + `content_alpha > 0` | content channel | opt-in; see §4.6 |
| per-item coldness | content weight / cold slots | content dilutes warm ranking when ungated (§4.6); cold slots make it structural instead (§4.8) |

The last-item channel is deliberately **not** burst-gated: it reads the
EASE row (co-occurrence structure) of the newest item, not within-burst
order, and it helps on ml1m where raw transitions hurt.

## 3. Methodology

### 3.1 Canonical protocol

Chronological 90/10 split, strided-500 eval users by sorted entity_id,
full-catalog ranking (no sampled negatives), k=10. Harness:
`kindling.benchmarks.parity._build_eval_set`.

**Realistic-protocol tier** (the second methodology, added after the
academic tier kept "disproving" cold-start layers): NO k-core
filtering, chronological global split, **segment-sliced reporting** —
held-out recovery bucketed by item warmth (0 / 1–4 / 5–19 / 20+ train
interactions) alongside the aggregate. The rationale is structural:
5-core preprocessing *deletes* the cold population, so academic
benchmarks cannot reward (or even test) content/LLM/cold-start
machinery — a layer flat on a 5-core aggregate is unproven, not
disproven. First member: `steam` (Kang & McAuley crawl, 7.8M reviews,
2.3M users, 14k train items, 15% of test events on cold items, 11.9%
of test items entirely unseen in train). Second: `amazon-book-chrono`
(the 5-core reviews on a chronological split — k-core baked into the
source, but the chronological boundary still yields train-cold items).

**Calibrate expectations to full-ranking literature.** Classic papers
(NCF, SASRec originals) rank against 100 sampled negatives; their
HR@10 ≈ 0.5 numbers deflate ~10× under full ranking (Krichene & Rendle
2020). Full-ranking SOTA on amazon-beauty is NDCG@10 ≈ 0.03–0.05 — the
band we are in.

### 3.2 Gap decomposition (`benchmarks/gap_decomposition.py`)

The diagnostic that drove the 2026-06 pivot. Brackets the system
between a popularity floor, an oracle ceiling on the same candidate
pool, and the retrieval ceiling (pool recall):

| | ml1m | amazon-beauty |
|---|---:|---:|
| popularity floor | 0.2492 | 0.0063 |
| raw-cooc base (pre-pivot) | 0.2561 | 0.0290 |
| oracle on same pool | 0.8845 | 0.2617 |
| pool recall@500 (median) | 0.56 | **0.00** |

Reading: ml1m's scorer was a popularity ranker in costume (3.5×
scoring headroom inside the existing pool); beauty was broken in both
stages (half its users had zero held-out items in the pool). Run this
diagnostic before believing any architectural conclusion.

### 3.3 Current results (engine defaults, June 2026)

| | NDCG@10 | MRR | recall@10 | HR@10 | notes |
|---|---:|---:|---:|---:|---|
| ml1m | **0.2931** | 0.4735 | 0.0612 | 0.756 | rating-weighted EASE |
| amazon-beauty (λ=250) | **0.0343** | 0.0441 | 0.0463 | 0.098 | + user_cf channel |
| steam (realistic tier) | **0.0660** | — | — | — | open-catalog, cold_slots=1, recency prior |
| amazon-book-chrono | **0.0318** | 0.0426 | 0.0443 | 0.080 | +24.5% NDCG over academic split — timestamps activate trend/transitions; cold_slots=1 + meta_Books; fit ~27min/17.9GB peak (extension auto-capped 200k→107k) |
| amazon-book† | 0.0253 | 0.0563 | 0.0246 | 0.140 | academic split; channels no-op |

† amazon-book (plain) now loads the McAuley 5-core JSONL if present
(596k users / 357k items, random split). The LightGCN *academic*
split (52k/91k, `train.txt`/`test.txt`) coexists in the cache; load it
explicitly via `_load_academic_split` (see `bench/run_book_academic.py`)
for the published-baseline comparison below.

Steam segment slice (the realistic-tier scoreboard): warm-20+ recovery
11.0%, cold-0 recovery 0% → **8.5%** via cold slots + content +
release-recency — items the interaction stack cannot score at all.

amazon-book-chrono cold slice: cold-0 recovery 0% → **0.5%** (2/418).
Coverage-limited, not mechanism-limited: only 26% of warmth-0 held-out
items are in the salesRank-top-107k extension (30% in metadata at all),
and one reserved slot competes against a 106k-item cold pool. Book's
unseen demand is long-tail-by-salesRank; steam (better coverage) is the
mechanism's real showcase.

### 3.4 vs published baselines — LightGCN academic amazon-book

The most-cited RecSys benchmark (LightGCN-PyTorch split: 52,643 users /
91,599 items / 2.38M train), full-catalog ranking, k=20, 5000 eval
users. kindling runs its **weakest** config here — cooc base (91k >
EASE gate), no trend/transitions (timestamp-less split):

| model | Recall@20 | NDCG@20 |
|---|---:|---:|
| NGCF (graph NN) | 0.0344 | 0.0263 |
| **kindling (cooc base)** | **0.0369** | **0.0285** |
| Mult-VAE | 0.0407 | 0.0315 |
| LightGCN (graph NN) | 0.0411 | 0.0315 |

The stripped-down cooc base **beats NGCF** and reaches **~90% of
LightGCN/Mult-VAE** with an 86-second CPU fit and zero training — the
Dacrema et al. (2019) "tuned shallow baselines rival GNNs" result,
reproduced on our own engine. Blocked/low-rank EASE at 91k (open front
§7.4) would likely close the remaining gap.

**Why our own LightGCN never matched the published 0.0411** (a question
worth settling, not hand-waving): it's *undertrained, not broken*. The
hand-rolled Rust LightGCN climbs monotonically with epochs and shows no
plateau — d32/L2 gives Recall@20 0.0065 @30ep → 0.0086 @90ep (+32% for
3× training); d64/L3 gives 0.0063 @50ep. It's on the early, slow part of
LightGCN's known amazon-book curve (≈1000 epochs to converge). The
blocker is purely compute: ~36s/epoch on CPU ⇒ ~10h to converge, with no
GPU and an environment that kills >30-min jobs — so it was only ever
measured lightly trained. The cooc base reaching 0.0369 in 86s is ~400×
less compute for ~90% of the converged-GNN quality, which is the entire
point. Harness: `bench/run_lightgcn_academic.py` (epoch sweep; config
overridable via `LGCN_DIM`/`LGCN_LAYERS`/`LGCN_BATCH`/`LGCN_LR`).

Progression on ml1m: 0.2561 (raw cooc) → ~0.269 (EASE) → 0.2841
(+trend) → 0.2879 (+last-item) → 0.2931 (rating-weighted EASE).
Beauty: 0.0290 → 0.0306 (+EASE+trend) → 0.0315 (+transitions) →
0.0326 (rating-weighted) → 0.0343 (+user_cf). Both began the 2026-06
pivot within 3% of the popularity floor; both now sit at or above
published full-ranking shallow-model results.

### 3.5 Warming & cold-user benchmark — vs standard algorithms

Two harnesses (`bench/run_warming_curve.py`, `run_user_warmth.py`;
plots `plot_*`) compare the v2 stack against popularity, item-item kNN,
and implicit ALS (the industry-standard trained MF) across four datasets
(ml1m, amazon-beauty, steam, amazon-book-academic). Honest, two-sided:

- **Data-density warming** (nested random interaction subsample 1%→100%,
  fixed test, fixed real-user population): kindling is the strongest
  *personalized* model on every dataset and its lead grows with data —
  full-data NDCG@10 ml1m 0.31 (vs ~0.25-0.26), beauty 0.032 (vs kNN 0.026
  / ALS 0.022, **5.6× popularity**), steam 0.050 (vs pop ~0.04), book
  0.043 (vs kNN 0.040 / ALS 0.014). But in the genuinely data-starved
  regime (≤~10% data) **popularity often wins** — EASE/cooc needs
  co-occurrence structure before personalization beats the popularity
  prior. So the global-density advantage is *not* a thin-dataset edge; it
  emerges with density. Speed: kindling is the slowest of these
  *classical* baselines (EASE inversion: ml1m ~10s, steam ~240s vs ALS
  ~2-100s) but trains with no GPU / no training loop and serves at ~0.5 ms;
  the speed edge is vs neural (LightGCN: hours).

- **Per-user warmth** (full data, eval sliced by user history length) —
  the direct cold-*user* test, and the real win. On the **cold-heavy
  realistic-tier** datasets where cold users are common, kindling leads
  the cold buckets:
  - **steam** (2657 of 4000 eval users hold ≤4 items): kindling leads
    **every** bucket including the coldest, beating even popularity —
    NDCG@10 at 1-4 items **0.053** vs pop 0.040 (+34%) / ALS 0.019
    (+180%) / kNN 0.034.
  - **beauty** (1559 cold users): kindling leads the cold buckets, margin
    **largest on the coldest** — 1-4 0.045 vs ALS 0.032 (+40%); 5-19
    0.024 vs ALS 0.014 (+72%); edge shrinks as users warm.
  - **book-academic** (5-core → no 1-4 users): kindling ≈ item-kNN on the
    coldest available (5-19: 0.047 vs 0.047) and beats it warm; both
    crush ALS (~3×) and popularity. Item-item CF is naturally strong on
    book (the wilson cooc base is a normalized item-kNN), so they
    converge there.
  - **ml1m** (dense, ~no cold users — n=5 at 1-4): popularity wins the
    tiny cold bucket; kindling wins the warm majority. Cold-start is moot
    on a dataset with no cold users.

**The precise, defensible claim:** kindling handles cold *users* better
than competing personalized algorithms (ALS, item-kNN) on cold-heavy
catalogs — on steam beating even the popularity prior, with its largest
margins on the shortest-history users — and is the strongest personalized
model overall on all four datasets. It dominates trained MF (ALS)
everywhere. The only thing that beats it is the non-personalized
popularity prior in genuinely data-starved *global* settings (a universal
cold-start truth). NOT "better on all cold data" — better on cold *users*
where they matter, and best personalized model always.

## 4. The experiment record

What was tried, what won, what was rejected, and why. Negative results
are kept deliberately — they are the fence posts.

### 4.1 Personas / clustering — **benched**

The longest arc: HDBSCAN-over-factors → Louvain on the projected
user-user graph (weight transforms: raw/log/cosine; γ-resolution;
user-trimming) → hand-rolled DC-SBM (Rust) → post-hoc **coherence
scoring** (mean cooc over distinctive items — algorithm-agnostic
quality) → **persona-vs-cooc differentiation metrics** (jaccard@K,
rank-shift).

Verdicts that survived: coherence filtering is essential when personas
run (unfiltered SBM personas were net-negative); DC-SBM found the most
differentiated communities; HDBSCAN is degenerate on binary-rating
embeddings. But the final measurement on the fused base: only 28/500
beauty eval users route through a coherence-passing persona and NDCG is
flat at every blend weight. **The fused base extracts the signal
personas used to carry.** All persona experiments lived within a few %
of the popularity floor — noise-band work on a broken base. The
machinery (coherence, differentiation, DC-SBM) is retained as
diagnostics.

### 4.2 ALS on binary data — **demoted**

Implicit ALS on 0/1 data reduces to weighted SVD on cooc structure; adds
no information over cooc. `use_als="auto"` runs it only when
`detect_rating_signal` finds true ratings.

### 4.3 graph_mf (graph-regularized MF) — **shelved**

Built (directional + co-ownership graphs, optional hierarchy, Jacobi
Laplacian). Numerically stable in f64, no quality lift as base or boost
on the measured datasets. The directional-cooc builder it produced is
now the transition channel — the lasting payoff.

### 4.4 Per-fit calibration — **rejected, kept as diagnostic**

`calibrate_base=True` grid-searches (λ, trend_α, trans_α) on an internal
chronological holdout. Measured: the internal ranking **inverts** the
test ranking for trend_α (internal prefers 0.0; test strongly prefers
0.5) — beauty 0.0310 → 0.0203, ml1m 0.2859 → 0.2741. Shifting every
window back one slice changes the popularity-drift structure the trend
channel exploits, so the holdout systematically undervalues it. Fixed
cross-dataset defaults transfer better than per-fit optimization.
Default False; the grid lands in `profile["base_calibration"]` for
inspection.

### 4.5 Sequential rungs

- **Directional transitions** (last-5, decay 0.7, α=0.25): +2.9% NDCG /
  +7% recall on beauty; hurts ml1m at every weight → burst-gated.
- **Last-item EASE row** (α=0.25): helps both (ml1m +1.3% NDCG / +1.7%
  MRR; beauty recall +4.6%) → shipped, not burst-gated.
- **Profile-wide recency decay** over the EASE sum: hurts at every
  half-life on ml1m (0.2841 → 0.2594 at h=50), noise on beauty —
  **rejected**. Full-history sums carry signal.
- **RRF rank fusion** as the final ranker: loses to z-blend on both
  datasets — rejected.
- **SASRec-class models**: out of scope by philosophy (heavy training
  dependency; the shallow rungs captured the available sequential lift
  under these protocols).

### 4.6 Content channel (`item_features.py`) — **wired, default off**

Schema-inferring extractor, zero dataset-specific code: numeric →
quantile bins; list/delimited → multi-hot; low-cardinality strings →
one-hot; high-cardinality → bag-of-tokens; IDF + L2 → CSR. On ml1m it
infers `title:text, genres:multi_categorical(|), category:categorical`
unaided and Toy Story's content neighbors are Toy Story 2 / Balto /
Antz / Mulan.

Measured: ungated blending **dilutes** warm ranking (ml1m 0.2841 →
0.2755 at α=0.5); cold-gated blending (`α·clip(1−count/20,0,1)·z`) is
perfectly protective but unrewarded on warm protocols. The
metadata-coverage caveat was later eliminated: with era-matched 2014
metadata (100% beauty coverage, 6.4k features) content is **still**
flat-to-negative on warm ranking, uniformly across user-history
segments — the negative is fundamental to warm protocols, not a data
artifact. Where content DOES pay: steam's cold segments via the cold-
slot mechanism (§4.8) — curated tags rank the true cold item at median
136 of 20k extension candidates. `content_alpha=0` for blending; the
cold-slot path is where the capability earns.

### 4.7 LLM enrichment program — **probe-gated; pays only on thin metadata**

The question: can a small on-device model (Phi-4-mini, 4-bit MLX)
manufacture missing metadata? `llm_enrich.py` (batched, resumable,
JSONL-cached) + `benchmarks/enrichment_probe.py` (stage-1 sample
diagnostic) + `dense_content.py` (MiniLM embeddings, niche prompts,
user profiles). The validated decision rule:

```
SKIP       probe gates fail → keywords carry no taste signal; don't pay
SIGNAL_OK  gates pass → enrich IFF a cold/sparse population exists
           (alignment is necessary, NOT sufficient: warm protocols are
           interaction-saturated and aligned-but-redundant features
           add nothing — measured twice, ml1m LLM keywords and beauty
           curated metadata)
```

Probe gates: separation_d ≥ 0.5 (keyword-similarity of interaction-
neighbors vs random pairs), substitution ≥ 2× chance (keyword-kNN
recovering interaction-kNN), non-degeneracy (representation-aware:
random-pair cosine ≤ 0.3 sparse / ≤ 0.6 dense — MiniLM anisotropy).
Use N ≥ 400 sample items; N=200 is noise.

Findings that generalize:
- **Dense embeddings beat multi-hot** on the same keywords (d 0.665
  vs 0.574, substitution +11%) — zero extra LLM cost.
- **Plain keyword prompts beat fancy niche-phrase prompts**: stricter
  formats taxed 4-bit compliance (28% parse failures vs 4%) AND
  discriminated worse (d 0.475 vs 0.579 apples-to-apples). Short
  per-item prompts with banned-filler rules are the sweet spot.
- **LLM user profiles** ('1950s-noir, camp-fantasy, screwball-comedy')
  exactly tie the mean-of-owned-item-embeddings control on warm data:
  a faithful 10-token compression of taste. Value is cold-user /
  cross-domain / explainability, not warm lift.
- **Curated community labeling beats small-model generation**: steam
  tags d=0.497 vs LLM keywords d=0.383, and combining dilutes (0.479).
  Enrichment's domain is catalogs with THIN metadata (ml1m's bare
  genres), not rich ones. The 3-minute probe answers this before the
  multi-hour enrichment spend — run it first, always.

### 4.8 Open catalog + reserved cold slots — **shipped; the cold-start answer**

Steam exposed the regime academic data deletes: 13% of held-out items
had ZERO train interactions — structurally unreachable by every
interaction channel (0/117 recovered, no scorer can fix it; ranks
were median 3410/12k, so bigger pools don't help either — measured
before building). The fix is structural, not score-blending:
metadata-only items join the catalog (`open_catalog`), and the last
`cold_slots` of the top-K are reserved for cold candidates ranked by
content-similarity + release-recency (`cold_recency_beta`, 180-day
exponential). Steam: cold-0 recovery 0% → 6.0% (content) → **8.5%**
(+recency), aggregate NDCG 0.0623 → 0.0660, warm recovery untouched.
One slot of guaranteed cold exposure also mirrors how real systems
bootstrap new-item feedback loops.

Two operational lessons from the book run (each a real bug, each fixed):
- **`cold_slots>0` needs content features even when `content_alpha=0`** —
  the cold-slot ranker scores candidates by content similarity, so
  feature-building is gated on `content_alpha>0 OR cold_slots>0`. The
  earlier coupling made `cold_slots=1` a silent no-op (recovered 0/418).
- **The open-catalog extension is memory-capped** (`_open_catalog_extension_cap`):
  a naive 200k salesRank extension OOM'd a 24GB box (357k train +
  200k ext ≈ 23GB). The cap reserves the estimated interaction-fit peak
  (two-term model A·n_obs + B·n_train_items, calibrated to the steam
  3.4GB and book 17.4GB fits) and spends only the headroom under 80% of
  PHYSICAL RAM (swap absorbs the rest — `available` would mislead).
  Book auto-caps 200k→107k (~18GB peak); small datasets unconstrained.
  `open_catalog_max_extension` pins it explicitly.

### 4.9 Embedding imputation — **wired, default off; the bench positive did not transfer**

The cold-slot ranker (§4.8) ranks reserved candidates by content-space
cosine. Embedding imputation (`graph/cooc_impute.py`, `cold_impute`
knob) was the attempt to do better: predict a cold item's position in
**cooc-embedding space** (PPMI-SVD of the warm cooc) from its content
via a ridge map, then rank by cosine to the user's cooc-space taste
centroid — one vector per cold item, not k grafted edges, so it cannot
flood the pool the way edge-grafting did (§4.7-adjacent; book cratered
10×). The metadata→cooc **mapping-R²/neighbor-recovery** metric
(`bench/run_meta_cooc_map.py`) is the validated gate: it ranks
steam>book matching grafting outcomes, zeros content-orthogonal
catalogs, and tops out ~0.10 even on ml-25m tag-genome (content
reconstructs only ~10% of cooc structure — the durable ceiling).

**Standalone it works** (`bench/run_ml25m_lift.py`, ml-25m 40k): cold-1-4
recall 0→0.0081 (~half the warm tier), NDCG held, warm untouched — the
first cost-free cold positive of the whole enrichment arc.

**Through the production engine it does not** (`bench/run_engine_impute.py`,
ml-25m 40k, the Phase-1 guardrail):

| arm | NDCG@20 | cold-1-4 recall |
|---|---:|---:|
| off (`cold_slots=0`) | **0.1728** | 0.0 |
| content ranker | 0.1702 | 0.0014 (~4 items) |
| impute (gate active, r2 0.088) | 0.1686 | 0.0011 (~3 items) |

Impute **ties** the content ranker (3 vs 4 recovered cold items — noise)
and **costs ~2% NDCG**, recovering ~7× less than the standalone bench.
The cause is the same EASE-path **scale mismatch** edge-grafting hit
(§4.7 architecture note): the bench scored warm *and* cold in one
unified cooc-space, so cold competed for all 20 top-K slots; the engine
scores warm with **EASE** (a different space), so cold cooc-space scores
are incomparable to warm EASE scores and are confined to the reserved
slots. One slot cannot surface a 2.7%-prevalence cold population in a
13.8k catalog, regardless of ranker. The deeper tension: catalogs with
good content→cooc mapping are content-coherent and EASE-sized
(≤20k → scale mismatch), while catalogs with a native cooc base (>20k →
clean scale-match) have poor mapping (book r2 0.058). The bench positive
lived in a regime production doesn't occupy.

**Verdict:** `cold_impute` defaults to `"content"`. The imputation path
is kept (`cold_impute="impute"/"auto"`) for the **cooc-base regime**
(>20k items, where warm+cold share cooc-space) should a content-coherent,
warm-dominated, high-mapping dataset be built there (open front §7.6).
The clean scale-calibrator that would let cold compete in the main EASE
ranking is the real unsolved problem.

## 5. Engine knobs (the ones that matter)

| knob | default | touch when |
|---|---|---|
| `base_scorer` | `"auto"` | force `"ease"`/`"cooc"` for experiments |
| `ease_max_items` | 20 000 | more RAM/patience → raise |
| `ease_lambda` | None (auto `20·nnz/n_items`) | beauty-like datasets may prefer ~250 |
| `trend_alpha` | 0.5 | 0 to disable; window via `trend_window_fraction` (0.10) |
| `transition_alpha` | 0.25 | auto-gated off on burst datasets |
| `last_item_alpha` | 0.25 | 0.5 overshoots everywhere measured |
| `content_alpha` | 0.0 | blending stays off; cold slots are the content path |
| `user_cf_alpha` / `user_cf_k` / `user_cf_history_gate` | 1.0 / 100 / 20 | gate gives it to sparse-history data only |
| `open_catalog` | True | metadata-only items become candidates |
| `cold_slots` | 0 | reserve N of top-K for cold items (set 1 on churning catalogs) |
| `cold_recency_beta` | 2.0 | release-recency prior in cold-slot ranking; 0 disables |
| `cold_impute` / `cold_impute_min_r2` | `"content"` / 0.06 | cold-slot ranker; `"impute"`/`"auto"` is the dormant cooc-space path (§4.9 — did not transfer on the EASE path) |
| `open_catalog_max_extension` | None (RAM-auto) | pin the metadata-only extension size; auto caps it under 80% physical RAM |
| `retrieval_budget` | 500 | oracle says little headroom from raising it alone |
| `calibrate_base` | False | diagnostics only — see §4.4 |
| `persona_*`, `use_als`, `use_graph_mf` | benched/auto | diagnostics; see §4.1–4.3 |

## 6. Code map

```
native/kindling_core/src/
  signals/   ease.rs (faer Cholesky)  cooccurrence.rs  directional_cooc.rs
             als.rs  svd.rs  cosine.rs  lightgcn.rs  graph_mf.rs
             path_family.rs  session_cooccurrence.rs  persona_cooccurrence.rs
  cluster/   hdbscan.rs  louvain.rs (γ-resolution)  dc_sbm.rs  user_user_graph.rs
  persona/   index.rs  fit_gate.rs  coherence.rs
  score/     layered.rs  retrieve/  repeat/

src/kindling/
  engine_v2.py        orchestrator: profile → gates → channels → blend
                      + open-catalog extension + cold slots
  item_features.py    schema-inferring extractor + content_scores
  llm_enrich.py       batched/resumable LLM keyword generation (MLX)
  dense_content.py    MiniLM embeddings, niche/user-profile prompts
  loaders/steam.py    realistic tier: no k-core, chronological, parquet-cached
  loaders/amazon_chrono.py   books 5-core on the chronological protocol
  benchmarks/
    parity.py                   canonical eval-set builder
    gap_decomposition.py        floor / oracle / pool-recall diagnostic
    enrichment_probe.py         stage-1 LLM-enrichment go/no-go (run FIRST)
    clustering_coherence_sweep.py, persona_diff.py, ...   (persona-era diagnostics)
```

## 7. Open fronts

1. **Cold-extension coverage policy** — **investigated (Stage-0 diagnostic),
   demand-aware *selection* refuted.** `bench/run_cold_coverage.py` measures,
   per dataset, the coverage of warmth-0 cold demand vs fraction of the
   metadata pool admitted, under salesRank / recency / content / random.
   Findings: (a) **book** is the only cap-bites dataset, and **salesRank is
   already the best policy** — content-proximity is *worse* at every cap
   fraction, and book has no date for recency; its cold ceiling is only
   **~18%** (most cold-demand books aren't in the 2014 metadata), so book is
   **metadata-coverage-limited, not selection-limited.** (b) **recency is a
   strong demand signal where it applies** — steam newest-25% covers **90%**
   of cold demand vs random 26% — but steam's cap doesn't bite (everything
   fits), so it's moot there; it's ranker-limited. (c) **ml-25m** recency is
   weak (movies are evergreen, not bought-near-release like games). The
   regime where demand-aware selection pays — cap bites AND a strong recency
   signal AND a high metadata ceiling — is represented by no current dataset;
   it coincides with the §7.6 hunt and would validate there. (Trivial
   separate lever: raising book's cap 107k→200k buys ~+3pp coverage — a
   cap-size knob, not a policy.)
2. **Remaining oracle headroom** — re-grounded on the *current* engine
   with the faithful EASE pool (`bench/run_gap_decomp.py`; the library
   diagnostic's pool was stale — raw cooc, pre-pivot). The two benchmark
   datasets are bound by **different** walls:

   | dataset | base | pop floor | current | oracle (pool) | pool recall (med) | bound by |
   |---|---|---:|---:|---:|---:|---|
   | ml1m | ease | 0.2492 | 0.2931 | **0.9315** | 0.66 | **ranking** |
   | beauty | ease | 0.0063 | 0.0325 | 0.2773 | **0.00** | **retrieval** |

   **ml1m is ranking-bound**: the pool holds the answers (oracle 0.93) but
   the scorer delivers 0.29 — a 0.64 gap, 3.2× scoring headroom *inside*
   the pool (retrieval improved 0.56→0.66 since the pre-pivot §3.2 table).
   Closing a pure ranking gap on a burst dataset is what sequence models
   do (out of philosophy); the shallow levers (recency decay, per-fit
   calibration, channel reweighting) were tried and rejected (§4.4–4.5),
   so this needs a bounded shallow probe before conceding it to sequence
   modeling. **beauty is retrieval-bound**: median pool recall 0 — half of
   users' held-out items never reach the pool, so no ranker can help; the
   in-philosophy lever is multi-source candidate generation.

   **ml1m-ranking probed (`bench/run_ml1m_rerank.py`) → gap is sequential.**
   On the engine's actual pool: in-philosophy shallow re-ranking can't
   move it (recent-window EASE *hurts* 0.26-0.27; trend reweight is
   noise-band — +1.8% NDCG / −0.8% recall, mixed across datasets). A
   *learned* non-linear re-ranker (LightGBM) over the SAME features lifts
   the eval-half +7% (0.287→0.308) — but a linear ranker (logreg) does
   not, and the learned ceiling stays ~0.62 below the oracle (0.93). So
   the gap is **feature-limited, not ranker-limited**: the missing signal
   is which item comes *next* (sequential), out of the no-training
   philosophy.

   **Learned re-ranker fully evaluated → rejected** (`bench/run_rerank.py`,
   `run_rerank_deploy.py`). (a) The LightGBM ceiling (+7% ml1m) does NOT
   generalize — it *craters −26% on sparse beauty* (trees overfit few
   positives) — and would add a compiled C++ runtime dep against the
   wheel-that-imports philosophy. (b) A dep-free closed-form ridge over
   the channels + numpy degree-2 crosses beats the z-blend with eval-label
   ceiling (+1.5% ml1m / +6% beauty), but the **deployable** version —
   trained on the only available internal (train-only) holdout — *craters
   −32% ml1m / −11% beauty*, the §4.4 inversion in full force (linear too).
   So learned re-ranking is undeployable on these protocols. This
   **vindicates** the fixed cross-dataset z-blend: per-fit/learned
   calibration inverts because the internal holdout's drift/next-item
   structure differs from the test slice. **ml1m-ranking is closed; the
   remaining oracle headroom is sequential (out of scope).**

   **beauty-retrieval probed (`bench/run_beauty_retrieval.py`,
   `run_beauty_rerank.py`) → reachable but unrankable.** Multi-source
   candidate generation DOES reach the missing items: union of
   EASE+popularity+2-hop lifts recall@500 mean 0.28→0.38, median 0→0.25,
   and the oracle over the union pool is 0.365 (vs current 0.034 — 10×
   headroom). But no in-philosophy fixed-weight blend surfaces them:
   adding 2-hop *hurts* (0.034→0.029 — it pools the items but scores them
   like noise, diluting EASE), popularity is flat (+0.9%, noise). Surfacing
   them needs a ranker that discriminates *which* union items are relevant
   — the learned re-ranker, undeployable per §4.4. **So §7.2 closes,
   unified: both walls (ml1m ranking, beauty retrieval) have large real
   oracle headroom that requires a discriminative/sequential ranker fixed-
   weight blending can't express and learned weights can't deploy. The
   engine is at its philosophy-bounded optimum.**
3. **EASE beyond the gate** — **CLOSED, negative (Phase 2).** Low-rank
   EASE (top-r eigendecomposition of the sparse Gram, scoring without
   materializing the dense n×n B; `bench/run_ease_large.py`) was the
   memory-feasible candidate. On book-91k it climbs but **decelerates
   far below wilson**: NDCG@20 0.0254 (r=128) → 0.0310 (r=256) → 0.0358
   (r=512) vs wilson's **0.0482**. λ barely moves it (rank, not
   regularization, is the binding constraint), so parity needs r≈4000+ —
   hours of decomposition and multi-GB Q vs wilson's 86s O(edges) fit.
   Mechanism: EASE's value is removing popularity/redundancy via the
   inverse-Gram, but wilson *already* removes popularity cheaply; EASE's
   residual edge lives in a near-full-rank inverse that has no low-rank-
   feasible form on this heavy-tailed spectrum. (Aside: r=512 EASE 0.0358
   already beats published LightGCN/Mult-VAE 0.0315 — but wilson is
   better *and* cheaper.) **The EASE gate stays at 20k; wilson is the
   >20k base.** CG-column / sparse-B variants face the same
   spectrum + a diag(P) obstacle and are not pursued.
4. **Cold-user serving** — **BUILT** (`EngineV2.recommend_for_items`).
   Brand-new / anonymous users (absent from training) are served from
   ad-hoc seed items: the closed-form base scores from *any* seed set
   with no per-user training, so a user who just interacted with a few
   items gets personalized recs immediately; zero/all-unknown seeds fall
   back to all-time popularity (`_cold_recommend` — the warming
   benchmark's cold-data champion). Onboarding curve
   (`bench/run_onboarding.py`, NDCG@10 vs # seeds): graceful 0-seed
   fallback (==popularity), then kindling personalizes and overtakes —
   crossover at **1 seed (beauty)**, ~3 (steam), ~7-10 (ml1m); the
   crossover comes later on popularity-heavy catalogs, with a 1-seed dip
   below popularity there (a single seed is a weak signal vs a strong
   popularity prior). Popularity is flat (ignores seeds) and trained MF
   cannot serve absent users at all. Open refinement: shrink the
   seed-based score toward popularity at low seed counts so 1-2 seeds
   never underperform the prior. LLM user profiles (§4.7) remain a
   separate, untested cross-domain bootstrap angle.
5. **More realistic-tier datasets** — RetailRocket (live clickstream,
   hashed metadata) would test content-channel mechanics under churn
   without LLM enrichability; H&M (rich readable metadata + churn)
   would exercise the full stack, Kaggle auth permitting.
6. **Cooc-base embedding imputation** — **CLOSED, regime absent
   (dataset screen, `bench/run_dataset_screen.py`).** Imputation (§4.9)
   needs the cooc-base sweet spot: >20k items AND content-coherent
   (mapping-R²) AND warm-dominated. The screen shows a structural
   anti-correlation — content-coherence lives at *small* scale here, and
   scaling past 20k loses it and gains cold-domination:

   | dataset | items | >20k | mapping-R² | warmth |
   |---|---:|:--:|---:|---|
   | steam (tags) | 14k | ✗ | 0.077 | warm-dom |
   | ml-25m genome | 13.4k | ✗ | 0.088 | warm-dom |
   | ml-25m unrestricted (genres) | 35.7k | ✓ | **0.035** | **53% cold** |
   | amazon-book (categories) | 357k | ✓ | 0.058 | cold-dom |

   No cached dataset occupies the sweet spot. Combined with the §4.9
   EASE-path scale-mismatch, the §7.1 selection non-bottleneck, and the
   ~10% content ceiling (§4.7), this **closes the content cold-start
   program**: across imputation, edge-grafting, the content channel, and
   enrichment, content is a structurally weak supplement that banks no
   production value in the regimes available. The validated stack is
   **wilson cooc base + EASE warm scorer**; cold exposure, where wanted,
   is the structural `cold_slots` mechanism (§4.8), not a learned content
   ranker. Reopen only with a downloaded dataset that screens into the
   sweet spot (Amazon Video Games is the untested candidate).
