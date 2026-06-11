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
        [+ content_alpha·coldness·z(content)]          (opt-in, default off)

base        = EASE row-sum   (catalog ≤ ease_max_items, default 20k)
            | raw cooc sum   (above the gate; "cooc_fused" path)
trend       = recent-window item popularity        [needs timestamps]
last_item   = EASE row of the user's newest item   [needs EASE]
transitions = directional cooc rows of last-5 items, decay 0.7
              [needs timestamps AND NOT rating-burst]
content     = IDF/L2 item-feature cosine, cold-gated per item
              [needs item_metadata; pays only on cold-start protocols]
```

Retrieval and scoring are the same pass: the candidate pool is the
top-`retrieval_budget` (500) of the *blended* full-catalog vector. On the
sparse cooc path this is genuine retrieval fusion — trend/transition
channels promote items the cooc retriever alone could never surface.
Boost layers (path/session/temporal z-gated boosts) then apply within
the pool as before.

### 2.1 EASE (the base that matters)

Steck, WWW 2019. `B = −P/diag(P)`, `P = (XᵀX + λI)⁻¹`, zero diagonal,
binarized X. One dense Cholesky inversion (`faer`), f64 math / f32
storage. λ defaults to the heuristic `20 · nnz / n_items`; explicit
`ease_lambda` pins it (amazon-beauty measures slightly better at 250
than its auto ≈ 299).

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
| timestamps present | trend, transitions | nothing to compute otherwise (amazon-book academic split correctly no-ops) |
| `rating_burst_detected` | transitions OFF | within-burst order is noise — ml1m measurably hurt by transitions at every weight |
| `item_metadata` present + `content_alpha > 0` | content channel | opt-in; see §4.6 |
| per-item coldness | content weight | content dilutes warm ranking when ungated (§4.6) |

The last-item channel is deliberately **not** burst-gated: it reads the
EASE row (co-occurrence structure) of the newest item, not within-burst
order, and it helps on ml1m where raw transitions hurt.

## 3. Methodology

### 3.1 Canonical protocol

Chronological 90/10 split, strided-500 eval users by sorted entity_id,
full-catalog ranking (no sampled negatives), k=10. Harness:
`kindling.benchmarks.parity._build_eval_set`.

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

| | NDCG@10 | MRR | recall@10 | HR@10 | p50 |
|---|---:|---:|---:|---:|---:|
| ml1m | **0.2879** | 0.4723 | 0.0579 | 0.744 | 0.6ms |
| amazon-beauty (λ=250) | **0.0312** | 0.0414 | 0.0429 | 0.096 | 0.4ms |
| amazon-book† | 0.0253 | 0.0563 | 0.0246 | 0.140 | 1.0ms |

† amazon-book is the LightGCN *academic* split locally: no timestamps,
non-chronological, 91k items (above the EASE gate) — a different
protocol family. Its channels no-op by design; compare it only against
itself.

Progression on ml1m: 0.2561 (raw cooc) → ~0.269 (EASE) → 0.2841
(+trend) → 0.2879 (+last-item). Beauty: 0.0290 → 0.0306 (+EASE+trend)
→ 0.0315 (+transitions) → 0.0312/recall 0.0429 (+last-item; NDCG noise,
recall +4.6%).

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
perfectly protective but unrewarded because the canonical protocol's
held-out items are never cold. Beauty's metadata covers 0.17% of its
catalog (2023 metadata vs 2014 reviews) — inert there regardless.
`content_alpha=0` until a cold-start protocol exists; the capability is
ready.

## 5. Engine knobs (the ones that matter)

| knob | default | touch when |
|---|---|---|
| `base_scorer` | `"auto"` | force `"ease"`/`"cooc"` for experiments |
| `ease_max_items` | 20 000 | more RAM/patience → raise |
| `ease_lambda` | None (auto `20·nnz/n_items`) | beauty-like datasets may prefer ~250 |
| `trend_alpha` | 0.5 | 0 to disable; window via `trend_window_fraction` (0.10) |
| `transition_alpha` | 0.25 | auto-gated off on burst datasets |
| `last_item_alpha` | 0.25 | 0.5 overshoots everywhere measured |
| `content_alpha` | 0.0 | enable on cold-start protocols with real metadata |
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
  item_features.py    schema-inferring extractor + content_scores
  benchmarks/
    parity.py                   canonical eval-set builder
    gap_decomposition.py        floor / oracle / pool-recall diagnostic
    clustering_coherence_sweep.py, persona_diff.py, ...   (persona-era diagnostics)
```

## 7. Open fronts

1. **amazon-book chronological protocol** — requires the McAuley 5-core
   Books JSONL (multi-GB). Would put book on the canonical protocol,
   activate trend/transitions there, and provide a real cold-start
   surface for the content channel.
2. **Cold-start eval protocol** — the content channel and coldness
   gating are built but unrewarded by the canonical splits; a protocol
   that holds out cold items would measure them honestly.
3. **Remaining oracle headroom** — ml1m oracle on the same pool is 0.88
   vs 0.2879 current. Closing more of it likely requires real sequence
   modeling (out of scope today) or richer user-state features.
4. **EASE beyond the gate** — blocked/low-rank EASE variants could
   extend the closed-form base past 20k items if book-scale catalogs
   become first-class.
