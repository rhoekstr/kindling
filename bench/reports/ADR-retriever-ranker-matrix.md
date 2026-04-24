# ADR: per-fraction retriever/ranker decomposition

**Date:** 2026-04-23
**Status:** diagnostic shipped; validates user's recall-vs-ranking hypothesis
**Related:** [ADR-standalone-retrievers.md](ADR-standalone-retrievers.md),
[ADR-retriever-union.md](ADR-retriever-union.md)

## Why this exists

Standalone retriever eval (ADR-standalone-retrievers) tested each signal
as a full recommender — its own retriever + its own ranking. That told
us which signals surface the right items AND rank them well together,
but didn't let us see where recall and ranking diverge.

This matrix decouples the two. For each pair (retriever R, ranker K), we
use R's retrieval output but re-score those candidates with K's scoring
function. That answers: "which signal gets the right items in the
candidate pool (retrieval)" and separately "which signal puts the right
items at the top (ranking)" — and whether cross-combining them helps.

Plus we sweep 5 training fractions (20/40/60/80/100%) so we see how
each signal's contribution changes with data volume.

## Standalone per-fraction (retriever = ranker)

### grocery-deep (NDCG@10 / recall@10)

| signal | 0.20 | 0.40 | 0.60 | 0.80 | 1.00 |
| -------- | ---: | ---: | ---: | ---: | ---: |
| cooccurrence | 0.169/0.648 | 0.213/0.707 | 0.252/0.753 | 0.288/0.752 | **0.319/0.738** |
| item_item_cosine | 0.169/0.634 | 0.208/0.709 | 0.245/0.759 | 0.285/0.746 | **0.320/0.742** |
| als_factor | 0.158/0.670 | 0.187/0.719 | 0.231/0.739 | 0.251/0.740 | 0.295/**0.757** |
| path_basket | 0.164/0.642 | 0.206/0.685 | 0.238/0.737 | 0.259/0.686 | 0.304/0.732 |
| persona | 0.154/0.602 | 0.198/0.683 | 0.236/0.747 | 0.277/0.764 | 0.315/0.751 |
| path_tail | 0.134/0.566 | 0.151/0.509 | 0.173/0.540 | 0.171/0.473 | 0.181/0.474 |
| path_full | 0.023/0.106 | 0.022/0.114 | 0.033/0.149 | 0.041/0.158 | 0.047/0.178 |

Top four signals (cooc, cosine, persona, path_basket) grow together
from ~0.17 at 20% to ~0.30 at 100%. All earn their place on session
data.

### ml1m (NDCG@10 / recall@10)

| signal | 0.20 | 0.40 | 0.60 | 0.80 | 1.00 |
| -------- | ---: | ---: | ---: | ---: | ---: |
| cooccurrence | 0.144/0.529 | 0.147/0.524 | 0.166/0.554 | 0.174/0.574 | **0.183/0.596** |
| item_item_cosine | 0.145/0.513 | 0.150/0.536 | 0.166/0.584 | 0.167/0.552 | **0.183/0.592** |
| als_factor | 0.136/0.534 | 0.134/0.530 | 0.157/0.568 | 0.151/0.552 | 0.163/0.586 |
| **persona (fixed)** | 0.132/0.577 | 0.091/0.441 | 0.138/**0.604** | 0.127/0.544 | 0.142/0.576 |
| path_tail | 0.062/0.307 | 0.066/0.355 | 0.083/0.394 | 0.073/0.356 | 0.088/0.424 |
| path_basket | 0.038/0.212 | 0.045/0.225 | 0.039/0.224 | 0.048/0.270 | 0.050/0.270 |
| path_full | 0.005/0.021 | 0.007/0.038 | 0.008/0.050 | 0.014/0.078 | 0.018/0.090 |

Persona on ml1m — now **0.142** at 1.00 after the cold-start scale
fix, not 0.0002. Session-specific retrievers (path_basket) stay weak
across all fractions. path_tail plateaus.

## Recall@K vs NDCG@K on the same retriever

The numbers that most cleanly illustrate the user's "retrieval quality
≠ ranking quality" insight:

| dataset | signal | rec@K | NDCG | delta |
| ------- | ------ | ----: | ---: | ----: |
| grocery-deep | persona | 0.751 | 0.315 | high recall, comparable NDCG |
| grocery-deep | als | 0.757 | 0.295 | **highest recall, mid-tier NDCG** — ALS finds the right items but puts them in the wrong order |
| ml1m | persona | 0.576 | 0.142 | **high recall, low NDCG** — persona's cluster-level pooling loses per-item distinctions |
| ml1m | als | 0.586 | 0.163 | high recall, mid NDCG |

Persona on ml1m is the cleanest example: it surfaces the correct item
in the top-10 **58% of the time** (comparable to cooccurrence at 60%)
but NDCG is only 0.142 vs cooc's 0.183. The items are IN the top-10,
just not in the top-3. Cluster-level pooling collapses per-item
distinctions.

## Retriever × Ranker at 100% data

The interesting half. Each row is "use R to retrieve, use K to score
those candidates, top-10 by K's score."

### grocery-deep (top-8 by NDCG)

| retriever | ranker | NDCG | rec@K |
| --------- | ------ | ---: | ----: |
| als_factor | cooccurrence | 0.320 | 0.742 |
| cooccurrence | item_item_cosine | 0.320 | 0.742 |
| path_basket | item_item_cosine | 0.320 | 0.742 |
| als_factor | item_item_cosine | 0.320 | 0.742 |
| persona | item_item_cosine | 0.320 | 0.742 |
| path_basket | cooccurrence | 0.319 | 0.738 |
| item_item_cosine | cooccurrence | 0.319 | 0.738 |
| persona | cooccurrence | 0.319 | 0.738 |

**On grocery at 100% data, every high-recall retriever + neighborhood
ranker gives the same NDCG (~0.320).** That's because every retriever
hits 100% recall@budget — they all surface the same candidate pool,
so ranking is what matters, and cooc/cosine are the best rankers.

### ml1m (top-8 by NDCG)

| retriever | ranker | NDCG | rec@K | rec@B |
| --------- | ------ | ---: | ----: | ----: |
| als_factor | cooccurrence | **0.184** | 0.608 | 0.982 |
| **persona** | item_item_cosine | **0.184** | 0.592 | 0.958 |
| als_factor | item_item_cosine | 0.183 | 0.598 | 0.982 |
| cooccurrence | item_item_cosine | 0.183 | 0.592 | 0.958 |
| persona | cooccurrence | 0.182 | 0.594 | 0.958 |
| item_item_cosine | cooccurrence | 0.182 | 0.596 | 0.976 |
| persona | als_factor | 0.164 | 0.584 | 0.958 |
| cooccurrence | als_factor | 0.163 | 0.586 | 0.958 |

**Two cross-combinations tie for the win on ml1m:**

1. `als_factor retriever + cooccurrence ranker` (0.184)
2. `persona retriever + item_item_cosine ranker` (0.184)

Both beat the cooc-alone baseline (0.183) by a hair. More importantly,
both use a **different retriever from the default**, meaning they
surface a different candidate pool, and the ranker extracts equally
good top-10 from it.

Specifically, `persona retriever + cosine ranker` is the user's
hypothesis literalized: persona has high recall (cluster-level taste
matching surfaces plausibly relevant items), cosine has precise
ranking (direct item-item affinity). Combined: 0.184. Standalone
persona: 0.142. Standalone cosine: 0.183. Combined matches or
barely beats the stronger of the two ingredients.

The magnitude is tiny on ml1m — +0.001 NDCG — because all viable
retrievers converge to the same ceiling once ranking is done well.
The lesson isn't "cross-combine for huge NDCG wins"; it's that
**retrievers are interchangeable when ranked well, and rankers are
where the precision lives**.

## What this implies for the engine architecture

Three things change how we should think about the current stack:

### 1. Retrieval is about coverage; ranking is about precision.

Already what we've been doing architecturally, but the matrix confirms
it empirically. A good engine runs multiple retrievers to maximize
recall@budget, then a precise ranker to extract the top-10. **The
Bayesian blend (stage 2 scoring) IS the ranker in this framing** —
it's where precision comes from. The fact that it collapses to
cooccurrence under the blend (ADR-signal-audit) is saying "cooccurrence
is kindling's best ranker on offline interaction data."

### 2. Cross-combinations don't buy much at full data — the gain comes from recall@budget where retrievers diverge.

Every retriever that hits rec@B = 1.000 is interchangeable when
ranked with cooc/cosine. The combinations that DO differ are the ones
at low-recall regimes: path_tail + als at rec@B=0.798 can't be
rescued by a good ranker because the right items aren't in the pool.

The useful architecture therefore is:
- **Union retrievers** (via RRF) to push recall@budget as high as
  possible across conditions.
- **Rank with cooc** (or a linear blend where cooc dominates — which
  is what we have).

### 3. Persona isn't dead; it's a retriever.

As a signal in the blend, persona is redundant with cooc (ADR-persona).
As a retriever, persona's recall@K on both datasets is competitive
(0.751 grocery, 0.576 ml1m). The engine's current stack already
includes PersonaRetriever on session data. This matrix validates that
decision.

## What's shipping in this commit

- `src/kindling/benchmarks/retriever_matrix.py`: the new harness.
- Two reports: `retriever_matrix_grocery.json`, `retriever_matrix_ml1m.json`.
- This ADR.
- No engine changes — the current engine architecture (stack +
  RRF + blend) is already consistent with what the matrix shows.

## Queue after this

1. Run end-to-end growth curves with the new engine to measure
   whether the architecture changes shipped in commits 1-4 move
   NDCG or latency relative to the previous baseline.
2. Repeat-consumption module per the user's design doc — owned
   items filtering is a subset of what that module does (and does
   it at the wrong architectural level).
