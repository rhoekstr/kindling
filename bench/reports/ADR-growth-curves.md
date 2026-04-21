# ADR: growth-curve and session-rich comparison

**Date:** 2026-04-21
**Status:** informational
**Supersedes context in:** ADR-baselines-comparison.md (single-point
  comparison at 100% data; this ADR extends it across the growth curve
  and to a session-rich dataset)

## What ran

Two harnesses:

1. **Comparison on synthetic-grocery** (session-rich by construction) —
   1500 entities, 10 sessions each, 6 items per session, 160 items.
   Report: [baselines_comparison_grocery.json](baselines_comparison_grocery.json).

2. **Growth-curve sweep** on MovieLens-1M and synthetic-grocery —
   chronological prefixes at 10%, 30%, 60%, 100% of training data, with
   eval entities fixed across fractions. 500 evaluated entities per
   fraction. Reports:
   [growth_curve_movielens.json](growth_curve_movielens.json),
   [growth_curve_grocery.json](growth_curve_grocery.json).

Also: the native basket-scoring kernel added in Phase 8 was not wired;
this sprint wired it and dropped kindling's p95 from 247ms → 216ms on
full ML-1M. Further latency work is still needed (see §6).

## Session-rich result (synthetic-grocery, 100% data)

| Model          | NDCG   | Recall  | MRR    | Coverage | Fit (s) | p95 (ms) |
| -------------- | ------ | ------- | ------ | -------- | ------- | -------- |
| **kindling**   | **0.243** | **0.402** | **0.235** | **1.000** | 3.4 | 4.9 |
| popularity     | 0.053  | 0.085   | 0.069  | 0.150    | <0.1    | <0.1     |
| item_item_knn  | 0.236  | 0.403   | 0.230  | 1.000    | <0.1    | <0.1     |
| implicit_als   | 0.169  | 0.263   | 0.203  | 1.000    | 0.2     | <0.1     |

**Read:** on the dataset kindling is designed for, it edges out
item-item kNN and dominates ALS by 44% NDCG. Catalog coverage is
perfect. This is the honest positioning — kindling is for session data.
ML-1M (ratings, no sessions) was the wrong test in the first ADR.

## Growth curves

### MovieLens-1M (ratings, no sessions)

| Frac | Interactions | kindling NDCG | pop NDCG | kNN NDCG | ALS NDCG | kindling p95 |
| ---- | ------------ | ------------- | -------- | -------- | -------- | ------------ |
| 0.10 | 52k          | 0.000         | 0.116    | 0.000    | 0.000    | <0.1 ms      |
| 0.30 | 155k         | 0.014         | 0.116    | 0.016    | 0.012    | 53 ms        |
| 0.60 | 311k         | 0.061         | 0.133    | 0.071    | 0.054    | 121 ms       |
| 1.00 | 518k         | 0.181         | 0.172    | 0.196    | 0.154    | 216 ms       |

### Synthetic-grocery (sessions by construction)

| Frac | Interactions | kindling NDCG | pop NDCG | kNN NDCG | ALS NDCG | kindling p95 |
| ---- | ------------ | ------------- | -------- | -------- | -------- | ------------ |
| 0.10 | 8k           | 0.073         | 0.047    | 0.078    | 0.076    | 0.8 ms       |
| 0.30 | 24k          | 0.106         | 0.049    | 0.106    | 0.088    | 1.5 ms       |
| 0.60 | 49k          | 0.150         | 0.053    | 0.150    | 0.113    | 3.1 ms       |
| 1.00 | 81k          | 0.243         | 0.053    | 0.236    | 0.169    | 4.9 ms       |

## Five honest findings

### 1. Kindling has no cold-start fallback

At 10% ML-1M data, most eval entities aren't in the training subsample,
so `engine.recommend(entity_id=...)` returns `[]`. That's why kindling
scores 0.000 NDCG at 10%. Popularity scores 0.116 because it doesn't
need the entity to exist. **Fix:** fall back to a popularity list (or
a session-aware cold-start) when entity has no interactions. This is
the single cheapest accuracy win available.

### 2. Kindling and item-item kNN track each other throughout

On both datasets and every fraction, kindling's NDCG is within 3% of
kNN. The Bayesian blend + seven signals + rerank stack is not adding
measurable value over simple item-item cosine on these datasets. This
suggests one of:

- Co-occurrence + popularity dominate the blend and the five other
  signals are near-zero weights.
- The other signals carry signal but the decorrelation + prior
  stiffness absorb it.
- The datasets aren't sensitive to the value kindling claims to add
  (calibrated uncertainty, explanations, coverage tradeoffs).

Worth a prior-sensitivity study on the blend before accepting that the
signals don't help.

### 3. Kindling beats ALS on sessions, loses to popularity on ratings at < 100% data

On synthetic-grocery, kindling leads ALS by +44% NDCG at 100% data and
+23% at 10% data. On ML-1M, kindling underperforms simple popularity
at every fraction below 100%. That's a surprising and worth-fixing
result — popularity is supposed to be the floor. Likely tied to
finding #1 (cold starts) plus finding #2 (blend not adding value).

### 4. Kindling's latency grows with data; baselines' doesn't

On ML-1M, p95 recommend scales from 53ms (30%) → 216ms (100%).
Baselines stay under 0.5ms throughout. The per-recommend cost is
dominated by the basket-index scan, which scales with posting-list
size. On synthetic-grocery (160 items) the kernel is fast because
posting lists are small.

**Implication:** kindling's latency story is a **catalog-size** story,
not a data-volume story in the simple sense. Large-catalog session
datasets (e.g., Amazon) will be worst-case. Phase 11 needs an
algorithmic fix, not just Rust — likely a per-recommend cap on
observation-scan size (truncate after N observations, the estimator
converges well before full scan).

### 5. Coverage tradeoffs are stable

ALS consistently surfaces more of the catalog than anyone else on
ML-1M (16% vs. kindling's 4%). On synthetic-grocery, all three modern
models hit 100% coverage — the dataset is too small to differentiate.
This supports the earlier finding: ALS is the coverage leader, and
kindling's popularity bias at low temperature hurts coverage.

## Recommendations (in order)

1. **Cold-start fallback** — cheapest, single largest accuracy win at
   low data. Fall through to popularity when entity is unseen.
2. **Prior-sensitivity study** — diagnose whether the blend is adding
   value or the signals are being absorbed. Requires bootstrapping the
   blend weights and checking stability.
3. **Latency cap on basket-index scan** — per-recommend observation
   budget with random sampling beyond it. Estimator converges; scan
   scales.
4. **Item-item kNN as an 8th signal** — if finding #2 is real and the
   blend doesn't add value over cosine, the fix is to *include*
   cosine in the blend and let the posterior downweight everything
   else on ratings data.
5. **Latent-factor signal** — add an ALS-derived retriever as a 9th
   signal to pick up the coverage kindling currently lacks.

## What's still not tested

- RetailRocket and Amazon (real session-rich datasets) — we don't have
  the data cached. Synthetic-grocery is a stand-in but not a
  replacement for real behavioral data.
- Online / sequential updates. All measurements here are offline
  retrain.
- Recommendations under constraints, temperature, diversity — the
  default call path only.
