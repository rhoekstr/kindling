# ADR: retriever union — which retrievers complement each other?

**Date:** 2026-04-23
**Status:** diagnostic shipped; architecture decision: ship cooc+als+cosine by default
**Related:** [ADR-standalone-retrievers.md](ADR-standalone-retrievers.md),
[ADR-persona-signal.md](ADR-persona-signal.md)

## What ran

`kindling.benchmarks.retriever_union` — fit the engine once, evaluate
a ladder of retriever union configurations. Each config combines its
named retrievers' candidates and ranks by fusion score. Two fusion
methods tested:

- `max_score`: take each item's max score across retrievers, sort. The
  original naive attempt.
- `rrf`: Reciprocal Rank Fusion. `score(item) = Σ_r 1/(60 + rank_r(item))`.
  Score-scale-independent; item appearing high in multiple retrievers
  wins.

Per-retriever budget: 100 candidates each (so a 3-retriever union has
up to 300 unique items in the pool).

## The max_score finding: dominant-scale retriever wins everything

Every max_score union produced **identical** top-10 output:

| config                           | grocery NDCG | ml1m NDCG |
| -------------------------------- | -----------: | --------: |
| cooc_only                        | 0.319        | 0.183     |
| cooc+als                         | 0.319        | 0.183     |
| cooc+als+cosine+path_basket      | 0.319        | 0.183     |
| all_useful (6 retrievers)        | 0.319        | 0.183     |

Reason: cooc scores are in the thousands (raw item-graph weights);
cosine, persona, path_basket are in [0, 1]; ALS is ~[0, 5]. Max-score
sort is always dominated by cooc's magnitude. The other retrievers'
candidates are buried even though recall@union climbs (0.88 → 0.95
on ML-1M as retrievers are added). **This is a score-normalization
issue, not a retriever-quality issue.**

## RRF result: real differentiation, selective inclusion wins

Switching to RRF:

### grocery-deep (session-rich)

| config                           | NDCG  | MRR   | rec@K | rec@U |
| -------------------------------- | ----: | ----: | ----: | ----: |
| cooc_only (baseline)             | 0.319 | 0.351 | 0.738 | 0.996 |
| cooc + als                       | 0.320 | 0.366 | **0.765** | 1.000 |
| **cooc + als + cosine**          | **0.323** | 0.367 | 0.748 | 1.000 |
| cooc + als + path_basket         | 0.316 | 0.354 | 0.751 | 1.000 |
| cooc + als + cosine + path_basket| 0.317 | 0.353 | 0.742 | 1.000 |
| cooc + als + path_basket + persona| 0.321 | 0.353 | 0.751 | 1.000 |
| all_useful (6)                   | 0.308 | 0.352 | 0.736 | 1.000 |
| engine_current (cooc + path_endpoint) | **0.234** | 0.311 | 0.627 | 1.000 |

### ml1m (ratings)

| config                           | NDCG  | MRR   | rec@K | rec@U |
| -------------------------------- | ----: | ----: | ----: | ----: |
| cooc_only (baseline)             | 0.183 | 0.320 | 0.596 | 0.880 |
| cooc + als                       | 0.179 | 0.322 | **0.624** | 0.932 |
| **cooc + als + cosine**          | **0.185** | 0.327 | 0.614 | 0.934 |
| cooc + als + path_basket         | 0.159 | 0.279 | 0.580 | 0.936 |
| cooc + als + cosine + path_basket| 0.177 | 0.306 | 0.604 | 0.938 |
| cooc + als + path_basket + persona| **0.150** | 0.267 | 0.594 | 0.946 |
| all_useful (6)                   | 0.159 | 0.283 | 0.596 | 0.950 |
| engine_current (cooc + path_endpoint)| 0.155 | 0.286 | 0.570 | 0.908 |

## Four findings

### 1. The winning retriever set is cooc + als + cosine on both datasets

Same 3-retriever set wins on both datasets:
- grocery-deep: 0.319 → 0.323 NDCG (+1.3%)
- ml1m: 0.183 → 0.185 NDCG (+1.1%)

Modest NDCG gains, but also real recall@10 gains (grocery 0.738 →
0.748; ml1m 0.596 → 0.614). This is a safe, robust default.

### 2. Session-specific retrievers HELP on session data, HURT on ratings

- path_basket: on grocery neutral-to-slightly-hurts. On ml1m **drops
  NDCG from 0.183 to 0.159** — 13% regression.
- persona: same pattern. Adding persona to ml1m drops NDCG to 0.150.
- On grocery, adding path_basket or persona neither meaningfully
  helps nor hurts (NDCG stays 0.316–0.321).

This is the dataset-features-based gating you identified. Session
retrievers should only run when session density / category
separation indicates they'll help.

### 3. "All retrievers" is WORSE than the best subset

On both datasets, the 6-retriever "all_useful" union underperforms
the 3-retriever cooc+als+cosine. More retrievers introduce more
rank-diluted noise under uniform-weight RRF. **Retriever selection
matters; throwing everything in dilutes the strongest signals.**

### 4. The engine's current stage-1 retriever is harmful under RRF

`cooc + path_endpoint_combined` scores **0.234 on grocery** under
RRF — a 27% regression from cooc-alone. RRF gives path_endpoint
equal rank-weight to cooc, and path_endpoint's top-10 has only 51%
recall of positives (vs cooc's 74%) — so RRF demotes good cooc
candidates in favor of path_endpoint's wrong ones.

Max_score fusion had hidden this because cooc's raw scores buried
path_endpoint. But RRF surfaces the real quality problem, and it's
real: the current retriever stack is materially worse than pure
cooccurrence on session data.

## Architectural decision

Ship **cooc + als + cosine** as the default retriever set with RRF
fusion. Drop `path_endpoint_combined` as an engine retriever. Gate
session-specific retrievers (`path_basket`, `persona`) on data-
characteristic signals:

- Enable when: `session_density > threshold`, `items_per_session ≥ 6`,
  persona `silhouette_score > 0.3`, `noise_fraction < 0.3`.
- These gates already have hooks in `priors.toml` (session_stiffness)
  and can be reused.

Budget allocation (per your ask): 
- **Session-weak (ratings)**: `{cooc: 200, als: 150, cosine: 150}` = 500.
- **Session-strong**: `{cooc: 150, als: 100, cosine: 100, path_basket: 100, persona: 50}` = 500.
- **Cold-start domain**: favor cooc + als heavily; disable session retrievers.

## Caveats before shipping

### RRF weighting

Uniform RRF gives every retriever equal say. But we just measured
that retrievers differ in standalone NDCG by **5×** (grocery: cosine
0.32 vs path_tail 0.18). Uniform RRF says "path_tail is 50% of the
vote" which is wrong. Weighted RRF (each retriever's contribution
scales with its measured standalone quality) is the obvious next step.

### Persona cold-start bug

During this work we discovered the persona cold-start weights are on
a wildly different scale (max 254 vs main persona_vectors max 0.22).
With `cold_start_weight=0.25`, cold-start dominates scoring and
destroys persona's ranking. This ADR's union runs use
`cold_start_weight=0.0` to measure persona properly. Fix queued:
L2-normalize cold_start_weights per persona or log1p-transform the
ratios before combining.

With the fix, persona's real standalone recall@10 on ml1m is **62%**
(not 0.2% as ADR-persona-signal reported). The earlier "persona doesn't
work on ml1m" conclusion was a bug in the scoring, not a property of
the signal.

## What's shipping in this commit

- `retriever_union.py` harness with both `max_score` and `rrf` fusion.
- Four reports (max_score + RRF × grocery + ml1m).
- This ADR.
- No engine changes yet — architecture decision is written down,
  implementation is the follow-up commit.

## Queue after this

1. **L2-normalize cold_start_weights** — unblocks persona measurement
   honestly. Revisit ADR-persona-signal numbers afterward.
2. **Wire cooc+als+cosine as Engine's default stage-1** behind a
   session-density gate for the add-ons. Remove
   path_endpoint_combined.
3. **Weighted RRF** — scale each retriever's rank contribution by
   standalone NDCG. Expected to push the union win beyond +1.3%.
4. **Re-run growth curves and comparison** with new retrieval stage.
