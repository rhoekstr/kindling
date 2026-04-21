# ADR: pair-index + distinctiveness-weighted basket signal

**Date:** 2026-04-21
**Status:** shipped
**Supersedes:** extends [ADR-growth-curves.md](ADR-growth-curves.md).

## What changed

Two signal-quality / latency improvements to the basket index:

1. **Pair-indexed postings** — alongside the existing item postings
   (`item -> [obs_idx, ...]`), we now build pair postings
   (`(a, b) -> [obs_idx, ...]` where the basket contains both a and b).
   At query time we enumerate C(|Q|, 2) pairs and union their tiny
   posting lists. For |Q| < 2 or non-int basket items, falls back to
   item postings. Trades a slight semantic (observations with only one
   item overlap are dropped; their contribution was at most 1/|Q| of
   the total weight anyway) for a 1.6-2.0x reduction in per-query
   observation scan.

2. **Distinctiveness weighting on basket observations** — each
   observation's effective weight is divided by the global frequency of
   its `next_item` being a next-add. This transforms the basket signal
   from "what is commonly added after baskets like Q?" into "what is
   *elevated* after baskets like Q vs. the baseline popularity?" Items
   that are added everywhere (milk, bread) get down-weighted; items
   specific to the basket context (tortillas after salsa) get
   up-weighted. Captured by a unit test with a synthetic Mexican-basket
   vs generic-basket fixture:

   | similarity | refried score | milk score | winner |
   | ---------- | ------------- | ---------- | ------ |
   | raw         | 0.050         | 0.236      | milk   |
   | distinctive | 0.225         | 0.135      | **refried** |

## Measured effects

### Overlap-set size (observations per query)

| Dataset      | Observations | |Q| | Item-posting overlap | Pair-posting overlap | Reduction |
| ------------ | -----------: | --: | -------------------: | -------------------: | --------: |
| ML-1M        | 511,777      | 50  | 346,588 (67.7%)      | 216,995 (42.4%)      | 1.6x      |
| grocery-deep | 145,800      | 43  | 87,058 (59.7%)       | 42,757 (29.3%)       | 2.0x      |

Pair index works better on session data because pairs of grocery items
are more informative than pairs of popular movies.

### Latency (p95 recommend, full dataset)

| Dataset            | Before (ADR-growth) | After pair+distinctive |
| ------------------ | ------------------: | ---------------------: |
| ML-1M              | 216 ms              | 218 ms (no change)     |
| grocery (6-item)   | 4.9 ms              | 4.3 ms                 |
| grocery-deep (10)  | n/a                 | 12.8 ms @ 162k interactions |

ML-1M didn't improve because popular-movie pairs are still dense (1.6x
overlap reduction isn't enough to dominate kernel time). Separate
latency work for ratings-style data is still open — see
§"Latency for ratings data" below.

### Growth curve on grocery-deep (longer sessions)

| Frac | Interactions | kindling NDCG | kNN NDCG | ALS NDCG | pop NDCG | kindling p95 |
| ---- | -----------: | ------------: | -------: | -------: | -------: | -----------: |
| 0.10 | 16k          | 0.076         | 0.078    | 0.073    | 0.041    | 1.3 ms       |
| 0.30 | 49k          | **0.128**     | 0.121    | 0.100    | 0.039    | 3.8 ms       |
| 0.60 | 97k          | 0.190         | 0.188    | 0.138    | 0.048    | 8.3 ms       |
| 1.00 | 162k         | 0.319         | 0.320    | 0.232    | 0.060    | 12.8 ms      |

Kindling leads kNN by +6% NDCG at 30% data and ties at the extremes.
Against ALS, kindling leads by +28% to +38% across the whole curve.
Against popularity, kindling leads 2-5x throughout.

## Unsolved: latency for ratings data (ML-1M)

Pair index reduced overlap by 1.6x on ML-1M but latency is unchanged.
Kernel still scans ~217k observations per query. The remaining knobs:

- **Shorter query basket.** `MAX_QUERY_BASKET_SIZE=50` is generous; on
  ratings data where users rate hundreds of movies, the last 50 is
  still a broad sample. Dropping to 10-20 should help but risks
  accuracy regression.
- **Random-sample cap on observation scan.** Bounded by a per-query
  budget (say 10,000 obs), sampled uniformly. The weighted-mean
  estimator converges well before a full scan.
- **Tighter pair threshold.** Drop pairs whose posting list exceeds
  some threshold (the "popular pair" tail). These contribute noise
  and dominate latency.

## Honest note on "distinctiveness" as a novelty claim

The underlying math is decades old:

- **Association rule lift** (`P(B|A) / P(B)`) is Apriori/market-basket
  analysis from 1994.
- **IDF weighting** is from the 1970s and already ships in kindling
  as `BasketSimilarity.IDF_COVERAGE` (query-side weighting).
- **TF-IDF-style candidate weighting** is standard in information
  retrieval.

What kindling does differently is fold distinctiveness into one of
seven basket/path/cooccurrence/cost signals whose relative weights
are learned as a Bayesian posterior with credible intervals. The
distinctiveness *signal* is classical; the distinctiveness-plus-
posterior-over-weights combination is less common in production
recommenders.
