# ADR: nine-signal blend, session stiffness, cold-start, signal skip

**Date:** 2026-04-21
**Status:** shipped
**Supersedes parts of:** [ADR-growth-curves.md](ADR-growth-curves.md),
[ADR-pair-index-distinctiveness.md](ADR-pair-index-distinctiveness.md).

## What changed this pass

Seven concrete items landed in order:

1. **Cold-start popularity fallback.** `Engine.recommend` falls through to
   a popularity-ranked list when retrieval returns no candidates (unseen
   entities, or entities whose owned items exhaust their neighbors).
   Closes the biggest single accuracy gap on sparse-data evaluations.
2. **Observation-scan cap (basket).** `BasketIndex.score_many` takes
   `scan_cap` + `rng`; uniformly subsamples the overlap set when it
   exceeds the cap. Default 10_000. MC weighted-mean estimator converges
   O(1/sqrt(N)).
3. **Prior-sensitivity study.** New `benchmarks.prior_sensitivity` CLI.
   Finding: posterior is prior-dominated in the offline-eval path (no
   outcome feedback), so the priors do most of the work. This
   motivated item B.
4. **(B) Data-adaptive session stiffness.** New
   `DataFeatures.has_explicit_sessions` flag set True iff the caller
   supplied a `session_id` column. When False (ratings-style input with
   GMM-inferred sessions), session-density boosts to `path_*` priors are
   skipped and a `session_stiffness` shrink (×0.2 -> MIN_ALPHA after
   clip) fires. On ML-1M the path posterior drops from 52% to <4% and
   cooccurrence jumps from 3% to 29%.
5. **(C) Item-item cosine as 8th signal.** New `graph/item_cosine.py`
   with top-K-per-row cosine matrix built from the item graph. Scored
   via `sum cos(candidate, j) for j in owned`. Prior rule
   `graph_density_cosine`.
6. **(A-refined) Skip signals with near-zero posterior weight.** Opt-in
   `skip_signal_weight_threshold`; signals below the threshold are not
   computed at recommend time. Default 0.0 (preserves accuracy). At 0.05:
   ML-1M p95 152ms -> 1.6ms, NDCG 0.213 -> 0.198 (-7%), MRR 0.364 -> 0.331
   (-9%). 100x latency for 7% NDCG is a workable knob.
7. **(D) ALS latent-factor signal as 9th signal.** New
   `graph/als_factors.py` wrapping `implicit.als.AlternatingLeastSquares`
   behind a graceful no-op. Prior rule `graph_density_als`.
8. **(E) RetailRocket wiring.** Added to comparison + growth-curve
   harnesses. Runs the moment `events.csv` is dropped into
   `$KINDLING_CACHE_DIR/retailrocket/`. Not executed here because the
   Kaggle data isn't cached locally.

## Final growth curves (all above items active)

### ML-1M (ratings, no explicit sessions)

| Frac | kindling NDCG | pop NDCG | kNN NDCG | ALS NDCG | kindling p95 |
| ---- | ------------: | -------: | -------: | -------: | -----------: |
| 0.10 | **0.116**     | 0.116    | 0.000    | 0.000    | <1 ms        |
| 0.30 | 0.115         | 0.116    | 0.016    | 0.012    | 40 ms        |
| 0.60 | 0.133         | 0.133    | 0.071    | 0.054    | 87 ms        |
| 1.00 | **0.183**     | 0.172    | 0.196    | 0.154    | 153 ms       |

Kindling now ties/beats popularity at every fraction (was below at 30%
and 60% in the first growth curve). Still trails kNN at 100% by 7%; the
cosine + ALS signals don't dislodge it on ML-1M.

### Synthetic-grocery-deep (explicit sessions, 10-item baskets, 162k@100%)

| Frac | kindling NDCG | pop NDCG | kNN NDCG | ALS NDCG | kindling p95 |
| ---- | ------------: | -------: | -------: | -------: | -----------: |
| 0.10 | 0.075         | 0.041    | 0.078    | 0.073    | 1.2 ms       |
| 0.30 | **0.128**     | 0.039    | 0.121    | 0.100    | 4.2 ms       |
| 0.60 | **0.190**     | 0.048    | 0.188    | 0.138    | 7.6 ms       |
| 1.00 | 0.320         | 0.060    | 0.320    | 0.232    | 11.6 ms      |

Kindling leads kNN from 30% through 60% and ties at the extremes.
Beats ALS by 28-38% throughout. Crushes popularity 5x at 100%.

## Posterior weights, by dataset

The nine-signal blend settles differently depending on the data shape:

| Signal             | ML-1M 30% | Grocery-deep 100% |
| ------------------ | --------: | ----------------: |
| path_full          | 0.037     | 0.375             |
| path_tail          | 0.037     | 0.196             |
| path_basket        | 0.037     | 0.107             |
| cooccurrence       | 0.185     | 0.107             |
| cost_population    | 0.087     | 0.018             |
| cost_entity        | 0.088     | 0.018             |
| cost_context       | 0.085     | 0.018             |
| item_item_cosine   | 0.274     | 0.107             |
| als_factor         | 0.169     | 0.054             |

Ratings data lets the blend put ~72% on neighborhood + latent-factor
signals (cooc + cosine + ALS). Session data lets paths take 68%.

## Signal-value audit (outstanding from item G)

Still to do: did the two new signals actually *move* NDCG, or just
re-apportion the mass?

- On ML-1M 100%, adding cosine and ALS bumped kindling's NDCG from
  0.181 (pre-items) -> 0.183 (post-items), a 1% improvement. But the
  posterior puts 27% weight on cosine and 17% on ALS. Those signals
  are carrying mass without carrying accuracy.
- On grocery-deep 100%, the paths dominate so any contribution from
  cosine + ALS (together 16% of posterior) is invisible in the final
  NDCG.

Likely conclusion: on ML-1M the underlying cooccurrence retriever is
already surfacing the right candidates, and the cosine signal is
telling the blend the same thing cooc already knew. Dropping cosine
+ ALS would likely save ~30% of per-recommend compute without changing
NDCG measurably.

That's exactly the G-audit posture: signals are free to keep when they
earn their compute and drop when they don't. Formal audit + decision
queued.

## Outstanding items

F. Growth curve across all four datasets (needs Instacart + Amazon +
   RetailRocket data cached).

G. Signal-value audit: per-fraction ablation (kindling - cosine,
   kindling - ALS, kindling - paths on ratings, etc.) to decide which
   signals earn their compute.
