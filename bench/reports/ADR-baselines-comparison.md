# ADR: kindling vs. industry-standard baselines (MovieLens-1M)

**Date:** 2026-04-21
**Status:** informational
**Context:** First apples-to-apples comparison of kindling against three
  standard baselines on the same split, same metrics, same harness.

## What ran

`python -m kindling.benchmarks.comparison --max-eval-entities 2000` on
MovieLens-1M, chronological 90/10 split, k=10, 1,157 evaluated entities
(the intersection of entities with train-and-test interactions after
subsampling). Catalog size 3,489 items. Report:
[baselines_comparison.json](baselines_comparison.json).

Baselines:

- **Popularity** — global item-interaction counts, entity history masked.
- **Item-Item kNN** — cosine similarity over binarized user-item matrix,
  top-200 neighbors per item, score = sum over owned items of similarity.
- **implicit ALS** — weighted matrix factorization (Hu-Koren-Volinsky)
  via the `implicit` package, 64 factors, 15 iterations.

## Results

| Model           | NDCG@10    | Recall@10 | MRR        | HitRate    | Coverage   | Fit (s) | p95 latency |
| --------------- | ---------- | --------- | ---------- | ---------- | ---------- | ------- | ----------- |
| kindling        | 0.182      | 0.049     | 0.318      | 0.582      | 0.062      | 105.6   | 246.6 ms    |
| popularity      | 0.178      | 0.046     | 0.312      | 0.564      | 0.028      | 0.11    | 0.006 ms    |
| **item_item_knn** | **0.197**  | **0.053** | **0.340**  | **0.595**  | 0.061      | 0.41    | 0.19 ms     |
| implicit_als    | 0.159      | 0.051     | 0.295      | 0.580      | **0.182**  | 1.44    | 0.054 ms    |

## What this says

1. **Item-item kNN wins on accuracy.** It is +8% NDCG, +8% Recall, +7%
   MRR over kindling. This is consistent with the Phase 2 finding that
   MovieLens-1M is a ratings-not-sessions dataset where path signals
   under-perform. The cross-dataset Phase 7 results show kindling's
   path family beating popularity by a wider margin on
   synthetic-grocery and RetailRocket, where sessions exist.

2. **Implicit ALS wins coverage.** 18% catalog coverage vs. kindling's
   6% — latent factors surface more tail items. Kindling's
   `temperature=0` default collapses toward popular items; raising
   temperature should narrow the gap but was not swept in this
   comparison. Worth revisiting once the coverage U-shape sweep lands
   in `bench/reports/`.

3. **Kindling vs. popularity is only marginal on ML-1M.** +2.5% NDCG
   and +3.2% hit rate. The real separator is coverage (6.2% vs. 2.8%),
   which matches kindling's design goal but doesn't show up in the
   accuracy numbers.

4. **Latency is the honest weakness.** Kindling's p95 recommend is
   247ms; kNN and ALS are under 0.2ms. Kindling is running the full
   stack per call — retrieve + rank + DPP + temperature + calibration
   + explanation — not just a dot product. The Phase 8 profile measured
   retrieve-only at 5.5ms; the rest of the stack adds ~240ms. That is
   ≈5× the PRD's 50ms target for small catalogs. Worth a Phase 11
   investigation before release.

5. **Fit is slower too** — 105s vs. under 2s for the baselines. This is
   the Bayesian VI pass. Acceptable for offline retrain cadence;
   unacceptable if anyone tries to treat it as online.

## What this doesn't say

- **Not a head-to-head on sessions.** ML-1M is the wrong shape of data
  for kindling's headline signals. A session-rich benchmark
  (RetailRocket, Instacart) would be fairer and should be the next
  comparison.
- **No hyperparameter tuning.** Baselines use library defaults
  (ALS at 64 factors, kNN at k=200). Kindling runs defaults too. A
  fair comparison at release time should tune each model on a
  validation fold.
- **No statistical significance test.** Differences of 2-5% on 1,157
  entities are within the noise envelope that a bootstrap confidence
  interval would reveal. The qualitative ranking (kNN > kindling >
  popularity > ALS on accuracy; ALS >> kindling on coverage) is
  plausibly stable but not formally shown.
- **Single-dataset snapshot.** Re-running on Phase 7's other three
  datasets would change this picture substantially.

## Implications for the project

1. **Don't market kindling as an accuracy leader on rating data.** The
   honest positioning is: competitive at accuracy, stronger at
   coverage + diversity + calibrated uncertainty + explanations. The
   README and user-guide already avoid overclaiming; this data lets us
   stay there with receipts.

2. **Latency is the real pre-release item.** 247ms p95 is the most
   concerning number. Profile to determine whether DPP, temperature
   beam search, or the explanation step dominates. This belongs in
   Phase 11's release-engineering scope.

3. **Run the session-rich comparison next.** Extending this harness to
   RetailRocket is ~20 lines of loader wiring and would make the
   comparison fair to kindling's actual design.

4. **Keep the baselines in CI.** A regression gate on "kindling NDCG ≥
   item-item kNN NDCG − 5%" on ML-1M would catch the case where we
   accidentally break the signal stack and drop below parity with a
   much simpler model.
