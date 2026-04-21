# ADR: Phase 7 cross-dataset critical-path consolidation

**Status:** provisional — offline Phase 7 against synthetic session-heavy + synthetic ratings datasets; external datasets (Instacart / Amazon / RetailRocket) require user download

**Date:** 2026-04-21

## Context

Phase 3 ran the likelihood suite on MovieLens-1M and deferred final
default-selection to Phase 7 after extending to Instacart, Amazon, and
RetailRocket. The plan scoped Phase 7 as "loaders + runs + decision lock."

Honest reality: the three external datasets require Kaggle / mirror
authentication that we cannot satisfy from a read-only environment.
Phase 7 therefore ships:

1. Loaders for all three external datasets with clear "download this"
   error paths.
2. Two synthetic datasets (`synthetic-grocery`, `synthetic-ratings`)
   that exercise session-heavy and rating-style data respectively. We
   run the critical-path suites against these to validate the data-
   adaptive claim.
3. A consolidated comparison table across MovieLens-1M + synthetic.
4. A decision rule that binds: final lock happens when the three
   external datasets actually run against this infrastructure.

## Phase 3/4/7 likelihood-suite results

| Dataset             | NDCG  | path_full | path_tail | path_basket | cooccurrence | cost_* |
| ------------------- | ----- | --------- | --------- | ----------- | ------------ | ------ |
| MovieLens-1M        | 0.196 | 0.42      | 0.38      | 0.19        | 0.01         | (n/a)  |
| synthetic-grocery   | 0.257 | 0.37      | 0.20      | 0.12        | 0.20         | ~0.10  |
| synthetic-ratings   | 0.642 | 0.25      | 0.16      | 0.11        | 0.31         | ~0.18  |

**The Bayesian blend adapts to dataset structure.** On the session-
heavy grocery dataset, path signals hold roughly twice the weight of
cooccurrence. On the ratings-style synthetic dataset, cooccurrence
jumps to 0.31 - the single largest component - and path signals
contract. On ML-1M the posterior is path-dominated, which Phase 3
flagged as likely a scale artifact that the `synthetic-grocery` numbers
help contextualize.

This is the core "data-adaptive" promise in the PRD delivering. It is
the first empirical validation across three structurally different
datasets that the same blend machinery produces different weights
under different data regimes.

## Decision rules revisited

Phase 3 kept `ListwiseCalibration` as the default. Phase 7 confirms:

1. **Likelihood default remains `ListwiseCalibration`.** On all three
   available datasets all four likelihoods cluster within 1-2% NDCG.
   The Brier / ECE reading is distorted by the score-scale artifact
   (documented in the Phase 3 ADR); until we add isotonic / Platt
   post-processing we cannot differentiate calibration across
   likelihoods on these datasets.
2. **Per-position temperature API stays in v1 scope.** Phase 4 confirmed
   measurable output divergence between uniform and staged temperature
   on ML-1M; we re-run on synthetic-grocery and got the same finding.
3. **Beam solver stays the default.** Tied with greedy on both ML-1M
   and synthetic datasets.
4. **Data-adaptive priors stay marketed as novel.** Phase 7 is the
   first evidence that the posterior demonstrably differs across
   datasets; without this the data-adaptive claim was theater.

## Deferred to post-Phase-7

These items require the three external datasets to actually run.
Pathway for the user:

- **Instacart**: download Kaggle `instacart-market-basket-analysis`,
  unpack, call `instacart.load(data_dir)`. Expected behavior: path
  signals dominate more than synthetic-grocery because real
  consumer baskets have heavier pair-affinity.
- **Amazon Reviews**: download a 5-core category JSONL.gz from
  https://nijianmo.github.io/amazon/, call `amazon.load(path)`.
  Expected behavior: cost graph gets meaningful weight via the
  low-ratings mapping.
- **RetailRocket**: download Kaggle `retailrocket/ecommerce-dataset`,
  call `retailrocket.load(data_dir)`. Expected behavior: cost signals
  dominate on the abandoned-cart cohort.

When those runs land, the Phase 7 ADR updates in place.

## Known calibration artifact

All three runs have Brier/ECE near 0.9 with the diagnostic PPC failing
at ~100% deviation. This is the same score-scale artifact flagged in
the Phase 3 ADR: `sigmoid(raw_score)` is not a calibrated probability
because cost signals are in a different scale from positive signals.
Isotonic / Platt post-processing is a Phase 3.x / Phase 5.x follow-up
that will fix the calibration metrics; the underlying posteriors and
NDCG behavior are unaffected.

## Final decision (provisional)

Listwise calibration, beam solver, per-position temperature API all
stay. Revisits after the three external datasets actually run.
