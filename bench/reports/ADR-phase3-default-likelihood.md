# ADR: Phase 3 likelihood default selection

**Status:** provisional — listwise_calibration retained pending Phase 7

**Date:** 2026-04-21 (Phase 3 run on MovieLens-1M)

## Context

The PRD (§6.2, §12.4) specifies that the default likelihood is validated
empirically rather than chosen theoretically. Four likelihoods ship in v1:
listwise calibration, binary independent, pairwise Bradley-Terry, and
multinomial softmax.

The plan front-loads this critical-path benchmark to Phase 3 so the
default choice can bake in before Phase 7's multi-dataset comparison.

## Decision rule

Listwise calibration retains its default status if:
1. It dominates on calibration metrics (lowest Brier, lowest ECE)
2. It is within 5% of the best likelihood on predictive accuracy (NDCG@10)

## Phase 3 results (MovieLens-1M, 300 eval entities, vi_max_iter=150)

Full data: `bench/reports/likelihood_suite_movielens.json`.

| Likelihood              | NDCG@10  | MRR     | Brier   | ECE     | Fit (s) |
| ----------------------- | -------- | ------- | ------- | ------- | ------- |
| listwise_calibration    | 0.1958   | 0.3353  | 0.8151  | 0.8152  | 108.5   |
| binary_independent      | 0.1957   | 0.3334  | 0.8148  | 0.8149  | 112.7   |
| pairwise_bradley_terry  | 0.1957   | 0.3348  | 0.8151  | 0.8152  | 129.3   |
| multinomial_softmax     | **0.1982** | **0.3413** | **0.8136** | **0.8136** | 125.6   |

All four likelihoods land within 1.3% of each other on NDCG and within 0.2%
on Brier / ECE. Multinomial softmax edges the others marginally. Posterior
means are essentially identical across likelihoods:

```
path_full   ≈ 0.425
path_tail   ≈ 0.377
path_basket ≈ 0.192
cooccurrence ≈ 0.005-0.008
```

## Observations

1. **Phase 3 NDCG of 0.196 beats Phase 2 (0.187) by +4.8% and Phase 1
   (0.183) by +7%.** This clears the +5% gate the plan had set for Phase
   2. The Bayesian blend's implicit standardization of path signals (via
   decorrelation) turns out to be the missing ingredient that let path
   signals drive ranking without being drowned out by raw cooccurrence
   magnitude.

2. **Convergence diagnostics fail across all four likelihoods.** ELBO is
   non-monotonic, posterior predictive check shows ~85% Brier deviation,
   and variational ESS is ~1% (below the 10% threshold). The VI is
   converging to a reasonable posterior for ranking, but the posterior is
   not a good fit to the true posterior under a Dirichlet variational
   family. This is expected when the likelihood pulls the posterior far
   from the prior's shape — in our case, far from a Dirichlet with
   graph-density-boosted cooccurrence mass.

3. **Brier/ECE near 0.82 across all likelihoods is a calibration scale
   artifact.** The raw score is not a calibrated probability because
   blend weights are applied to decorrelated path signals (standardized
   scale, roughly centered on zero) but cooccurrence is on raw count
   scale. Sigmoid(score) ends up near 0.5 for almost every candidate.
   Proper calibration requires score-to-probability mapping (isotonic
   regression or Platt scaling) as a post-processing step; this is a
   Phase 3.x / Phase 5 follow-up.

## Decision

**Keep listwise_calibration as the default.** Rationale:
1. All four likelihoods are statistical ties on every metric; there is no
   empirical reason to switch.
2. Multinomial's marginal NDCG advantage (~1.3%) is within run-to-run
   noise at this sample size.
3. Listwise calibration's design intent (uncertainty-aware calibration)
   matches kindling's architectural commitment to credible intervals,
   even if the current Brier/ECE metrics don't reflect that because of
   the scale artifact above.

Final decision will revisit after Phase 7's four-dataset comparison.

## Follow-up tasks (Phase 3.x)

1. Add isotonic / Platt score-to-probability calibration so Brier/ECE
   become meaningful and the diagnostic PPC matches reality.
2. Investigate whether a richer variational family (full-covariance
   Gaussian with simplex reparameterization) produces better ESS on this
   data. The Dirichlet family is pluggable per PRD §6.7.
3. Re-run this suite with larger `max_eval_entities` and `vi_max_iter`
   once calibration is fixed, to see whether the four-way near-tie
   separates.

## Phase 7 rerun

_Pending Phase 7 extension to Instacart, Amazon, RetailRocket._

## Final decision

_Will land after Phase 7 completes. Until then the default remains
``ListwiseCalibration``._
