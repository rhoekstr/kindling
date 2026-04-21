# ADR: Phase 3 likelihood default selection (placeholder)

**Status:** draft - awaiting Phase 7 data

**Context:** the PRD §6.2 and §12.4 specify that kindling's default
likelihood model is validated empirically rather than chosen theoretically.
Four likelihoods ship in v1: listwise calibration, binary independent,
pairwise Bradley-Terry, and multinomial softmax.

The plan front-loads this critical-path benchmark to Phase 3 (not Phase 7
as in the PRD) so the default choice can bake in early. MovieLens-1M is the
only dataset available in Phase 3; Phase 7 extends to Instacart, Amazon,
and RetailRocket before locking the v1 default.

## Decision rule (plan Phase 3)

Listwise calibration retains its default status if:
1. It dominates on calibration metrics (lowest Brier, lowest ECE)
2. It is within 5% of the best likelihood on predictive accuracy (NDCG@10)

If another likelihood clearly dominates across datasets, the default
changes with documented rationale.

## Phase 3 results (MovieLens-1M)

See `bench/reports/likelihood_suite_movielens.json` for the full table.

Summary (filled by the suite runner): _pending first run_.

## Phase 7 results

_Pending Phase 7 extension to Instacart, Amazon, RetailRocket._

## Final decision

_Will land after Phase 7 completes and results across all four datasets
are available. Until then the default remains ``ListwiseCalibration``._
