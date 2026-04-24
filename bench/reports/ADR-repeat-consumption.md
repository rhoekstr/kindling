# ADR: repeat-consumption module — architecture shipped, calibration open

**Date:** 2026-04-24
**Status:** module shipped opt-in; grocery gains marginal; calibration tuning queued
**Related:** design doc (repeat_consumption_design.md)

## What shipped

Three-commit module implementing the PRD design doc:

1. **Commit 1** — core module in `src/kindling/repeat/`:
   `Pattern` enum (REPEAT/REPLENISH/SATIATION/ONE_SHOT), `RepeatProfile`
   dataclass, `RepeatProfileTable`; KDE-based period detection on
   log-scaled intervals with median fallback for sparse items;
   prototype + KS-distance shape matching; neighbor pooling; the
   four multiplier functional forms.

2. **Commit 2** — engine integration. `RepeatConfig(enabled=True)`
   switches the engine to include owned items in retrieval and apply
   the multiplier between stage-2 scoring and stage-3 rerank.
   Off by default; ML-1M and other no-repeat datasets are unaffected.

3. **Commit 3** — this ADR + a pattern-aware confidence fix. Fixed a
   bug where pure pattern-4 items (e.g., all ML-1M items, which never
   repeat) had zero confidence, causing the multiplier to degrade
   to 1.0 and failing to suppress owned items in the candidate pool.
   Confidence is now derived from intervals for distributional
   patterns and from user count for pattern-4.

## Measurement on grocery-deep @ 100%

500 eval entities, same split we've been using:

| eval metric | repeat off | repeat on | delta |
|-------------|-----------:|----------:|------:|
| **new-only NDCG@10** (test \ train) | 0.3197 | 0.1484 | **−54%** |
| **full NDCG@10** (all test positives) | 0.1601 | 0.1699 | **+6%** |
| full Recall@10 | 0.1266 | 0.1403 | +11% |
| full MRR | 0.3352 | 0.3315 | ~flat |

ML-1M sanity (no repeats in data, so repeat-on should be a no-op):

| eval metric | repeat off | repeat on | delta |
|-------------|-----------:|----------:|------:|
| NDCG@10 | 0.1826 | 0.1826 | **0 (exact no-op)** |
| Recall@10 | 0.0470 | 0.0470 | 0 |
| MRR | 0.3191 | 0.3191 | 0 |

## Interpreting the numbers

**ML-1M is a perfect no-op.** The confidence fix worked: pure pattern-4
items get high confidence from user count, multiplier drops owned-item
scores to epsilon, and they never displace test positives. ✓

**Grocery's "new-only" drop is expected.** When repeat is on, owned
items enter the candidate pool. The engine now recommends owned items
in some top-10 slots (when their multiplier is high, e.g.,
due-for-replenishment). Those slots can't be counted as hits against
"test positives that are new," so the new-only NDCG drops. This is
measuring a different thing — not a regression on the same
population.

**Grocery's "full" only moves +6%** — smaller than the ceiling lift
the 65%-repeat-positive analysis predicted.

## Why the full-metric lift is modest

Three contributing causes, in order of likely impact:

### 1. The multiplier's parametric defaults are hand-tuned, not grocery-calibrated

The sigmoid on replenishment uses `sigmoid(6 * (r - 0.7))` — a ramp
that hits 0.5 at r=0.7 of period and ~0.95 at r=1.3. Real grocery
replenishment may have a different shape (e.g., sharper transition at
r=1, wider tolerance above). The defaults were picked to be
"reasonable across a range of possible repeat patterns" rather than
empirically-tuned against any dataset.

### 2. Period detection via KDE peaks at the mode of observed intervals

For grocery items bought every 7-10 days with variance, KDE-on-log
picks a period near the mode (~7 days). Scaled intervals cluster
around 1.0, which our REPLENISH prototype matches. But the confidence
weighting then treats these as high-confidence REPLENISH, and the
multiplier aggressively suppresses items last-bought <0.5 periods ago
— which includes **many items that the grocery test window actually
contains as repeats.** The test window is only the last 10% of the
data, so test positives are typically last-bought 1-2 weeks before,
which straddles the sigmoid's transition zone where multiplier is 0.3-0.7.

### 3. Cooc-dominated rankings still prefer new items at the top

Even when the multiplier keeps owned items in the top-100, cooc scores
for new items (that co-occur with owned) are often similar in magnitude
to the multiplier-reduced owned items. The top-10 ends up as a mix
of new + old, which is structurally correct but doesn't dramatically
move the full-metric NDCG because the new items were already there.

## What this validates

- **The architecture is correct.** Owned items enter the pool; the
  multiplier decides suppression per-pattern; rerank sees
  time-adjusted scores. ML-1M's perfect no-op proves the zero-repeat
  case doesn't regress.
- **The module degrades gracefully.** Ratings datasets get exact
  equivalence; replenishment datasets get a modest lift.
- **Persistence round-trips.** `RepeatProfileTable` + last-timestamp
  map survive save/load.

## What's queued for tuning

1. **Multiplier functional forms are the biggest lever.** The current
   shapes are defensible but not grocery-calibrated. Two specific knobs
   worth investigating:
   - Sharpen the REPLENISH sigmoid so items at r=1.0 get multiplier
     ~0.95 (currently 0.82), reducing the scoring penalty for
     due-for-repeat items.
   - Consider a period-specific override: learn the sigmoid center
     per-item rather than fixing it at 0.7 of period.

2. **Generalized-gamma parametric fit** (queued from commit-1 plan).
   Discrete prototypes + KS distance force items onto one of three
   fixed shapes. A gengamma fit would let us directly estimate
   per-item distribution parameters. Most useful for items with
   enough observations; falls back to prototype matching below
   threshold.

3. **Per-user multiplier modulation.** Design doc section on user-
   specific repeat variance. A user who replenishes milk every 3
   days has different timing than one who does every 10. Item-level
   period is a population average; per-user offset would help.

4. **Different test-split design.** The current eval measures
   "predict the last 10% of interactions" — but 65% of those are
   repeats that happen within the normal replenishment cycle. An
   eval split that explicitly tests "did we recommend THIS item at
   THIS user's due time?" would better exercise the module's
   designed strength.

## What's shipping in this commit

- `src/kindling/repeat/` (new package): profile, config, period,
  shape, pool, multiplier, fit.
- `Engine(repeat_config=...)` opt-in; default off preserves all
  existing behavior.
- Pattern-aware confidence formula.
- `CoOccurrenceRetriever.retrieve(include_owned=True)` so the cooc
  retriever can surface owned items when the caller opts in. Rust
  kernel updated correspondingly.
- Two test files: `test_repeat_module.py` (24 tests),
  `test_repeat_engine_integration.py` (5 tests).
- Persistence for `_repeat_table` + `_last_interaction_ts`.

Full suite: 277 passed, 1 skipped.

## What this does NOT do

- Does not change the Engine's behavior when `repeat_config` is None
  or `RepeatConfig(enabled=False)` — zero regression risk for
  non-repeat datasets.
- Does not automatically tune multiplier shapes to a dataset. That's
  the calibration work queued above.
- Does not handle explicit "session-aware temporal distance" for
  within-session suppression. Deferred: uses wall-clock time-since
  as-is.
- Does not integrate with substitutes/complements (buying car →
  suppress cars, activate tires/insurance). Design doc flags this
  as out of scope for the module.
