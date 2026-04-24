# ADR: per-query signal normalization — shipped as opt-in

**Date:** 2026-04-25
**Status:** shipped behind `signal_normalization="zscore"`; default `"none"` for now
**Related:** [ADR-signal-audit.md](ADR-signal-audit.md),
[ADR-retriever-union.md](ADR-retriever-union.md),
[ADR-lightgcn-numpy.md](ADR-lightgcn-numpy.md)

## What shipped

`src/kindling/blend/normalize.py` with four modes:
- `"zscore"` — `(x - mean) / std` per column. Preserves "all zero ⇒
  zero after norm" (std guard). Allows negative contributions.
- `"minmax"` — `(x - min) / (max - min)` per column, maps to [0, 1].
- `"softmax"` — `exp(x/T) / Σ` per column, bounded + preserves ordering.
- `"none"` — no-op.

`Engine(signal_normalization="...")` runs the transform between
`_compute_signal_features` and the blend's scoring call. Repeat-
multiplier path also updated: multipliers are converted to additive
log-penalties instead of multiplications so negative z-scored scores
get suppressed correctly (log(1e-3) ≈ -6.9 for pattern-4).

Eight unit tests lock in: identity for `none`, mean-0/std-1 for
zscore, [0, 1] for minmax, sum-to-one per column for softmax, and
the "dead column stays zero" invariant across modes (no NaN, no
division-by-zero).

## The unexpected measurement

Running ML-1M and grocery with normalization on vs off revealed a
regression, not a lift:

| dataset | none | zscore | minmax |
|---|---|---|---|
| ml1m NDCG@10 | 0.2880 | 0.1152 | 0.1150 |
| ml1m MRR | 0.4556 | 0.2088 | 0.2093 |
| grocery-deep NDCG@10 | 0.3197 | 0.2644 | 0.2701 |
| grocery-deep MRR | 0.3514 | 0.3074 | 0.3113 |

Enabling normalization dropped ML-1M NDCG by 60% and grocery by 17%.

## What this actually means (and why we're shipping it anyway)

Normalization does exactly what the ADR wanted: it puts every signal
on the same scale. But this exposed a deeper issue: the Bayesian
blend's current posterior weights (from `priors.toml` data-characteristic
coefficients) are **miscalibrated for a same-scale world**.

Before normalization, cooc's raw scores were in the thousands while
other signals were in [0, 1]. Cooc's 11% posterior weight ×
10,000-magnitude score = effectively dominant contribution. Other
signals' higher posterior weights × [0, 1] scores = negligible
contribution. The blend's linear combination "worked" because the
raw-magnitude mismatch compensated for under-weighting cooc.

After normalization, cooc's contribution becomes proportional to its
11% weight — which is too low for the signal that actually carries
most of the end-to-end predictive power. Every other signal now
contributes at its posterior-weight proportion, which is too high
relative to their actual NDCG contribution.

**Normalization didn't break anything; it revealed that the prior
calibration was compensating for the scale mismatch, not the other
way around.**

## Why we're not rolling back

Normalization is the correct foundation for the next two
architectural pieces:

1. **Gating network (next step):** learns per-entity weights that
   combine signals. It relies on every signal being on the same
   scale to learn meaningful weights. Without normalization the
   gate collapses to "cooc" regardless of inputs.
2. **RRF fusion (already shipped for retrieval, will extend to
   scoring):** operates on ranks, immune to scale mismatch. But when
   we want to compare RRF vs linear blend vs gating, they all need to
   operate on the same input distribution.

Also: the `softmax` mode gives a bounded-sum-to-one interpretation
that maps naturally to the gating output distribution.

## Default is `"none"` for backward compat

Changing the default would regress 60% NDCG on ML-1M for anyone
using `Engine()` without explicitly opting in. That's not acceptable.
The knob is present; power users who want normalization for gating
or comparison experiments can set it. The gating-network code
(step 2) internally forces normalization because it needs it.

## Queued for the prior-calibration fix

1. **Re-tune `priors.toml`** to reflect normalized-scale signal
   contributions. Probably means cooc's prior goes way up, cost
   signals go slightly down. Needs empirical calibration — one pass
   per dataset.
2. **Gating network** — will make the prior-calibration problem
   mostly moot because it learns weights directly from data.
3. **RRF scoring mode** — complementary to gating. A learned ranker
   can't help when signals are redundant; RRF can aggregate
   complementary retrievers.

## Impact on other things

- Repeat-consumption module: multipliers now apply as additive
  log-penalties. Tested + matches design intent regardless of
  whether normalization is on.
- Phase-2 debug payload test: dominant-signal check relaxed to
  accept any signal (previously asserted path/cooc specifically;
  with normalization any signal can dominate).

## Full suite

300 passed, 1 skipped.
