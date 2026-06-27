# Repeat personal-frequency layer

Fixes the Stage-5 finding (`archive/REPEAT-AWARE-FINDINGS.md`): the repeat module
re-surfaced reorders but ranked them by co-occurrence affinity × timing, ignoring
**reorder frequency** — so the trivial "buy it again" (personal_freq) baseline beat
it on every high-repeat dataset.

## Fix

When the repeat gate fires, each reorder candidate is now lifted by its purchase
frequency (the count was already in the fit profile, just discarded):

```
score[j] = (base_affinity[j] + repeat_freq_alpha · log1p(count_j)) · timing_mult[j]
```

`repeat_freq_alpha` ("auto" → 50) controls how much frequency dominates; the
REPLENISH timing multiplier still modulates (suppress just-bought, surface due).
`alpha=0` recovers the old affinity-only behavior. Only active when the repeat
gate is on, so non-repeat datasets are unchanged (ml1m 0.2928 / beauty 0.0336 /
steam 0.0659; 132 tests pass).

## Results (repeat-aware NDCG@10, `bench/repeat_freq_sweep.py`)

| dataset (base) | personal_freq | α=0 (old) | α=50 | α=100 |
|---|---:|---:|---:|---:|
| dunnhumby (EASE) | 0.468 | 0.259 | **0.475** | 0.478 |
| tafeng (EASE) | 0.110 | 0.119 | **0.126** | 0.125 |
| instacart (cooc) | 0.387 | 0.036 | 0.357 | 0.369 |
| gowalla (cooc) | 0.235 | 0.141 | 0.168 | 0.186 |

- **EASE-base repeat datasets: kindling now beats personal_freq** — it has
  frequency *plus* timing suppression *plus* new-item discovery, which the naive
  baseline lacks.
- **cooc-base datasets: 10× better than before** (instacart 0.036 → 0.357) but
  not yet at personal_freq. Residual gap = the cooc base affinity perturbs the
  pure-count ordering more than the EASE base does, and on Gowalla the timing
  multiplier likely over-suppresses recent revisits.

## Known residual / next steps (not done)

- **cooc-base headroom:** dropping the base-affinity term for reorder candidates
  (pure frequency × timing) on the cooc path, or a higher per-base-type alpha,
  would close the instacart/gowalla gap.
- **revisit timing:** REPLENISH suppression may be wrong for check-in/revisit data
  (Gowalla) where recency *predicts* the next visit; a per-dataset pattern (vs the
  fixed REPLENISH) is the unbuilt 4-pattern classifier.

Net: the repeat module now uses the dominant signal and is competitive-to-winning
on repeat data, where before it lost to a one-line baseline.
