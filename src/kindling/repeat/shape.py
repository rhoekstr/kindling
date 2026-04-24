"""Shape matching: classify scaled intervals against the four prototypes.

Patterns 1, 2, 3 are distributional; pattern 4 is rate-based. The final
output is a probability distribution over all four patterns, summing
to 1.0.

Prototype CDFs are defined on scaled intervals ``r = interval / period``.
Scales are chosen so each prototype is well-separated in both mean and
coefficient of variation, matching what the period detector naturally
produces (KDE-log peak normalizes to ~mean=1):

- Pattern 1 (REPEAT): exponential, mean=1, CV=1. Heavy right tail -
  most intervals small, occasional huge ones. expon(scale=1.0).
- Pattern 2 (REPLENISH): gamma, mean=1, CV=0.5. Tight peak near the
  characteristic period. gamma(a=4, scale=0.25).
- Pattern 3 (SATIATION): log-normal, mass far from zero (median ~4.5
  periods). lognorm(s=0.5, scale=exp(1.5)).

The CV distinction is what separates REPEAT from REPLENISH - both have
mean ~1 after normalization, but REPEAT has CV=1 (high variance, long
tail) while REPLENISH has CV=0.5 (tight).

Classification proceeds in two steps:
1. Compute KS distance to each of prototypes 1, 2, 3. Soft-max with
   temperature to get pattern_1/2/3 probabilities summing to (1 - p4).
2. Separately score pattern-4 from the item's repeat_rate.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from kindling.repeat.profile import Pattern

_PATTERN_1 = stats.expon(scale=1.0)
_PATTERN_2 = stats.gamma(a=4, scale=0.25)
_PATTERN_3 = stats.lognorm(s=0.5, scale=float(np.exp(1.5)))

_PROTOTYPE_SAMPLE_N = 500
_PROTOTYPE_RNG = np.random.default_rng(42)

# Pre-compute prototype samples once. ks_2samp needs a sample; using a
# fixed sample instead of re-drawing each call is faster and
# reproducible.
_PROTOTYPE_SAMPLES: dict[Pattern, np.ndarray] = {
    Pattern.REPEAT: _PATTERN_1.rvs(size=_PROTOTYPE_SAMPLE_N, random_state=_PROTOTYPE_RNG),
    Pattern.REPLENISH: _PATTERN_2.rvs(size=_PROTOTYPE_SAMPLE_N, random_state=_PROTOTYPE_RNG),
    Pattern.SATIATION: _PATTERN_3.rvs(size=_PROTOTYPE_SAMPLE_N, random_state=_PROTOTYPE_RNG),
}


def classify_shape(
    scaled_intervals: np.ndarray,
    repeat_rate: float,
    temperature: float = 0.3,
    pattern_4_rate_threshold: float = 0.05,
) -> dict[Pattern, float]:
    """Return a probability distribution over the four patterns.

    ``scaled_intervals`` are already divided by the detected period, so
    they live on the unit-period scale. Empty arrays or NaN-laden input
    produce a uniform-ish distribution.
    """
    scaled_intervals = np.asarray(scaled_intervals, dtype=np.float64)
    scaled_intervals = scaled_intervals[np.isfinite(scaled_intervals) & (scaled_intervals > 0.0)]

    # Pattern-4 score: low repeat_rate -> high pattern-4 probability.
    # Sigmoid centered on the configured threshold.
    p4_raw = 1.0 / (1.0 + np.exp(50.0 * (repeat_rate - pattern_4_rate_threshold)))

    # If we have no distributional data (new item, or all single-obs),
    # rely entirely on pattern-4 heuristic + uniform among 1/2/3.
    if scaled_intervals.size < 3:
        remainder = 1.0 - p4_raw
        return {
            Pattern.REPEAT: remainder / 3.0,
            Pattern.REPLENISH: remainder / 3.0,
            Pattern.SATIATION: remainder / 3.0,
            Pattern.ONE_SHOT: p4_raw,
        }

    # KS distance to each of prototypes 1, 2, 3.
    distances: dict[Pattern, float] = {}
    for pat, proto in _PROTOTYPE_SAMPLES.items():
        stat = stats.ks_2samp(scaled_intervals, proto).statistic
        distances[pat] = float(stat)

    # Softmax over negative distances.
    neg = np.array([-distances[p] / temperature for p in (Pattern.REPEAT, Pattern.REPLENISH, Pattern.SATIATION)])
    neg -= neg.max()  # numerical stability
    w = np.exp(neg)
    w /= w.sum()
    # Scale 1/2/3 probs by (1 - p4) so they + p4 sum to 1.
    remainder = 1.0 - p4_raw
    return {
        Pattern.REPEAT: float(w[0] * remainder),
        Pattern.REPLENISH: float(w[1] * remainder),
        Pattern.SATIATION: float(w[2] * remainder),
        Pattern.ONE_SHOT: float(p4_raw),
    }


def dominant_pattern(pattern_probs: dict[Pattern, float]) -> Pattern:
    """Return the highest-probability pattern (ties broken by enum order)."""
    return max(pattern_probs, key=lambda p: (pattern_probs[p], -int(p)))
