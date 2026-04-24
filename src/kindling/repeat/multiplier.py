"""Recommend-time repeat multiplier.

Implements the four functional forms agreed in ADR:
- Pattern 1 (REPEAT):     m = 1.0     (no adjustment)
- Pattern 2 (REPLENISH):  m = sigmoid(6 * (r - 0.7))
- Pattern 3 (SATIATION):  m = 1 - exp(-(r / refractory_r)^2)
- Pattern 4 (ONE_SHOT):   m = epsilon (default 1e-3)

The final multiplier is a probability-weighted mixture across the
four patterns (so items on pattern boundaries get a blended output)
and then dampened by confidence toward 1.0 for low-confidence profiles.

r = time_since_last_interaction / period. When the entity has never
interacted with the candidate, returns 1.0 unconditionally (no past,
no adjustment).
"""

from __future__ import annotations

import math

from kindling.repeat.profile import Pattern, RepeatProfile


def _pattern_multiplier(pattern: Pattern, r: float, profile: RepeatProfile, one_shot_epsilon: float) -> float:
    if pattern is Pattern.REPEAT:
        return 1.0
    if pattern is Pattern.REPLENISH:
        # sigmoid(6 * (r - 0.7)); near 0 below ~0.5 period, ~1 above 1 period.
        x = 6.0 * (r - 0.7)
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        return math.exp(x) / (1.0 + math.exp(x))
    if pattern is Pattern.SATIATION:
        refractory_r = profile.refractory_seconds / max(profile.period_seconds, 1e-9)
        ratio = r / max(refractory_r, 1e-9)
        # Saturates to 1.0 as r grows past several refractory units.
        return 1.0 - math.exp(-(ratio * ratio))
    if pattern is Pattern.ONE_SHOT:
        return one_shot_epsilon
    return 1.0


def multiplier(
    profile: RepeatProfile,
    time_since_last_seconds: float | None,
    one_shot_epsilon: float = 1e-3,
) -> float:
    """Compute the recommend-time multiplier for a candidate.

    ``time_since_last_seconds``:
        How long since the entity last interacted with this item.
        ``None`` means never interacted -> returns 1.0 unconditionally.
    """
    if time_since_last_seconds is None:
        return 1.0
    period = max(profile.period_seconds, 1e-9)
    r = max(float(time_since_last_seconds), 0.0) / period

    # Probability-weighted mixture across the four patterns.
    weighted = 0.0
    for pattern, prob in profile.pattern_probs.items():
        if prob <= 0.0:
            continue
        m = _pattern_multiplier(pattern, r, profile, one_shot_epsilon=one_shot_epsilon)
        weighted += prob * m

    # Confidence dampening toward 1.0.
    conf = max(0.0, min(1.0, profile.confidence))
    return conf * weighted + (1.0 - conf) * 1.0
