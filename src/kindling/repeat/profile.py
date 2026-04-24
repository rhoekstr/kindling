"""Per-item repeat profile + the four-pattern enum."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Pattern(IntEnum):
    """The four fundamental repeat-consumption patterns.

    - ``REPEAT`` (1): past interaction makes future more likely. Songs,
      favorite movies. Multiplier ~1.0 regardless of time-since-last.
    - ``REPLENISH`` (2): past interaction is neutral long-term but
      predicts *when* the next interaction should occur. Groceries,
      consumables. Multiplier low pre-period, high post-period.
    - ``SATIATION`` (3): past interaction suppresses future for an
      extended refractory period, with recovery. Novels, vacations.
    - ``ONE_SHOT`` (4): past interaction suppresses future ~permanently.
      Durable goods, major purchases. Multiplier near-zero after any
      interaction.
    """

    REPEAT = 1
    REPLENISH = 2
    SATIATION = 3
    ONE_SHOT = 4


@dataclass(frozen=True)
class RepeatProfile:
    """Fitted repeat profile for one item.

    Attributes
    ----------
    pattern:
        The dominant (highest-probability) pattern classification.
    pattern_probs:
        Soft assignment across all four patterns. Sums to 1.0. Used by
        the multiplier to compute a probability-weighted mixture so
        items on pattern boundaries get a blended multiplier.
    period_seconds:
        Characteristic repeat period in seconds. Derived from the
        observed inter-interaction intervals (KDE peak on log-scaled
        intervals, or median fallback for sparse items).
    refractory_seconds:
        Pattern-3 refractory period: how long to strongly suppress.
        Defaults to 3x the characteristic period when not provided.
    confidence:
        [0, 1] based on observation count and fit quality. Low
        confidence dampens the multiplier toward 1.0 (no adjustment).
    n_observations:
        Number of inter-interaction intervals observed for this item
        (or the pooled neighborhood if ``pooled``).
    pooled:
        Whether this profile came from the neighbor-pooling fallback
        rather than the item's own data.
    repeat_rate:
        Fraction of users who interacted with this item more than
        once. Used to score pattern-4 (one-shot) probability
        separately from distributional matching.
    """

    pattern: Pattern
    pattern_probs: dict[Pattern, float]
    period_seconds: float
    refractory_seconds: float
    confidence: float
    n_observations: int
    pooled: bool
    repeat_rate: float


@dataclass
class RepeatProfileTable:
    """Catalog-wide collection of per-item profiles.

    Unknown items fall through to ``default_profile``, which has
    pattern-1 probability 1.0 (safe default: don't suppress items
    we know nothing about).
    """

    profiles: dict[object, RepeatProfile] = field(default_factory=dict)
    default_profile: RepeatProfile = field(
        default_factory=lambda: RepeatProfile(
            pattern=Pattern.REPEAT,
            pattern_probs={
                Pattern.REPEAT: 1.0,
                Pattern.REPLENISH: 0.0,
                Pattern.SATIATION: 0.0,
                Pattern.ONE_SHOT: 0.0,
            },
            period_seconds=86400.0 * 30.0,  # 30d default, unused when pattern=REPEAT
            refractory_seconds=86400.0 * 90.0,
            confidence=0.0,  # zero confidence -> multiplier = 1.0 (no adjustment)
            n_observations=0,
            pooled=False,
            repeat_rate=0.0,
        )
    )

    def get(self, item_id: object) -> RepeatProfile:
        return self.profiles.get(item_id, self.default_profile)

    def __len__(self) -> int:
        return len(self.profiles)
