"""User-facing configuration for the repeat-consumption module."""

from __future__ import annotations

from dataclasses import dataclass, field

from kindling.repeat.profile import RepeatProfile


@dataclass
class RepeatConfig:
    """Control the repeat-consumption module.

    Default is ``enabled=False`` to preserve backward-compatible
    "exclude owned items" behavior on datasets without repeats (ML-1M,
    ratings-style). Turn on for grocery / replenishment / media where
    past interactions are legitimately reusable signal.

    Attributes
    ----------
    enabled:
        Master switch.
    min_observations_individual:
        Items with fewer observed intervals than this fall through
        to neighbor-pooled profile estimation. Signed-off default 5.
    neighbor_pool_k:
        How many cooc-nearest neighbors to pool with when an item's
        individual data is below threshold.
    temperature:
        Softmax temperature for converting KS distances to per-pattern
        probabilities. Lower = more decisive single-pattern assignment.
    pattern_4_rate_threshold:
        Items with repeat_rate below this score high on pattern-4
        regardless of distributional shape.
    explicit_overrides:
        Per-item explicit profiles. Bypass the inference pipeline for
        these items. Highest-precedence.
    category_profiles:
        Per-category explicit profiles. Applied to items whose
        item_to_category entry is present. Lower-precedence than
        ``explicit_overrides``.
    item_to_category:
        Mapping from item_id to category name for ``category_profiles``.
    one_shot_epsilon:
        Pattern-4 multiplier floor. ``1e-3`` so a pattern-4 item can
        still rank if every other candidate has score 0, but otherwise
        is strongly suppressed.
    """

    enabled: bool = False
    min_observations_individual: int = 5
    neighbor_pool_k: int = 10
    temperature: float = 0.3
    pattern_4_rate_threshold: float = 0.05
    explicit_overrides: dict[object, RepeatProfile] = field(default_factory=dict)
    category_profiles: dict[str, RepeatProfile] = field(default_factory=dict)
    item_to_category: dict[object, str] = field(default_factory=dict)
    one_shot_epsilon: float = 1e-3
