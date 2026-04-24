"""Repeat-consumption module.

Four patterns x item-level profiling x scale-invariant shape matching.
See bench/reports/design/repeat_consumption_design.md for the design
document this module implements.

Public surface:
- ``Pattern``: the four-way enum (REPEAT, REPLENISH, SATIATION, ONE_SHOT)
- ``RepeatProfile``: per-item fitted profile
- ``RepeatProfileTable``: catalog-wide collection of profiles
- ``RepeatConfig``: user-facing configuration
- ``fit_repeat_profiles``: builds a table from timestamped interactions
- ``multiplier``: compute the recommend-time adjustment for a candidate
"""

from kindling.repeat.config import RepeatConfig
from kindling.repeat.fit import fit_repeat_profiles
from kindling.repeat.multiplier import multiplier
from kindling.repeat.profile import Pattern, RepeatProfile, RepeatProfileTable

__all__ = [
    "Pattern",
    "RepeatConfig",
    "RepeatProfile",
    "RepeatProfileTable",
    "fit_repeat_profiles",
    "multiplier",
]
