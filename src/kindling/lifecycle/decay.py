"""Time decay functions (PRD §3.5).

A single decay function is configured once at Engine construction and applied
consistently across all structures that care about age - path trees, item
graph, cost graph, basket index. This is a PRD commitment: temporal relevance
stays consistent across signals.

Decay functions must satisfy two invariants (enforced by property tests):
- ``decay(0) == 1.0`` (no decay at age zero).
- Monotonic non-increasing in age.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DecayProtocol(Protocol):
    """The decay interface. Takes an age (in days) and returns a weight in
    ``[0, 1]``."""

    name: str

    def __call__(self, age_days: float | np.ndarray) -> float | np.ndarray: ...


@dataclass(frozen=True)
class ExponentialDecay:
    """``exp(-ln(2) * age / half_life_days)``. Half the weight every
    ``half_life_days``. The PRD default."""

    half_life_days: float = 180.0
    name: str = "exponential"

    def __post_init__(self) -> None:
        if self.half_life_days <= 0:
            raise ValueError(f"half_life_days must be positive, got {self.half_life_days}")

    def __call__(self, age_days: float | np.ndarray) -> float | np.ndarray:
        k = math.log(2) / self.half_life_days
        if isinstance(age_days, np.ndarray):
            result: np.ndarray = np.exp(-k * np.maximum(age_days, 0.0))
            return result
        return math.exp(-k * max(age_days, 0.0))


@dataclass(frozen=True)
class LinearDecay:
    """Linear ramp from 1 at age 0 to 0 at ``zero_at_days``. Weight stays at 0
    for ages beyond ``zero_at_days``."""

    zero_at_days: float = 365.0
    name: str = "linear"

    def __post_init__(self) -> None:
        if self.zero_at_days <= 0:
            raise ValueError(f"zero_at_days must be positive, got {self.zero_at_days}")

    def __call__(self, age_days: float | np.ndarray) -> float | np.ndarray:
        if isinstance(age_days, np.ndarray):
            return np.clip(1.0 - age_days / self.zero_at_days, 0.0, 1.0)
        return max(0.0, min(1.0, 1.0 - age_days / self.zero_at_days))


@dataclass(frozen=True)
class NoDecay:
    """Stationary system - every entry contributes 1.0 regardless of age."""

    name: str = "none"

    def __call__(self, age_days: float | np.ndarray) -> float | np.ndarray:
        if isinstance(age_days, np.ndarray):
            return np.ones_like(age_days, dtype=np.float64)
        return 1.0


@dataclass(frozen=True)
class CustomDecay:
    """User-supplied decay function. The user is responsible for the two
    invariants (``f(0) == 1``, monotonic non-increasing). Not enforced here
    because violating them is a user error that should surface at property-
    test time, not at every decay evaluation."""

    fn: Callable[[float], float]
    name: str = "custom"

    def __call__(self, age_days: float | np.ndarray) -> float | np.ndarray:
        if isinstance(age_days, np.ndarray):
            return np.array([self.fn(float(a)) for a in age_days])
        return self.fn(float(age_days))
