"""Data lifecycle: decay, pruning, drift detection (PRD §3.5)."""

from kindling.lifecycle.decay import (
    CustomDecay,
    DecayProtocol,
    ExponentialDecay,
    LinearDecay,
    NoDecay,
)
from kindling.lifecycle.drift import DriftMetrics, DriftReport, DriftTracker
from kindling.lifecycle.pruning import PreservedAggregate, PruningConfig

__all__ = [
    "CustomDecay",
    "DecayProtocol",
    "DriftMetrics",
    "DriftReport",
    "DriftTracker",
    "ExponentialDecay",
    "LinearDecay",
    "NoDecay",
    "PreservedAggregate",
    "PruningConfig",
]
