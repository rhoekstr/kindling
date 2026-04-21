"""Data lifecycle: decay, pruning, drift detection (PRD §3.5)."""

from kindling.lifecycle.decay import (
    CustomDecay,
    DecayProtocol,
    ExponentialDecay,
    LinearDecay,
    NoDecay,
)

__all__ = [
    "CustomDecay",
    "DecayProtocol",
    "ExponentialDecay",
    "LinearDecay",
    "NoDecay",
]
