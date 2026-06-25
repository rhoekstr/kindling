"""kindling — a hybrid recommender system that grows with your data.

The public ``Engine`` is the validated v2 stack (EASE / wilson-cooc base +
auto-gated z-normalized channels, Rust core). See ``docs/REFERENCE.md`` for
the architecture and ``docs/EXPERIMENTS.md`` for the experiment record.
"""

from importlib.metadata import PackageNotFoundError, version

from kindling.activation import ActivationPlan, LayerActivation
from kindling.engine import Engine, Recommendation
from kindling.explain import Explanation

try:
    __version__ = version("kindling")
except PackageNotFoundError:  # pragma: no cover - source tree / not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "ActivationPlan",
    "Engine",
    "Explanation",
    "LayerActivation",
    "Recommendation",
]
