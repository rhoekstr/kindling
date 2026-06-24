"""kindling — a hybrid recommender system that grows with your data.

The public ``Engine`` is the validated v2 stack (EASE / wilson-cooc base +
auto-gated z-normalized channels, Rust core). See ``docs/REFERENCE.md`` for
the architecture and ``docs/EXPERIMENTS.md`` for the experiment record.
"""

from kindling.engine_v2 import EngineV2 as Engine
from kindling.engine_v2 import RecommendationV2 as Recommendation
from kindling.explain import Explanation

__all__ = ["Engine", "Explanation", "Recommendation"]
__version__ = "0.0.1.dev0"
