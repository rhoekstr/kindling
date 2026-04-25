"""Dataset profiling + layer planning.

Implements the user-asked 8-step pipeline for auto-configuring the
engine based on dataset shape:

1. Dataset assessment (size, features, time/segmentation, density)
2. Determine time-use (real timestamps vs rating-burst vs none)
3. Determine layers used
4. Assess item repeat schema
5. Cooc weighting choice (kernel-decay or count)
6. Applicable layers
7. Results building
8. Repeat filtering management

The profile is computed once at engine fit time. The plan derived
from it determines which subsystems the engine builds and which
layers participate in adaptive boosting.
"""

from kindling.profile.profile import DatasetProfile, profile_dataset
from kindling.profile.plan import LayerPlan, plan_layers

__all__ = [
    "DatasetProfile",
    "LayerPlan",
    "plan_layers",
    "profile_dataset",
]
