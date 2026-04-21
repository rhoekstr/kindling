"""Stage 3 re-ranking - constraints, diversity, temperature, calibration, lift."""

from kindling.rerank.calibration import (
    CategoryIndex,
    apply_calibration,
    build_category_index,
)
from kindling.rerank.constraints import ConstraintPredicate, apply_constraints
from kindling.rerank.dpp import (
    CooccurrenceCosineKernel,
    DPPGreedy,
    SimilarityKernel,
)
from kindling.rerank.lift import (
    PopulationBaselines,
    apply_lift,
    compute_population_baselines,
)
from kindling.rerank.temperature import TemperatureObjective, resolve_temperature
from kindling.rerank.temperature import solve as solve_temperature

__all__ = [
    "CategoryIndex",
    "ConstraintPredicate",
    "CooccurrenceCosineKernel",
    "DPPGreedy",
    "PopulationBaselines",
    "SimilarityKernel",
    "TemperatureObjective",
    "apply_calibration",
    "apply_constraints",
    "apply_lift",
    "build_category_index",
    "compute_population_baselines",
    "resolve_temperature",
    "solve_temperature",
]
