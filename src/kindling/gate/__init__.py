"""Gating network for per-entity signal combination.

A small MLP takes per-entity context features (n_interactions,
session_density, rating stats, persona presence, ...) and outputs a
softmax distribution over the K signals. The gate's output replaces
the Bayesian blend's posterior mean for scoring: candidate_score =
gate_weights . normalized_signal_vector.

Trained with BPR SGD + pure-numpy gradients (no torch). Lives behind
``Engine(gating_config=GatingConfig(enabled=True))``.
"""

from kindling.gate.config import GatingConfig
from kindling.gate.features import compute_context_features
from kindling.gate.fit import fit_gating_network
from kindling.gate.model import GatingNetwork

__all__ = [
    "GatingConfig",
    "GatingNetwork",
    "compute_context_features",
    "fit_gating_network",
]
