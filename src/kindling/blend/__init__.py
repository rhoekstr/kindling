"""Signal blending - heuristic in Phase 2, Bayesian in Phase 3."""

from kindling.blend.bayesian import BayesianBlend
from kindling.blend.decorrelate import DecorrelationBasis, fit_decorrelation
from kindling.blend.diagnostics import DiagnosticsReport, run_diagnostics
from kindling.blend.heuristic import HeuristicBlend, SignalFeatures
from kindling.blend.likelihoods import (
    BinaryIndependent,
    LikelihoodProtocol,
    ListwiseCalibration,
    MultinomialSoftmax,
    OutcomeBatch,
    PairwiseBradleyTerry,
)
from kindling.blend.priors import DataFeatures, construct_prior

__all__ = [
    "BayesianBlend",
    "BinaryIndependent",
    "DataFeatures",
    "DecorrelationBasis",
    "DiagnosticsReport",
    "HeuristicBlend",
    "LikelihoodProtocol",
    "ListwiseCalibration",
    "MultinomialSoftmax",
    "OutcomeBatch",
    "PairwiseBradleyTerry",
    "SignalFeatures",
    "construct_prior",
    "fit_decorrelation",
    "run_diagnostics",
]
