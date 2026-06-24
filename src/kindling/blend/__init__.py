"""Blend utilities.

The v1 signal-blending stack (Bayesian/heuristic blends, likelihoods,
priors, decorrelation) was removed in the production consolidation; the
v2 engine scores via the Rust core. Only ``layer_scoring`` (metric
helpers + ``MetricReport``) remains, imported directly where needed.
"""
