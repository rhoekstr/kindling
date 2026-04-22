"""Construct Dirichlet prior concentration parameters from data features
(PRD §6.7).

The coefficients live in ``priors.toml`` next to this file. The mapping
itself (which feature adjusts which signal) is theoretically motivated and
stable; the exact magnitudes are empirical and expected to tune as Phase 7
benchmarks run on the four reference datasets.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib import resources
from typing import cast

import numpy as np

# Imported lazily for the clip range; avoids a circular import with bayesian.py.
MIN_ALPHA = 0.5
MAX_ALPHA = 100.0


@dataclass(frozen=True)
class DataFeatures:
    """Graph-theoretic summaries of the training data that drive the prior.

    All fields are dimensionless scalars in sensible ranges; computed from
    the input interactions and the item graph at ``Engine.fit`` time.
    """

    graph_density: float  # edges / max_edges, in [0, 1]
    clustering_coefficient: float  # mean local clustering, in [0, 1]
    session_density: float  # avg items per session
    catalog_to_entity_ratio: float  # n_items / n_entities
    n_interactions: int
    # True iff the input DataFrame carried an explicit ``session_id``
    # column. Without it, sessions are GMM-inferred from inter-event
    # timestamps, which on ratings-style data reconstructs rapid-fire
    # rating bursts rather than semantic sessions. When False, the prior
    # builder shrinks path_* priors via the `session_stiffness` rule so
    # the blend does not over-trust path signals on ratings input.
    has_explicit_sessions: bool = False


def load_prior_coefficients() -> dict[str, dict[str, object]]:
    """Load the coefficient table from ``priors.toml``. Cached by importer."""
    with resources.files("kindling.blend").joinpath("priors.toml").open("rb") as f:
        return tomllib.load(f)


def construct_prior(
    signal_names: tuple[str, ...],
    features: DataFeatures,
    coefficients: dict[str, dict[str, object]] | None = None,
) -> np.ndarray:
    """Build the Dirichlet prior alpha vector for the given signals.

    Unknown signal names get ``alpha = baseline``. Known names receive
    multiplicative adjustments per the rules in ``priors.toml``.
    """
    coefs = coefficients if coefficients is not None else load_prior_coefficients()
    baseline = float(coefs["baseline"]["alpha"])  # type: ignore[arg-type]
    alpha = np.full(len(signal_names), baseline, dtype=np.float64)
    name_to_idx = {name: i for i, name in enumerate(signal_names)}

    _apply_single(
        alpha,
        name_to_idx,
        coefs.get("graph_density"),
        features.graph_density,
    )
    _apply_single(
        alpha,
        name_to_idx,
        coefs.get("graph_density_cosine"),
        features.graph_density,
    )
    _apply_multi(
        alpha,
        name_to_idx,
        coefs.get("clustering_coefficient"),
        features.clustering_coefficient,
    )
    # Session-density boosts only apply when the caller supplied an explicit
    # session_id column. On inferred (ratings-style) sessions the density
    # measure is misleading - see `session_stiffness` below - so the boosts
    # are skipped entirely and path priors stay near baseline.
    if features.has_explicit_sessions:
        _apply_single(
            alpha,
            name_to_idx,
            coefs.get("session_density_full"),
            features.session_density,
        )
        _apply_single(
            alpha,
            name_to_idx,
            coefs.get("session_density_tail"),
            features.session_density,
        )
        _apply_single(
            alpha,
            name_to_idx,
            coefs.get("session_density_basket"),
            features.session_density,
        )

    ratio_cfg = coefs.get("catalog_to_entity_ratio")
    if ratio_cfg is not None:
        threshold = float(ratio_cfg["threshold"])  # type: ignore[arg-type]
        shrink_factor = float(ratio_cfg["shrink_factor"])  # type: ignore[arg-type]
        if features.catalog_to_entity_ratio > threshold:
            alpha = alpha / shrink_factor

    # Session-structure stiffness: path signals require *semantic* session
    # structure to carry information. When the caller did not provide a
    # session_id column, sessions are GMM-inferred from timestamps - which
    # on ratings-style data reconstructs rapid-fire rating bursts, not
    # meaningful baskets. In that case, shrink path_* priors so the blend
    # relies on cooccurrence + cost signals instead. Resolves ADR
    # growth-curves finding #3 (popularity beats kindling on ratings data
    # below 100%) and the prior-sensitivity finding that path_full
    # dominates even on ratings input.
    stiffness_cfg = coefs.get("session_stiffness")
    if stiffness_cfg is not None and not features.has_explicit_sessions:
        shrink = float(stiffness_cfg.get("shrink_factor", 0.2))  # type: ignore[arg-type]
        targets_s = cast("list[str]", stiffness_cfg.get("targets", []))
        for target in targets_s:
            idx = name_to_idx.get(str(target))
            if idx is not None:
                alpha[idx] *= shrink

    return np.clip(alpha, MIN_ALPHA, MAX_ALPHA)


def _apply_single(
    alpha: np.ndarray,
    name_to_idx: dict[str, int],
    cfg: dict[str, object] | None,
    value: float,
) -> None:
    if cfg is None:
        return
    target = str(cfg.get("target", ""))
    coefficient = float(cfg.get("coefficient", 0.0))  # type: ignore[arg-type]
    idx = name_to_idx.get(target)
    if idx is not None:
        alpha[idx] *= 1.0 + coefficient * value


def _apply_multi(
    alpha: np.ndarray,
    name_to_idx: dict[str, int],
    cfg: dict[str, object] | None,
    value: float,
) -> None:
    if cfg is None:
        return
    targets_raw = cfg.get("targets", [])
    targets = cast("list[str]", targets_raw)
    coefficient = float(cast("float", cfg.get("coefficient", 0.0)))
    for target in targets:
        idx = name_to_idx.get(str(target))
        if idx is not None:
            alpha[idx] *= 1.0 + coefficient * value
