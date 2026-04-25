"""Cooccurrence with adaptive boosting (layered scoring architecture).

A third scoring architecture alongside the Bayesian blend and gating
network. Frames recommendation as: cooc rules; refinement signals
nudge, only when they fire confidently. Designed to address a
structural weakness of linear blending: the Bayesian posterior weight
on a sparse signal (path_basket, session_cooccurrence, temporal_cooc
on rating-burst data) is a single scalar that can't distinguish "this
signal has confident information on this candidate" from "this signal
is contributing weak noise to all candidates." The adaptive-boosting
architecture makes refinement contributions **conditional on within-
pool evidence** (z-threshold) AND **conditional on dataset-level
appropriateness** (meaningfulness gate at fit time).

Mechanism:

    score(c) = primary(c) + sum_layers boost · I[ z_layer(c) > tau ]

where:

- ``primary(c)`` is the broad ranker (default cooccurrence).
- Each refinement layer contributes an additive boost iff the
  candidate's **one-tailed z-score** within the layer's non-zero
  subset exceeds ``tau`` (default 2.0).
- ``boost = boost_multiplier × median(adjacent gaps in primary's
  top-20)``. Default boost_multiplier=3.0 means a layer firing on a
  candidate moves it ~3 positions in the primary ranking. Physical
  units, not abstract weights.

Why one-tailed: refinement signals have asymmetric semantics. A
high path_basket score on item ``c`` means "yes, this confidently
appears in baskets near recent activity" - that's a real signal of
relevance. A low path_basket score means the basket index has no
data on ``c``, not that ``c`` is bad. So the boost is "+B if this
fires, +0 otherwise" - never a penalty.

Why z over the non-zero subset: sparse refinement signals (path_basket
hits maybe 20% of candidates) have a long zero tail. Z-scoring against
all candidates makes σ tiny so almost every non-zero item passes the
threshold. Z-scoring against just the items that have any signal
gives a fair "stand out among items that registered" judgment.

Why cumulative: each refinement layer adds an independent boost when
it fires. Layers don't compete; they compose. An item with z>tau on
both session_cooccurrence and temporal_cooccurrence gets 2·boost
on top of its primary score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LayeredConfig:
    """Scalars for the layered scorer.

    Attributes
    ----------
    z_threshold:
        One-tailed z above which a refinement layer fires. Default 2.0
        - "stand out by 2 sigma above the non-zero population mean".
    boost_multiplier:
        How many primary-rank-position equivalents one firing layer is
        worth. Default 3.0.
    top_k_for_boost_calibration:
        Window over which the median adjacent-gap is measured. Default
        20 - a "top-20" rank window. Smaller (e.g. 10) makes the boost
        more aggressive on tightly-clustered top scores; larger (e.g.
        50) calibrates against a broader baseline.
    min_nonzero_for_zscore:
        Minimum non-zero candidates required for the layer to fire at
        all. Below this, the z computation is unstable and the layer
        contributes nothing. Default 3.
    """

    z_threshold: float = 2.5
    boost_multiplier: float = 3.0
    top_k_for_boost_calibration: int = 20
    min_nonzero_for_zscore: int = 3
    # Primary signal name. The base score every candidate gets, on top
    # of which boost layers stack. Default "cooccurrence" (global cooc
    # via U.T @ U). Alternative: "persona_cooccurrence" (soft-weighted
    # per-persona cooc) for cold-start regimes where global cooc has
    # thin signal but persona-cooc pools cluster-level evidence.
    primary_signal: str = "cooccurrence"


def layered_score(
    primary_scores: np.ndarray,
    refinement_scores: list[np.ndarray],
    config: LayeredConfig | None = None,
) -> np.ndarray:
    """Compose primary + cumulative one-tailed z-gated boosts.

    Parameters
    ----------
    primary_scores:
        ``(n_candidates,)`` of the primary ranker (e.g. cooc).
    refinement_scores:
        List of ``(n_candidates,)`` arrays, one per refinement layer.
        Each layer's contribution is independent and additive.
    config:
        Defaults: z=2, boost_multiplier=3, top-20 calibration window.

    Returns
    -------
    ``(n_candidates,)`` composite scores. Same shape as primary,
    so ranking-by-score continues to work for the caller.
    """
    cfg = config or LayeredConfig()
    out = primary_scores.astype(np.float64, copy=True)

    # Boost magnitude: median adjacent gap in the primary's top-K.
    boost = _calibrate_boost(primary_scores, cfg)
    if boost <= 0.0 or not refinement_scores:
        return out

    for layer in refinement_scores:
        if layer.shape != primary_scores.shape:
            continue
        nonzero_mask = layer > 0.0
        n_nonzero = int(nonzero_mask.sum())
        if n_nonzero < cfg.min_nonzero_for_zscore:
            continue
        # Z over the non-zero subset only.
        nz_values = layer[nonzero_mask]
        mu = float(nz_values.mean())
        sigma = float(nz_values.std()) or 1e-9
        # Compute z only where signal exists; zeros never fire.
        z = np.full_like(layer, fill_value=-np.inf, dtype=np.float64)
        z[nonzero_mask] = (nz_values - mu) / sigma
        out += np.where(z > cfg.z_threshold, boost, 0.0)

    return out


def is_layer_meaningful(
    refinement_scores: np.ndarray,
    primary_scores: np.ndarray,
    config: LayeredConfig | None = None,
    min_fire_rate: float = 0.01,
    max_fire_rate: float = 0.30,
) -> tuple[bool, str]:
    """Test whether a refinement layer is worth including for THIS dataset.

    A layer should be enabled when:

    1. **It has any non-zero signal.** ``min_nonzero_for_zscore`` non-zero
       candidates required (default 3).
    2. **Its fire rate falls in a sensible band.** Layers that almost
       never fire (< ``min_fire_rate``, default 1%) add cost without
       lift. Layers that fire on too many candidates (> ``max_fire_rate``,
       default 30%) aren't being selective - the z-threshold isn't
       gating useful information, just adding boost to most candidates.

    The fire rate is measured against the candidate pool with the
    primary scores (so the boost calibration matches what we'd actually
    use). The check is a simple gate to skip pathological layers; the
    z-threshold + boost_multiplier knobs handle the within-layer
    calibration.

    Returns ``(is_meaningful, reason)``. Reason is a short tag:
    ``"ok"`` | ``"too_few_nonzero"`` | ``"low_fire_rate"`` |
    ``"high_fire_rate"`` | ``"degenerate_primary"``.
    """
    cfg = config or LayeredConfig()
    boost = _calibrate_boost(primary_scores, cfg)
    if boost <= 0.0:
        return False, "degenerate_primary"

    nonzero_mask = refinement_scores > 0.0
    n_nonzero = int(nonzero_mask.sum())
    if n_nonzero < cfg.min_nonzero_for_zscore:
        return False, "too_few_nonzero"

    nz_values = refinement_scores[nonzero_mask]
    sigma = float(nz_values.std()) or 1e-9
    if sigma < 1e-9:
        return False, "degenerate_layer"

    mu = float(nz_values.mean())
    z = np.full_like(refinement_scores, fill_value=-np.inf, dtype=np.float64)
    z[nonzero_mask] = (nz_values - mu) / sigma
    fire_rate = float((z > cfg.z_threshold).mean())

    if fire_rate < min_fire_rate:
        return False, "low_fire_rate"
    if fire_rate > max_fire_rate:
        return False, "high_fire_rate"
    return True, "ok"


def _calibrate_boost(
    primary_scores: np.ndarray,
    cfg: LayeredConfig,
) -> float:
    """Boost magnitude = boost_multiplier × median(adjacent gaps in primary top-K).

    Adjacent gaps are computed on the K largest scores after sorting
    descending. Zero gaps (ties) are excluded so we measure the
    typical separation between distinct ranks. Returns 0 when too
    few non-zero scores to estimate.
    """
    if primary_scores.size < 2:
        return 0.0
    sorted_desc = np.sort(primary_scores)[::-1]
    k = min(cfg.top_k_for_boost_calibration, sorted_desc.size)
    top = sorted_desc[:k]
    deltas = -np.diff(top)  # >= 0 since sorted desc
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return 0.0
    return float(cfg.boost_multiplier * np.median(positive))


def diagnostic_report(
    primary_scores: np.ndarray,
    refinement_scores_by_name: dict[str, np.ndarray],
    config: LayeredConfig | None = None,
) -> dict[str, object]:
    """Return diagnostics about how the layered scorer would behave on
    this candidate pool, without applying the score. Useful for the
    probe harness to surface the boost magnitude, fire rates per
    layer, and z-distribution shape.
    """
    cfg = config or LayeredConfig()
    boost = _calibrate_boost(primary_scores, cfg)
    layers: dict[str, dict[str, object]] = {}
    for name, layer in refinement_scores_by_name.items():
        nonzero_mask = layer > 0.0
        n_nonzero = int(nonzero_mask.sum())
        if n_nonzero < cfg.min_nonzero_for_zscore:
            layers[name] = {
                "n_nonzero": n_nonzero,
                "fire_rate": 0.0,
                "would_skip": True,
            }
            continue
        nz_values = layer[nonzero_mask]
        mu = float(nz_values.mean())
        sigma = float(nz_values.std()) or 1e-9
        z = np.full_like(layer, fill_value=-np.inf, dtype=np.float64)
        z[nonzero_mask] = (nz_values - mu) / sigma
        fired = z > cfg.z_threshold
        layers[name] = {
            "n_nonzero": n_nonzero,
            "n_fired": int(fired.sum()),
            "fire_rate": float(fired.mean()),
            "z_max": float(z[nonzero_mask].max()),
            "would_skip": False,
        }
    return {
        "boost_magnitude": boost,
        "z_threshold": cfg.z_threshold,
        "boost_multiplier": cfg.boost_multiplier,
        "layers": layers,
    }
