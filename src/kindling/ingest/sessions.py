"""Session inference (PRD §4.4).

Two strategies:

1. Explicit ``session_id`` present - use it directly.
2. Timestamps only - fit a 2-component GMM on log-transformed inter-event
   deltas per entity, find the gap threshold between within-session and
   cross-session behavior. The PRD promises "domain-agnostic" inference.

Plan gap from the PRD: a goodness-of-fit check. If the 2-component GMM does
not meaningfully outperform a 1-component fit (by log-likelihood ratio), the
inter-event distribution isn't bimodal - forcing a threshold would be
arbitrary. Fall back to a configurable manual threshold (default 30 minutes)
and surface a warning so the user knows their data doesn't look like sessions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

DEFAULT_MANUAL_THRESHOLD_SECONDS = 30 * 60  # 30 minutes
_GOF_MIN_LLR_GAIN = 10.0  # 2-component must beat 1-component by this much
_MIN_SAMPLES_FOR_GMM = 50


@dataclass(frozen=True)
class SessionInference:
    """Result of session inference.

    Attributes
    ----------
    session_ids:
        Int array aligned with the input interactions, one session id per row.
    gap_threshold_seconds:
        The inferred (or configured) gap beyond which a new session starts.
    strategy:
        ``"explicit"``, ``"gmm"``, or ``"manual_fallback"``.
    gof_log_likelihood_ratio:
        Log-likelihood ratio of 2-component over 1-component Gaussian fit on
        log-transformed deltas. Only populated when the GMM is attempted.
    """

    session_ids: np.ndarray
    gap_threshold_seconds: float
    strategy: str
    gof_log_likelihood_ratio: float | None = None


def infer_sessions(
    interactions: pd.DataFrame,
    manual_threshold_seconds: float = DEFAULT_MANUAL_THRESHOLD_SECONDS,
) -> SessionInference:
    """Assign a session id to each row of ``interactions``.

    Input must be validated (via ``ingest.contract.validate_interactions``)
    and canonicalized - requires at minimum ``entity_id`` and ``item_id``.
    """
    if "session_id" in interactions.columns:
        return SessionInference(
            session_ids=interactions["session_id"].to_numpy(),
            gap_threshold_seconds=0.0,
            strategy="explicit",
        )

    if "timestamp" not in interactions.columns:
        # No time signal - every (entity, row) is its own session of length 1.
        # Downstream structures (PathTree, TailIndex) won't fire meaningfully.
        return SessionInference(
            session_ids=np.arange(len(interactions), dtype=np.int64),
            gap_threshold_seconds=0.0,
            strategy="manual_fallback",
        )

    # Compute inter-event deltas within each entity.
    sorted_df = interactions.sort_values(["entity_id", "timestamp"], kind="mergesort")
    ts_seconds = sorted_df["timestamp"].astype("int64").to_numpy() // 10**9
    entity_ids = sorted_df["entity_id"].to_numpy()

    same_entity = np.zeros(len(sorted_df), dtype=bool)
    same_entity[1:] = entity_ids[1:] == entity_ids[:-1]
    deltas_seconds = np.zeros(len(sorted_df), dtype=np.float64)
    deltas_seconds[1:] = np.where(same_entity[1:], ts_seconds[1:] - ts_seconds[:-1], 0.0)

    within_entity_deltas = deltas_seconds[same_entity]
    within_entity_deltas = within_entity_deltas[within_entity_deltas > 0]

    threshold_seconds = manual_threshold_seconds
    strategy = "manual_fallback"
    llr: float | None = None

    if len(within_entity_deltas) >= _MIN_SAMPLES_FOR_GMM:
        log_deltas = np.log(within_entity_deltas)
        threshold, llr = _fit_gmm_threshold(log_deltas)
        if threshold is not None and llr is not None and llr >= _GOF_MIN_LLR_GAIN:
            threshold_seconds = float(np.exp(threshold))
            strategy = "gmm"
        else:
            warnings.warn(
                "GMM session inference did not find a clear bimodal gap "
                f"(log-likelihood ratio {llr:.2f} < {_GOF_MIN_LLR_GAIN}). "
                f"Falling back to manual threshold of {manual_threshold_seconds}s.",
                stacklevel=2,
            )

    # Assign session ids by cutting at gaps exceeding the threshold, respecting
    # entity boundaries.
    is_new_session = (~same_entity) | (deltas_seconds > threshold_seconds)
    session_ids_sorted = np.cumsum(is_new_session)

    # Restore original row order so the returned array aligns with the input.
    sorted_df = sorted_df.assign(_session_id=session_ids_sorted)
    session_ids = sorted_df.sort_index()["_session_id"].to_numpy()

    return SessionInference(
        session_ids=session_ids,
        gap_threshold_seconds=threshold_seconds,
        strategy=strategy,
        gof_log_likelihood_ratio=llr,
    )


def _fit_gmm_threshold(log_deltas: np.ndarray) -> tuple[float | None, float | None]:
    """Fit a 2-component Gaussian mixture on log-transformed deltas; return
    the threshold (log-seconds) and the 2-vs-1 component log-likelihood ratio.

    Uses a minimal hand-rolled EM to avoid pulling in scikit-learn as a hard
    dependency for one algorithm.
    """
    rng = np.random.default_rng(seed=0)
    n = len(log_deltas)
    if n < _MIN_SAMPLES_FOR_GMM:
        return None, None

    # 1-component fit: mean/std.
    mu_1 = float(log_deltas.mean())
    sigma_1 = float(log_deltas.std(ddof=0)) or 1e-6
    ll_1 = float(norm.logpdf(log_deltas, loc=mu_1, scale=sigma_1).sum())

    # 2-component EM with sensible init: percentile split.
    p25, p75 = np.percentile(log_deltas, [25, 75])
    mu = np.array([p25, p75], dtype=np.float64)
    if mu[0] >= mu[1]:
        mu = np.array([mu_1 - 1.0, mu_1 + 1.0])
    sigma = np.array([sigma_1, sigma_1], dtype=np.float64)
    pi = np.array([0.5, 0.5], dtype=np.float64)

    for _ in range(100):
        # E-step
        r = _responsibilities(log_deltas, mu, sigma, pi)
        # M-step
        nk = r.sum(axis=0)
        if np.any(nk < 1.0):
            # Degenerate component - bail to 1-component.
            return None, ll_1 - ll_1
        mu = (r * log_deltas[:, None]).sum(axis=0) / nk
        sigma_sq = (r * (log_deltas[:, None] - mu) ** 2).sum(axis=0) / nk
        sigma = np.sqrt(np.maximum(sigma_sq, 1e-8))
        pi = nk / n

    ll_2 = float(
        np.log(
            pi[0] * norm.pdf(log_deltas, loc=mu[0], scale=sigma[0])
            + pi[1] * norm.pdf(log_deltas, loc=mu[1], scale=sigma[1])
            + 1e-300
        ).sum()
    )
    if mu[0] > mu[1]:
        mu = mu[::-1]
        sigma = sigma[::-1]

    # Threshold: midpoint where the two component densities intersect.
    threshold = _gaussian_crossover(mu[0], sigma[0], mu[1], sigma[1])
    if threshold is None or not (mu[0] < threshold < mu[1]):
        threshold = float((mu[0] + mu[1]) / 2.0)
    _ = rng  # reserved for future multi-restart init
    return float(threshold), ll_2 - ll_1


def _responsibilities(
    x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, pi: np.ndarray
) -> np.ndarray:
    """Per-sample soft assignment to each component."""
    px = np.stack(
        [pi[k] * norm.pdf(x, loc=mu[k], scale=sigma[k]) for k in range(len(mu))],
        axis=1,
    )
    total = px.sum(axis=1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    return px / total


def _gaussian_crossover(mu_a: float, sigma_a: float, mu_b: float, sigma_b: float) -> float | None:
    """Return the log-space crossover of two normals (the point where their
    pdfs are equal, between the means)."""
    a = 1.0 / (2 * sigma_a**2) - 1.0 / (2 * sigma_b**2)
    b = mu_b / (sigma_b**2) - mu_a / (sigma_a**2)
    c = mu_a**2 / (2 * sigma_a**2) - mu_b**2 / (2 * sigma_b**2) - np.log(sigma_b / sigma_a)
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return None
        return float(-c / b)
    disc = b**2 - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    roots = [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]
    lo, hi = sorted([mu_a, mu_b])
    for r in roots:
        if lo <= r <= hi:
            return float(r)
    return None
