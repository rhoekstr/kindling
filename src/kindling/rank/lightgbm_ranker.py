"""LightGBM ranker wrapper (PRD §6.3).

Activates once the Bayesian posterior variance is low enough (the
"warm regime" per PRD §3.2), training on the signal-matrix features
the engine already produces. The ranker is optional: when LightGBM
isn't installed the engine falls back to the heuristic Bayesian-
posterior-mean ranker.

Phase 10 ships the wrapper + protocol conformance + minimal unit
tests. Training orchestration (when to activate, how to source
labels) layers on top in v1.x.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from kindling.retrieve.protocol import Candidate


class LightGBMNotAvailableError(ImportError):
    """Raised when LightGBMRanker is constructed without the lightgbm
    package installed."""


def _require_lightgbm():  # type: ignore[no-untyped-def]
    try:
        import lightgbm  # noqa: F401 - imported for its side effect
    except ImportError as exc:  # pragma: no cover - optional dep
        raise LightGBMNotAvailableError(
            "LightGBMRanker requires the optional 'lightgbm' package. "
            "Install with ``pip install lightgbm``."
        ) from exc
    import lightgbm as lgb

    return lgb


class LightGBMRanker:
    """LambdaRank via LightGBM with kindling-friendly defaults.

    Attributes
    ----------
    name:
        ``"lightgbm_lambdarank"``. Used in debug payloads and
        persistence manifests.
    num_leaves / learning_rate / n_estimators:
        Hyperparameters passed through to ``LGBMRanker``. Defaults
        chosen to be conservative (fast training, low overfit risk on
        moderate data volumes).
    """

    name = "lightgbm_lambdarank"

    def __init__(
        self,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 0,
    ) -> None:
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
    ) -> None:
        """Train the LambdaRank model.

        Parameters
        ----------
        features:
            ``(n_obs, k_signals)`` feature matrix.
        labels:
            ``(n_obs,)`` graded relevance.
        groups:
            ``(n_groups,)`` group sizes.
        """
        lgb = _require_lightgbm()
        self._model = lgb.LGBMRanker(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbose=-1,
        )
        # Fit on a bare ndarray so predict() matches at inference time
        # without the "X does not have valid feature names" warning.
        self._model.fit(np.asarray(features), labels, group=groups)

    def score(
        self,
        candidates: list["Candidate"],
        owned_items: np.ndarray,  # noqa: ARG002 - protocol alignment
    ) -> np.ndarray:
        """Fall-through scorer when no SignalFeatures are available.

        Returns the retriever-stage scores. The engine's warm-regime
        recommend path uses ``score_features`` below with the full 9-
        column feature matrix; this path only fires when the caller
        bypasses feature computation.
        """
        if not candidates:
            return np.array([], dtype=np.float64)
        return np.array([c.score for c in candidates], dtype=np.float64)

    def score_features(self, features: np.ndarray) -> np.ndarray:
        """Score a (n_candidates, k_signals) feature matrix.

        Returns the raw LambdaRank output. The engine routes this as the
        final candidate score when ``is_fitted``; otherwise the Bayesian-
        blend posterior mean is used (cold-regime fallback).
        """
        import warnings

        if self._model is None:
            raise RuntimeError("LightGBMRanker.score_features called before fit()")
        # sklearn 1.8 complains when we predict on a bare ndarray but the
        # model was fitted on one too - benign mismatch because lightgbm
        # doesn't actually use the names for scoring.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            pred = self._model.predict(features)  # type: ignore[attr-defined]
        return np.asarray(pred, dtype=np.float64)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None


class NoRanker:
    """Pass-through ranker. Returns each candidate's retriever-stage
    score unchanged. Useful when the caller wants to disable the
    learned ranker path without pulling in LightGBM."""

    name = "no_ranker"

    def score(
        self,
        candidates: list["Candidate"],
        owned_items: np.ndarray,  # noqa: ARG002 - protocol alignment
    ) -> np.ndarray:
        if not candidates:
            return np.array([], dtype=np.float64)
        return np.array([c.score for c in candidates], dtype=np.float64)
