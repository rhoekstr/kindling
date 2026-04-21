"""Ranker protocol tests (plan Phase 10, PRD §6.3)."""

from __future__ import annotations

import numpy as np
import pytest

from kindling.rank import (
    HeuristicRanker,
    LightGBMNotAvailableError,
    LightGBMRanker,
    NoRanker,
    RankerProtocol,
)
from kindling.retrieve.protocol import Candidate


def _candidates() -> list[Candidate]:
    return [
        Candidate(item_id=1, score=0.9, source="test"),
        Candidate(item_id=2, score=0.7, source="test"),
        Candidate(item_id=3, score=0.2, source="test"),
    ]


def test_no_ranker_passthrough() -> None:
    ranker = NoRanker()
    scores = ranker.score(_candidates(), owned_items=np.array([]))
    np.testing.assert_allclose(scores, [0.9, 0.7, 0.2])


def test_heuristic_ranker_returns_retriever_scores() -> None:
    ranker = HeuristicRanker()
    scores = ranker.score(_candidates(), owned_items=np.array([]))
    np.testing.assert_allclose(scores, [0.9, 0.7, 0.2])


def test_all_rankers_implement_protocol() -> None:
    for ranker in (NoRanker(), HeuristicRanker(), LightGBMRanker()):
        assert isinstance(ranker, RankerProtocol)


def test_lightgbm_ranker_pre_train_passthrough() -> None:
    """Before fit, LightGBMRanker returns the retriever-stage scores
    (kindling heuristic path)."""
    ranker = LightGBMRanker()
    scores = ranker.score(_candidates(), owned_items=np.array([]))
    np.testing.assert_allclose(scores, [0.9, 0.7, 0.2])


def test_lightgbm_ranker_fit_raises_without_lightgbm() -> None:
    """When lightgbm isn't installed, ``fit`` raises a helpful error."""
    # lightgbm may or may not be importable here; run only when it isn't.
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        ranker = LightGBMRanker()
        with pytest.raises(LightGBMNotAvailableError, match="lightgbm"):
            ranker.fit(
                features=np.zeros((5, 3)),
                labels=np.zeros(5),
                groups=np.array([5]),
            )
    else:
        pytest.skip("lightgbm is installed; this path only tests the stub")


def test_empty_candidates_returns_empty_array() -> None:
    for ranker in (NoRanker(), HeuristicRanker(), LightGBMRanker()):
        out = ranker.score([], owned_items=np.array([]))
        assert out.shape == (0,)
