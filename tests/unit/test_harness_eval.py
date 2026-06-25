"""Eval-harness evaluator + baselines (popularity only — no optional deps)."""

from __future__ import annotations

import pytest

from kindling.harness import EvalReport, evaluate, format_report
from kindling.harness.baselines import PopularityBaseline, available_baselines, build_baselines
from kindling.loaders import synthetic


@pytest.fixture
def split():
    return synthetic.make_ratings(n_entities=120, n_items=80, ratings_per_entity=20, seed=0)


def test_evaluate_returns_structured_report(split) -> None:
    report = evaluate(
        split.train, split.test, dataset="syn", k=10, baselines=["popularity"], max_users=200
    )
    assert isinstance(report, EvalReport)
    assert report.dataset == "syn"
    assert report.models[0] == "kindling" and "popularity" in report.models
    assert report.n_items > 0 and report.n_eval_users > 0
    assert report.base_scorer  # non-empty
    # An "all" bucket is always present when there are eval users.
    assert any(b.bucket == "all" for b in report.buckets)
    for b in report.buckets:
        assert set(report.models) <= set(b.metrics)
        assert "ndcg@10" in b.metrics["kindling"]


def test_kindling_beats_popularity_overall(split) -> None:
    report = evaluate(
        split.train, split.test, dataset="syn", baselines=["popularity"], max_users=200
    )
    k_ndcg = report.metric("kindling", "all", "ndcg@10")
    p_ndcg = report.metric("popularity", "all", "ndcg@10")
    assert k_ndcg > 0.0
    assert k_ndcg >= p_ndcg  # personalization should not lose to raw popularity here


def test_format_report_mentions_models_and_metric(split) -> None:
    report = evaluate(split.train, split.test, dataset="syn", baselines=["popularity"])
    text = format_report(report)
    assert "NDCG@10" in text
    assert "kindling" in text and "popularity" in text
    assert "syn" in text


def test_unknown_baseline_is_skipped_not_fatal(split) -> None:
    report = evaluate(
        split.train, split.test, baselines=["popularity", "not-a-model"], max_users=100
    )
    assert any("not-a-model" in s for s in report.skipped_baselines)
    assert "not-a-model" not in report.models


def test_available_baselines_always_has_popularity() -> None:
    assert "popularity" in available_baselines()


def test_popularity_baseline_excludes_owned(split) -> None:
    built, _ = build_baselines(split.train, ["popularity"])
    assert isinstance(built[0], PopularityBaseline)
    owned = {split.train["item_id"].iloc[0]}
    recs = built[0].recommend(entity_id="anyone", owned=owned, k=5)
    assert owned.isdisjoint(recs)
    assert len(recs) <= 5
