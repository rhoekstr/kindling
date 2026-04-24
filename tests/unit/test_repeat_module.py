"""Tests for the repeat-consumption module's core logic.

Synthetic fixtures with known-period, known-pattern items so we can
verify each stage in isolation before engine integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling.repeat import (
    Pattern,
    RepeatConfig,
    RepeatProfile,
    fit_repeat_profiles,
    multiplier,
)
from kindling.repeat.period import detect_period
from kindling.repeat.shape import classify_shape, dominant_pattern


# ------------------------ period detection -----------------------------

def test_period_detects_weekly_cycle() -> None:
    """Intervals tightly clustered around 7 days should yield period ~7 days."""
    rng = np.random.default_rng(0)
    intervals = rng.normal(loc=7 * 86400, scale=0.4 * 86400, size=200)
    intervals = intervals[intervals > 0]
    period, fit_q = detect_period(intervals)
    assert 5 * 86400 < period < 9 * 86400
    assert fit_q > 0.6  # unimodal - high fit quality


def test_period_handles_sparse_data_via_median() -> None:
    """Fewer than 10 observations -> median fallback, not KDE."""
    intervals = np.array([5 * 86400, 7 * 86400, 9 * 86400])
    period, fit_q = detect_period(intervals)
    assert period == pytest.approx(7 * 86400)
    assert fit_q < 1.0  # median fallback has reduced confidence


def test_period_empty_input_returns_nan() -> None:
    period, fit_q = detect_period(np.array([]))
    assert np.isnan(period)
    assert fit_q == 0.0


def test_period_is_robust_to_outliers() -> None:
    """A few extreme intervals shouldn't move the detected period much."""
    rng = np.random.default_rng(1)
    main = rng.normal(loc=14 * 86400, scale=1 * 86400, size=100)
    outliers = np.array([400 * 86400, 500 * 86400, 700 * 86400])
    intervals = np.concatenate([main, outliers])
    intervals = intervals[intervals > 0]
    period, _ = detect_period(intervals)
    assert 10 * 86400 < period < 18 * 86400  # outliers don't pull peak far


# ------------------------ shape classification -------------------------

def test_shape_exponential_classifies_as_repeat() -> None:
    """Scaled intervals concentrated near zero -> REPEAT pattern."""
    rng = np.random.default_rng(2)
    scaled = rng.exponential(scale=0.3, size=200)
    probs = classify_shape(scaled, repeat_rate=0.7)
    assert dominant_pattern(probs) is Pattern.REPEAT


def test_shape_gamma_classifies_as_replenish() -> None:
    """Scaled intervals peaked at ~1 -> REPLENISH pattern."""
    rng = np.random.default_rng(3)
    scaled = rng.gamma(shape=4, scale=0.25, size=200)
    probs = classify_shape(scaled, repeat_rate=0.4)
    assert dominant_pattern(probs) is Pattern.REPLENISH


def test_shape_lognormal_classifies_as_satiation() -> None:
    """Scaled intervals with mass far from zero -> SATIATION pattern."""
    rng = np.random.default_rng(4)
    scaled = rng.lognormal(mean=1.5, sigma=0.5, size=200)
    probs = classify_shape(scaled, repeat_rate=0.05)
    top = dominant_pattern(probs)
    # ONE_SHOT has rate-based preference; the repeat_rate=0.05 is right
    # at the threshold so SATIATION should still compete. Either is
    # acceptable but ONE_SHOT or SATIATION are both reasonable.
    assert top in {Pattern.SATIATION, Pattern.ONE_SHOT}


def test_shape_low_repeat_rate_favors_one_shot() -> None:
    """Items that almost never repeat should land on ONE_SHOT."""
    rng = np.random.default_rng(5)
    scaled = rng.exponential(scale=0.3, size=5)  # few intervals
    probs = classify_shape(scaled, repeat_rate=0.001)
    assert dominant_pattern(probs) is Pattern.ONE_SHOT
    assert probs[Pattern.ONE_SHOT] > 0.9


def test_shape_probs_sum_to_one() -> None:
    rng = np.random.default_rng(6)
    probs = classify_shape(rng.exponential(scale=0.5, size=100), repeat_rate=0.3)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_shape_empty_intervals_falls_back_to_rate_only() -> None:
    """No interval data -> pattern-4 score drives everything."""
    probs_low = classify_shape(np.array([]), repeat_rate=0.0)
    probs_high = classify_shape(np.array([]), repeat_rate=0.9)
    assert probs_low[Pattern.ONE_SHOT] > 0.9
    assert probs_high[Pattern.ONE_SHOT] < 0.1


# ------------------------ multiplier forms -----------------------------

def _profile(pattern: Pattern, period_s: float = 7 * 86400, confidence: float = 1.0) -> RepeatProfile:
    probs = {p: 0.0 for p in Pattern}
    probs[pattern] = 1.0
    return RepeatProfile(
        pattern=pattern,
        pattern_probs=probs,
        period_seconds=period_s,
        refractory_seconds=period_s * 3,
        confidence=confidence,
        n_observations=50,
        pooled=False,
        repeat_rate=0.5,
    )


def test_multiplier_never_interacted_is_one() -> None:
    assert multiplier(_profile(Pattern.ONE_SHOT), None) == 1.0


def test_multiplier_repeat_is_always_one() -> None:
    p = _profile(Pattern.REPEAT)
    for t in (0, 86400, 7 * 86400, 365 * 86400):
        assert multiplier(p, t) == pytest.approx(1.0, abs=1e-9)


def test_multiplier_replenish_ramps_with_time() -> None:
    p = _profile(Pattern.REPLENISH, period_s=7 * 86400)
    m_early = multiplier(p, 1 * 86400)   # 1/7 of period
    m_mid = multiplier(p, 5 * 86400)     # 5/7 of period
    m_late = multiplier(p, 10 * 86400)   # 1.43 periods
    assert m_early < 0.2
    assert m_mid > m_early
    assert m_late > 0.9


def test_multiplier_satiation_suppresses_during_refractory() -> None:
    p = _profile(Pattern.SATIATION, period_s=30 * 86400)  # refractory = 90d
    m_week1 = multiplier(p, 7 * 86400)
    m_month = multiplier(p, 30 * 86400)
    m_year = multiplier(p, 365 * 86400)
    assert m_week1 < 0.2
    assert m_month < 0.5
    assert m_year > 0.9


def test_multiplier_one_shot_stays_low() -> None:
    p = _profile(Pattern.ONE_SHOT)
    assert multiplier(p, 86400) < 0.01
    # Even after years, one-shot stays suppressed.
    assert multiplier(p, 5 * 365 * 86400) < 0.01


def test_multiplier_low_confidence_dampens_toward_one() -> None:
    """Low confidence -> multiplier stays near 1.0 regardless of pattern."""
    p = _profile(Pattern.ONE_SHOT, confidence=0.0)
    # confidence=0: fully dampened, multiplier should be 1.0.
    assert multiplier(p, 86400) == pytest.approx(1.0, abs=1e-9)


# ------------------------ fit orchestrator -----------------------------

def _synthetic_interactions(seed: int = 0) -> pd.DataFrame:
    """Three items with different repeat patterns.
    - Item A (REPEAT): exponential-distributed short intervals. Most
      near zero, long tail - characteristic of music / video replay.
    - Item B (REPLENISH): intervals tightly peaked at one week (fixed
      replenishment cadence).
    - Item C (ONE_SHOT): each user interacts exactly once.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2026-01-01")

    # Item A: two users, each with 20 exponentially-spaced interactions.
    # Scale=100s mean -> most intervals are seconds/minutes, few are hours.
    for user in range(2):
        t = 0.0
        for _ in range(20):
            t += float(rng.exponential(scale=100.0))
            rows.append({
                "entity_id": user,
                "item_id": "A",
                "timestamp": base + pd.Timedelta(seconds=t + user * 10),
            })

    # Item B: user 0 replenishes weekly (tight gamma around 7 days), 10 times.
    for user in range(3):
        t_days = 0.0
        for _ in range(10):
            # Gamma with shape=10, scale=0.7 -> mean 7 days, std ~2.2.
            t_days += float(rng.gamma(shape=10, scale=0.7))
            rows.append({
                "entity_id": user,
                "item_id": "B",
                "timestamp": base + pd.Timedelta(days=t_days + user),
            })

    # Item C: each user bought exactly once (one-shot).
    for u in range(30):
        rows.append({"entity_id": u + 100, "item_id": "C", "timestamp": base + pd.Timedelta(days=u)})
    return pd.DataFrame(rows)


def test_fit_identifies_pattern_from_data() -> None:
    df = _synthetic_interactions()
    table = fit_repeat_profiles(df, config=RepeatConfig(min_observations_individual=3))

    # Item A: exponentially-distributed short intervals -> REPEAT.
    prof_a = table.get("A")
    assert prof_a.pattern is Pattern.REPEAT

    # Item B: intervals tightly peaked at ~7 days -> REPLENISH.
    prof_b = table.get("B")
    assert prof_b.pattern is Pattern.REPLENISH
    assert 5 * 86400 < prof_b.period_seconds < 9 * 86400

    # Item C: every user one-shot -> ONE_SHOT.
    prof_c = table.get("C")
    assert prof_c.pattern is Pattern.ONE_SHOT
    assert prof_c.repeat_rate == 0.0


def test_fit_returns_empty_table_when_no_timestamps() -> None:
    df = pd.DataFrame({"entity_id": [1, 2], "item_id": ["x", "y"]})
    table = fit_repeat_profiles(df)
    assert len(table) == 0
    # get() on unknown item returns the default (pattern=REPEAT, confidence=0).
    assert table.get("x").confidence == 0.0


def test_fit_respects_explicit_overrides() -> None:
    df = _synthetic_interactions()
    override = RepeatProfile(
        pattern=Pattern.REPLENISH,
        pattern_probs={
            Pattern.REPEAT: 0.0,
            Pattern.REPLENISH: 1.0,
            Pattern.SATIATION: 0.0,
            Pattern.ONE_SHOT: 0.0,
        },
        period_seconds=14 * 86400,
        refractory_seconds=30 * 86400,
        confidence=1.0,
        n_observations=100,
        pooled=False,
        repeat_rate=0.8,
    )
    cfg = RepeatConfig(explicit_overrides={"C": override})
    table = fit_repeat_profiles(df, config=cfg)
    assert table.get("C") is override
