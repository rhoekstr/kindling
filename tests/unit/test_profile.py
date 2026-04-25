"""Tests for DatasetProfile + plan_layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from kindling.profile import DatasetProfile, LayerPlan, plan_layers, profile_dataset
from kindling.profile.profile import _bucket_density, _bucket_session_depth


@dataclass
class _MockSessionInference:
    session_ids: np.ndarray
    strategy: str
    gap_threshold_seconds: float
    gof_log_likelihood_ratio: float | None = None


@dataclass
class _MockKernel:
    strategy: str
    midpoint_seconds: float


def test_bucket_density() -> None:
    assert _bucket_density(2) == "sparse"
    assert _bucket_density(15) == "moderate"
    assert _bucket_density(100) == "dense"
    assert _bucket_density(500) == "very_dense"


def test_bucket_session_depth() -> None:
    # No deep sessions -> "none"
    assert _bucket_session_depth(median_size=1.0, deep_fraction=0.01) == "none"
    # Shallow: median 2 with 50% deep
    assert _bucket_session_depth(median_size=2.0, deep_fraction=0.5) == "shallow"
    # Moderate
    assert _bucket_session_depth(median_size=4.0, deep_fraction=0.8) == "moderate"
    # Deep
    assert _bucket_session_depth(median_size=10.0, deep_fraction=0.9) == "deep"
    assert _bucket_session_depth(median_size=20.0, deep_fraction=0.95) == "very_deep"


def test_profile_minimum_size() -> None:
    df = pd.DataFrame({
        "entity_id": [1, 1, 2, 2, 3, 3, 3],
        "item_id": [1, 2, 1, 3, 2, 3, 4],
    })
    p = profile_dataset(df)
    assert p.n_users == 3
    assert p.n_items == 4
    assert p.n_interactions == 7
    assert p.user_density == "sparse"  # avg 2.33 events/user
    assert p.has_timestamps is False
    assert p.has_explicit_sessions is False
    assert p.repeat_user_fraction == 0.0  # no repeats


def test_profile_repeat_detection() -> None:
    """Repeat-dataset gate fires when >=10% of users have repeat pairs."""
    rows = []
    # 50 users, 20 with repeat (user, item) pairs.
    for u in range(50):
        rows.append({"entity_id": u, "item_id": 1})
        rows.append({"entity_id": u, "item_id": 2})
        if u < 20:
            rows.append({"entity_id": u, "item_id": 1})  # repeat
    df = pd.DataFrame(rows)
    p = profile_dataset(df)
    assert p.repeat_dataset is True
    assert p.repeat_user_fraction == pytest.approx(0.4)


def test_profile_rating_burst_classification() -> None:
    """Rating-burst kernel pushes time_use to 'rating_burst'."""
    base = pd.Timestamp("2024-01-01").value // 10**9
    rows = [
        {"entity_id": u, "item_id": i,
         "timestamp": pd.to_datetime(base + u * 1000 + i, unit="s")}
        for u in range(20) for i in range(5)
    ]
    df = pd.DataFrame(rows)
    p = profile_dataset(
        df,
        kernel_params=_MockKernel(strategy="rating_burst_detected", midpoint_seconds=87),
    )
    assert p.time_use == "rating_burst"


def test_profile_session_consumption_classification() -> None:
    """GMM with minutes-scale midpoint -> session_consumption."""
    base = pd.Timestamp("2024-01-01").value // 10**9
    rows = [
        {"entity_id": u, "item_id": i,
         "timestamp": pd.to_datetime(base + u * 86400 + i * 30, unit="s")}
        for u in range(20) for i in range(5)
    ]
    df = pd.DataFrame(rows)
    p = profile_dataset(
        df,
        kernel_params=_MockKernel(strategy="gmm", midpoint_seconds=600),
    )
    assert p.time_use == "session_consumption"


def test_profile_long_horizon_classification() -> None:
    base = pd.Timestamp("2024-01-01").value // 10**9
    rows = [
        {"entity_id": u, "item_id": i,
         "timestamp": pd.to_datetime(base + u * 86400 * 30 + i * 86400 * 7, unit="s")}
        for u in range(20) for i in range(5)
    ]
    df = pd.DataFrame(rows)
    p = profile_dataset(
        df,
        kernel_params=_MockKernel(strategy="gmm", midpoint_seconds=86400 * 7),
    )
    assert p.time_use == "long_horizon"


def test_plan_skeleton_for_no_timestamp_dataset() -> None:
    """No timestamps + shallow sessions: just item_graph + path_tail."""
    df = pd.DataFrame({"entity_id": [1, 2, 3], "item_id": [1, 2, 3]})
    profile = profile_dataset(df)
    plan = plan_layers(profile)
    assert "item_graph" in plan.enabled_subsystems
    assert "path_tail" in plan.enabled_subsystems
    assert plan.temporal_kernel_active is False
    assert plan.repeat_module_active is False
    assert "session_cooc_graph" not in plan.enabled_subsystems


def test_plan_session_rich_dataset() -> None:
    """Deep-session timestamped data: enable session_cooc + path_basket."""
    base = pd.Timestamp("2024-01-01").value // 10**9
    rows = []
    for u in range(50):
        for s in range(5):
            for j in range(8):  # 8 items per session = deep
                rows.append({
                    "entity_id": u,
                    "item_id": int(np.random.default_rng(u * 100 + s).integers(0, 50)),
                    "timestamp": pd.to_datetime(base + u * 86400 * 7 + s * 86400 + j * 60, unit="s"),
                    "session_id": u * 10 + s,
                })
    df = pd.DataFrame(rows)
    profile = profile_dataset(
        df,
        kernel_params=_MockKernel(strategy="gmm", midpoint_seconds=1800),
    )
    plan = plan_layers(profile)
    assert "session_cooc_graph" in plan.enabled_subsystems
    assert "path_basket" in plan.enabled_subsystems
    assert "temporal_graph" in plan.enabled_subsystems
    assert plan.temporal_kernel_active is True
    assert "session_cooccurrence" in plan.enabled_boost_layers
    assert "path_basket" in plan.enabled_boost_layers
    assert "temporal_cooccurrence" in plan.enabled_boost_layers


def test_plan_rating_burst_skips_session_layers() -> None:
    """Rating-burst regime: skip session_cooc + path_basket."""
    base = pd.Timestamp("2024-01-01").value // 10**9
    rows = [
        {"entity_id": u, "item_id": i,
         "timestamp": pd.to_datetime(base + u * 1000 + i, unit="s")}
        for u in range(50) for i in range(8)
    ]
    df = pd.DataFrame(rows)
    profile = profile_dataset(
        df,
        kernel_params=_MockKernel(strategy="rating_burst_detected", midpoint_seconds=87),
    )
    plan = plan_layers(profile)
    assert "session_cooc_graph" not in plan.enabled_subsystems
    assert "path_basket" not in plan.enabled_subsystems
    # Still allow temporal_graph (history-cap effect even when kernel off).
    assert "temporal_graph" in plan.enabled_subsystems
    assert plan.temporal_kernel_active is False
    # Boost layer for temporal_cooc still wired (history-cap version).
    assert "temporal_cooccurrence" in plan.enabled_boost_layers


def test_plan_rationale_explains_decisions() -> None:
    df = pd.DataFrame({"entity_id": [1, 2, 3], "item_id": [1, 2, 3]})
    profile = profile_dataset(df)
    plan = plan_layers(profile)
    # Every decision should have a rationale.
    for key in ("temporal_graph", "session_cooc_graph", "path_basket",
                "temporal_kernel", "repeat_module"):
        assert key in plan.rationale
        assert plan.rationale[key]


def test_plan_summary_is_human_readable(capsys) -> None:
    df = pd.DataFrame({"entity_id": [1, 2, 3], "item_id": [1, 2, 3]})
    plan = plan_layers(profile_dataset(df))
    summary = plan.summary()
    assert "subsystems" in summary
    assert "boost_layers" in summary
    assert "temporal_kernel" in summary


def test_profile_summary_is_human_readable() -> None:
    df = pd.DataFrame({"entity_id": [1, 2, 3], "item_id": [1, 2, 3]})
    p = profile_dataset(df)
    s = p.summary()
    assert "interactions" in s
    assert "users" in s
    assert "items" in s
