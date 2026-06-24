"""Gating network: per-entity softmax weights over signals, pure numpy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling.engine import Engine
from kindling.gate import (
    GatingConfig,
    GatingNetwork,
    compute_context_features,
    fit_gating_network,
)
from kindling.gate.features import CONTEXT_FEATURE_NAMES


def _toy_df(n_users: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-01")
    rows: list[dict[str, object]] = []
    for u in range(n_users):
        group = u // 100  # two groups
        items = list(range(group * 10, group * 10 + 10))
        picks = rng.choice(items, size=5, replace=False)
        for i, pick in enumerate(picks):
            rows.append({
                "entity_id": u,
                "item_id": int(pick),
                "timestamp": base + pd.Timedelta(days=i),
            })
    return pd.DataFrame(rows)


def test_gate_initializes_with_valid_shapes() -> None:
    rng = np.random.default_rng(0)
    gate = GatingNetwork.initialize(n_ctx=8, n_signals=11, hidden_dim=16, rng=rng)
    assert gate.W1.shape == (16, 8)
    assert gate.W2.shape == (11, 16)
    assert gate.b1.shape == (16,)
    assert gate.b2.shape == (11,)


def test_gate_forward_produces_softmax_distribution() -> None:
    rng = np.random.default_rng(0)
    gate = GatingNetwork.initialize(n_ctx=4, n_signals=5, hidden_dim=8, rng=rng)
    gate.ctx_mean = np.zeros(4, dtype=np.float32)
    gate.ctx_std = np.ones(4, dtype=np.float32)
    out = gate.forward(np.array([1.0, 0.5, -0.2, 0.0]))
    assert out.shape == (5,)
    assert out.sum() == pytest.approx(1.0, abs=1e-6)
    assert (out >= 0).all()


def test_gate_batch_forward_matches_per_entity() -> None:
    rng = np.random.default_rng(0)
    gate = GatingNetwork.initialize(n_ctx=4, n_signals=5, hidden_dim=8, rng=rng)
    gate.ctx_mean = np.zeros(4, dtype=np.float32)
    gate.ctx_std = np.ones(4, dtype=np.float32)
    ctx = np.array([[1.0, 0.5, -0.2, 0.0], [0.1, 0.2, 0.3, 0.4]])
    batch_out = gate.forward_batch(ctx)
    single_0 = gate.forward(ctx[0])
    single_1 = gate.forward(ctx[1])
    assert np.allclose(batch_out[0], single_0, atol=1e-6)
    assert np.allclose(batch_out[1], single_1, atol=1e-6)


def test_context_features_produce_known_columns() -> None:
    df = _toy_df()
    engine = Engine(signal_normalization="zscore").fit(df)
    ctx = compute_context_features(engine)
    assert len(ctx) > 0
    any_vec = next(iter(ctx.values()))
    assert any_vec.shape == (len(CONTEXT_FEATURE_NAMES),)
    # Sanity: log(n_interactions) = log(5 + 1) ≈ 1.79 for each test user.
    assert 1.0 < any_vec[0] < 3.0


def test_gate_fits_on_toy_dataset() -> None:
    df = _toy_df()
    engine = Engine(signal_normalization="zscore").fit(df)
    cfg = GatingConfig(
        enabled=True,
        n_epochs=3,
        batch_size=64,
        min_users=20,
        min_positives=50,
        seed=0,
    )
    gate = fit_gating_network(engine, cfg)
    assert gate is not None
    # Check gate produces reasonable output: softmax-distributed weights.
    ctx_by_entity = compute_context_features(engine)
    sample_entity = next(iter(ctx_by_entity))
    weights = gate.forward(ctx_by_entity[sample_entity])
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert weights.shape == (11,)


def test_gate_end_to_end_engine_integration() -> None:
    """Engine(gating_config=GatingConfig(enabled=True)) should fit the gate,
    switch signal_normalization to zscore, and use gate weights for scoring."""
    df = _toy_df()
    cfg = GatingConfig(
        enabled=True,
        n_epochs=2,
        batch_size=64,
        min_users=20,
        min_positives=50,
        seed=0,
    )
    engine = Engine(gating_config=cfg).fit(df)
    assert engine.signal_normalization == "zscore"
    assert engine._gate is not None
    recs = engine.recommend(entity_id=0, n=5)
    # Gate-scored recs should still have a sensible ordering (monotone).
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_gate_skips_when_too_few_users() -> None:
    tiny = pd.DataFrame({
        "entity_id": [1, 2, 3],
        "item_id": [10, 20, 30],
    })
    engine = Engine().fit(tiny)
    cfg = GatingConfig(enabled=True, min_users=100)
    gate = fit_gating_network(engine, cfg)
    assert gate is None
