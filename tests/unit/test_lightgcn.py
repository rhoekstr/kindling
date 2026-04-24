"""Pure-numpy LightGCN (graph-smoothed BPR-trained embeddings).

No PyTorch dependency. Two-stage: BPR-train base embeddings via SGD,
then propagate K layers at inference. Tests the training loop runs,
scoring is bounded, and taste-group structure is recovered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kindling.graph.lightgcn import (
    LightGCNConfig,
    LightGCNModel,
    build_lightgcn,
)


def _taste_group_interactions(n_users_per_group: int = 40, seed: int = 0) -> pd.DataFrame:
    """Two distinct taste groups. Each user in a group interacts with their
    group's signature items + a few cross-group stragglers. LightGCN
    should recover that structure and score group-A items higher for
    group-A users."""
    rng = np.random.default_rng(seed)
    group_a_items = list(range(10))     # 0-9
    group_b_items = list(range(10, 20)) # 10-19
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2026-01-01")
    uid = 0
    for group, items in [(0, group_a_items), (1, group_b_items)]:
        for u in range(n_users_per_group):
            picks = rng.choice(items, size=5, replace=False)
            for i, pick in enumerate(picks):
                rows.append({
                    "entity_id": uid,
                    "item_id": int(pick),
                    "timestamp": base + pd.Timedelta(minutes=i),
                })
            uid += 1
    return pd.DataFrame(rows)


def test_lightgcn_fits_and_scores() -> None:
    df = _taste_group_interactions()
    item_index = {i: i for i in range(20)}
    cfg = LightGCNConfig(dim=16, n_epochs=10, batch_size=256, min_users=10, min_items=5, seed=0)
    model = build_lightgcn(df, item_graph_item_index=item_index, config=cfg)
    assert model is not None
    assert model.entity_factors.shape == (80, 16)
    assert model.item_factors.shape == (20, 16)
    assert model.n_epochs_trained == 10
    assert np.isfinite(model.entity_factors).all()
    assert np.isfinite(model.item_factors).all()


def test_lightgcn_recovers_taste_group_structure() -> None:
    """User in group A should score group-A items higher than group-B."""
    df = _taste_group_interactions()
    item_index = {i: i for i in range(20)}
    cfg = LightGCNConfig(dim=32, n_epochs=30, batch_size=128, min_users=10, min_items=5, seed=0)
    model = build_lightgcn(df, item_index, cfg)
    assert model is not None
    # User 0 is in group A.
    a_items = np.arange(10, dtype=np.int64)
    b_items = np.arange(10, 20, dtype=np.int64)
    a_scores = model.score_many(0, a_items)
    b_scores = model.score_many(0, b_items)
    # Score is normalized per query; compare means to see if group A is
    # systematically ranked higher for a group-A user.
    assert a_scores.mean() > b_scores.mean(), (
        f"group-A user's group-A score {a_scores.mean():.3f} not higher than "
        f"group-B {b_scores.mean():.3f}"
    )


def test_lightgcn_returns_none_on_too_small_dataset() -> None:
    df = pd.DataFrame({"entity_id": [1, 2], "item_id": [1, 2]})
    cfg = LightGCNConfig(min_users=50)
    assert build_lightgcn(df, item_graph_item_index={1: 0, 2: 1}, config=cfg) is None


def test_lightgcn_scores_unknown_entity_to_zero() -> None:
    df = _taste_group_interactions()
    item_index = {i: i for i in range(20)}
    cfg = LightGCNConfig(dim=16, n_epochs=5, batch_size=128, min_users=10, min_items=5, seed=0)
    model = build_lightgcn(df, item_index, cfg)
    assert model is not None
    scores = model.score_many("unknown_entity", np.array([0, 1, 2]))
    assert (scores == 0).all()
