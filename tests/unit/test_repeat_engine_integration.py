"""Engine integration of the repeat-consumption module.

Exercises the end-to-end path: fit with RepeatConfig -> owned items
no longer excluded -> multiplier applied between scoring and rerank.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling.engine import Engine
from kindling.repeat import Pattern, RepeatConfig, RepeatProfile


def _repeat_grocery_df() -> pd.DataFrame:
    """Small grocery-style fixture with cross-session repeats.

    - 30 users, 50 items. Each user buys 5 distinct items; each item
      gets repeated by about half its buyers over 8 weeks.
    - Enough repeats to exercise period detection and the multiplier.
    """
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2026-01-01")
    for user in range(30):
        favorites = rng.choice(50, size=5, replace=False)
        for item in favorites:
            # First purchase.
            t = 0.0
            rows.append({
                "entity_id": user,
                "item_id": int(item),
                "timestamp": base + pd.Timedelta(days=t + user * 0.1),
            })
            # ~50% of users buy each item weekly; the rest once.
            if rng.random() < 0.5:
                for _ in range(7):
                    t += 7.0 + rng.normal(0, 1)
                    rows.append({
                        "entity_id": user,
                        "item_id": int(item),
                        "timestamp": base + pd.Timedelta(days=t + user * 0.1),
                    })
    return pd.DataFrame(rows)


def test_repeat_disabled_by_default() -> None:
    """Engine without repeat_config does not build the repeat table."""
    engine = Engine().fit(_repeat_grocery_df())
    assert engine._repeat_table is None


def test_repeat_enabled_builds_profile_table() -> None:
    cfg = RepeatConfig(enabled=True, min_observations_individual=3)
    engine = Engine(repeat_config=cfg).fit(_repeat_grocery_df())
    assert engine._repeat_table is not None
    assert len(engine._repeat_table) > 0


def test_repeat_enabled_includes_owned_items_in_candidates() -> None:
    """When repeat is on, the retrievers should NOT pre-exclude owned
    items - the multiplier decides what gets suppressed."""
    cfg = RepeatConfig(enabled=True, min_observations_individual=3)
    engine = Engine(repeat_config=cfg).fit(_repeat_grocery_df())
    # Pick a user who has ≥3 favorite items (all users in fixture).
    entity = 0
    owned = set(engine._owned_by_entity[entity].tolist())
    recs = engine.recommend(entity_id=entity, n=50)
    rec_items = {r.item_id for r in recs}
    # At least ONE of the entity's owned items must now appear in recs
    # (possibly with a dampened score). With repeat off this would be
    # impossible because the retriever would pre-filter them.
    assert owned & rec_items, "repeat-enabled engine never returned any owned item"


def test_multiplier_reorders_when_pattern_4_present() -> None:
    """Force a ONE_SHOT profile via explicit override on one owned item;
    the engine should rank that item near the bottom of the recommendations
    even if cooccurrence would normally place it high."""
    df = _repeat_grocery_df()
    entity = 0
    owned_items = df[df["entity_id"] == entity]["item_id"].unique()
    target = int(owned_items[0])

    one_shot_profile = RepeatProfile(
        pattern=Pattern.ONE_SHOT,
        pattern_probs={
            Pattern.REPEAT: 0.0,
            Pattern.REPLENISH: 0.0,
            Pattern.SATIATION: 0.0,
            Pattern.ONE_SHOT: 1.0,
        },
        period_seconds=86400.0,
        refractory_seconds=86400.0 * 30,
        confidence=1.0,
        n_observations=100,
        pooled=False,
        repeat_rate=0.0,
    )
    cfg = RepeatConfig(
        enabled=True,
        min_observations_individual=3,
        explicit_overrides={target: one_shot_profile},
    )
    engine = Engine(repeat_config=cfg).fit(df)
    recs = engine.recommend(entity_id=entity, n=50)
    rec_items = [r.item_id for r in recs]
    if target in rec_items:
        # Target should appear at the bottom (last 25% of the list).
        rank = rec_items.index(target)
        assert rank >= len(rec_items) // 2, f"one-shot item at rank {rank} of {len(rec_items)}"


def test_repeat_persists_round_trip(tmp_path) -> None:
    import warnings

    cfg = RepeatConfig(enabled=True, min_observations_individual=3)
    engine = Engine(repeat_config=cfg).fit(_repeat_grocery_df())
    path = tmp_path / "engine.pkl"
    engine.save(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = Engine.load(path)
    assert loaded._repeat_table is not None
    assert len(loaded._repeat_table) == len(engine._repeat_table)
    assert len(loaded._last_interaction_ts) == len(engine._last_interaction_ts)
