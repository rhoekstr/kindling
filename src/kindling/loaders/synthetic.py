"""Synthetic session-heavy datasets for benchmark sanity (plan Phase 7).

The plan calls for cross-dataset critical-path benchmarks, but three of
the four advertised datasets (Instacart, Amazon, RetailRocket) require
Kaggle / mirror auth and pre-downloaded files. For offline CI + for
testing whether the path signals actually work when a dataset is
session-heavy, we also ship a fully synthetic generator.

Two generators:

1. ``make_grocery`` - simulates grocery-basket behavior. Each user has
   a preference profile over a fixed set of categories (produce, dairy,
   snacks, ...); sessions sample a basket from 2-3 preferred categories
   with a strong pair-affinity structure (bread + butter, pasta +
   sauce). Path signals SHOULD dominate co-occurrence on this dataset.
2. ``make_ratings`` - simulates ML-1M-style single-event ratings over
   time, with no session structure. Co-occurrence should dominate.

Both produce ``DatasetSplit``s that plug directly into the benchmark
harness so likelihood / temperature comparisons run without external
data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kindling.loaders._base import DatasetSplit


def make_grocery(
    n_entities: int = 200,
    n_items_per_category: int = 15,
    n_categories: int = 6,
    n_sessions_per_entity: int = 8,
    items_per_session: int = 5,
    seed: int = 0,
    test_fraction: float = 0.1,
) -> DatasetSplit:
    """Session-heavy synthetic data with strong pair-affinity structure.

    Produces roughly ``n_entities * n_sessions_per_entity *
    items_per_session`` interactions. Path signals (especially
    path_basket and path_tail) are the dominant predictor by design -
    items within a session come from a small set of correlated
    categories.
    """
    rng = np.random.default_rng(seed)
    total_items = n_categories * n_items_per_category
    categories = np.repeat(np.arange(n_categories), n_items_per_category)

    # Entity preference profile: each entity prefers 2-3 categories
    # heavily, others lightly.
    prefs: list[np.ndarray] = []
    for _ in range(n_entities):
        prof = rng.random(n_categories) * 0.2
        favorites = rng.choice(n_categories, size=rng.integers(2, 4), replace=False)
        prof[favorites] += 1.0
        prefs.append(prof / prof.sum())

    base_time = pd.Timestamp("2024-01-01")
    rows: list[dict[str, object]] = []
    session_counter = 0
    for entity_idx, profile in enumerate(prefs):
        for session_n in range(n_sessions_per_entity):
            # Draw a basket: first two items drive the session theme.
            session_items: list[int] = []
            session_cats = rng.choice(n_categories, size=2, replace=False, p=profile)
            for _ in range(items_per_session):
                cat = rng.choice(session_cats)
                candidates = np.where(categories == cat)[0]
                # Draw uniformly within the category but exclude already-
                # picked items to avoid duplicates in a session.
                available = candidates[~np.isin(candidates, session_items)]
                if available.size == 0:
                    continue
                pick = int(rng.choice(available))
                session_items.append(pick)
            timestamp = base_time + pd.Timedelta(
                days=session_n * 3, minutes=int(rng.integers(0, 1440))
            )
            for order, item in enumerate(session_items):
                rows.append(
                    {
                        "entity_id": entity_idx,
                        "item_id": int(item),
                        "timestamp": timestamp + pd.Timedelta(seconds=order * 10),
                        "session_id": session_counter,
                        "action_type": "add",
                    }
                )
            session_counter += 1

    df = pd.DataFrame(rows).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cutoff = int(len(df) * (1.0 - test_fraction))
    train = df.iloc[:cutoff].copy().reset_index(drop=True)
    test = df.iloc[cutoff:].copy().reset_index(drop=True)

    items = pd.DataFrame(
        {
            "item_id": np.arange(total_items),
            "category": [f"cat_{c}" for c in categories],
        }
    )
    return DatasetSplit(
        name="synthetic-grocery",
        train=train,
        test=test,
        items=items,
        description=(
            f"Synthetic session-heavy dataset: {n_entities} entities, "
            f"{total_items} items across {n_categories} categories, "
            f"{n_sessions_per_entity} sessions/entity of "
            f"{items_per_session} items each. Path signals should "
            "dominate."
        ),
    )


def make_ratings(
    n_entities: int = 200,
    n_items: int = 100,
    ratings_per_entity: int = 30,
    seed: int = 0,
    test_fraction: float = 0.1,
) -> DatasetSplit:
    """Ratings-style synthetic data with no session structure. Co-
    occurrence should dominate path signals."""
    rng = np.random.default_rng(seed)
    # Latent-factor preference model: 5 factors per entity, per item.
    k = 5
    entity_factors = rng.normal(size=(n_entities, k))
    item_factors = rng.normal(size=(n_items, k))
    scores = entity_factors @ item_factors.T

    rows: list[dict[str, object]] = []
    base_time = pd.Timestamp("2024-01-01")
    for entity in range(n_entities):
        top_items = np.argsort(-scores[entity])[:ratings_per_entity]
        rng.shuffle(top_items)
        for item in top_items:
            rows.append(
                {
                    "entity_id": entity,
                    "item_id": int(item),
                    "timestamp": base_time + pd.Timedelta(days=int(rng.integers(0, 365))),
                    "action_type": "add",
                }
            )
    df = pd.DataFrame(rows).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cutoff = int(len(df) * (1.0 - test_fraction))
    return DatasetSplit(
        name="synthetic-ratings",
        train=df.iloc[:cutoff].copy().reset_index(drop=True),
        test=df.iloc[cutoff:].copy().reset_index(drop=True),
        items=None,
        description=(
            f"Synthetic ratings-style dataset: {n_entities} entities, "
            f"{n_items} items, ~{ratings_per_entity} interactions/entity, "
            "no session structure. Co-occurrence should dominate."
        ),
    )
