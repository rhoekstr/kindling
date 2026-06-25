"""Canonical eval-set construction for the verification harness.

The v1↔v2 parity sweep this module used to host was removed with the v1
engine in the production consolidation; only the eval-set builder is
retained — it is what ``bench/verify.py`` and the gap-decomposition
diagnostic use.

Canonical methodology (matches the historical sweeps so absolute NDCG
numbers compare directly): eligible eval users = train_users ∩ test_users,
sorted by ``str(entity_id)``, strided ``eligible[::step][:max_users]``;
k = 10; users with empty held-out sets participate in coverage only.
"""

from __future__ import annotations

import pandas as pd


def _build_eval_set(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_users: int = 500,
    seed: int = 0,
) -> dict[object, set[object]]:
    """Strided eval-set construction (canonical).

    Returns mapping ``entity_id → relevant_set`` where relevant_set is the
    held-out items per user (test − train-owned). Empty held-out sets are
    kept (the NDCG aggregator skips them for accuracy; coverage still
    counts). ``seed`` is informational — the stride is deterministic.
    """
    _ = seed
    train_users_to_items: dict[object, set[object]] = {}
    for u, g in train.groupby("entity_id"):
        train_users_to_items[u] = set(g["item_id"].tolist())
    test_users_to_items: dict[object, set[object]] = {}
    for u, g in test.groupby("entity_id"):
        test_users_to_items[u] = set(g["item_id"].tolist())
    eligible = sorted(
        set(train_users_to_items).intersection(test_users_to_items),
        key=str,
    )
    if not eligible:
        return {}
    step = max(1, len(eligible) // max_users)
    sampled = eligible[::step][:max_users]
    out: dict[object, set[object]] = {}
    for u in sampled:  # type: ignore[assignment]
        held = test_users_to_items[u] - train_users_to_items.get(u, set())
        out[u] = held
    return out
