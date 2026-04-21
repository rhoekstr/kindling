"""Build an OutcomeBatch for posterior fitting from training interactions.

Phase 5 adds the real outcome-reporting API (precise + simple). For Phase 3
we need SOME way to produce outcomes so the Bayesian blend has data to
fit. The approach: use the chronological tail 10% of training interactions
as "simulated outcomes" - each held-out interaction is treated as a
"selected" event, and a small sample of co-occurrence-close candidates is
treated as "shown but not selected". This is the leave-last-out
evaluation protocol, borrowed as a training signal.

This is a stand-in; Phase 5's real outcome log supersedes it. Documented
here so the intent is clear when Phase 5 lands.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kindling.blend.heuristic import SignalFeatures
from kindling.blend.likelihoods import OutcomeBatch

SignalBuilder = Callable[[object, list[object], np.ndarray], SignalFeatures]

DEFAULT_NEGATIVES_PER_POSITIVE = 4
DEFAULT_MAX_OUTCOMES = 2000


@dataclass(frozen=True)
class OutcomeBuildConfig:
    """Knobs for outcome construction from the chronological tail."""

    tail_fraction: float = 0.1
    negatives_per_positive: int = DEFAULT_NEGATIVES_PER_POSITIVE
    max_outcomes: int = DEFAULT_MAX_OUTCOMES


def build_outcomes(
    interactions: pd.DataFrame,
    compute_signals: SignalBuilder,
    config: OutcomeBuildConfig | None = None,
    rng: np.random.Generator | None = None,
) -> OutcomeBatch:
    """Construct an ``OutcomeBatch`` from the chronological tail.

    ``compute_signals(entity_id, candidate_items)`` returns a
    ``SignalFeatures`` matrix for a list of candidates. The engine owns
    this; outcome_builder stays decoupled from the Engine's internals.
    """
    cfg = config if config is not None else OutcomeBuildConfig()
    rng = rng if rng is not None else np.random.default_rng(seed=0)

    if "timestamp" in interactions.columns:
        sorted_df = interactions.sort_values("timestamp", kind="mergesort")
    else:
        sorted_df = interactions
    n = len(sorted_df)
    cutoff = int(n * (1.0 - cfg.tail_fraction))
    tail = sorted_df.iloc[cutoff:]
    head = sorted_df.iloc[:cutoff]

    # Prior-owned items per entity come from the head.
    owned_by_entity: dict[object, set[object]] = {}
    for entity, group in head.groupby("entity_id", sort=False):
        owned_by_entity[entity] = set(group["item_id"].tolist())

    # Catalog of all items ever seen - candidate pool for negatives.
    all_items = np.sort(interactions["item_id"].unique())

    list_id_counter = 0
    rows_signal: list[np.ndarray] = []
    rows_selected: list[np.ndarray] = []
    rows_positions: list[np.ndarray] = []
    rows_list_ids: list[np.ndarray] = []
    signal_names: tuple[str, ...] | None = None
    n_outcomes_so_far = 0

    tail_sample = tail.sample(
        n=min(len(tail), cfg.max_outcomes),
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    for _, event in tail_sample.iterrows():
        entity = event["entity_id"]
        positive_item = event["item_id"]
        owned = owned_by_entity.get(entity, set())
        if positive_item in owned or len(owned) == 0:
            continue

        # Sample negatives uniformly from catalog minus owned+positive.
        forbidden = owned | {positive_item}
        forbidden_arr = np.array(list(forbidden), dtype=all_items.dtype)
        candidates = all_items[~np.isin(all_items, forbidden_arr)]
        if candidates.size == 0:
            continue
        n_neg = min(cfg.negatives_per_positive, candidates.size)
        neg_sample = rng.choice(candidates, size=n_neg, replace=False)

        items_in_list = [positive_item, *neg_sample.tolist()]
        owned_arr = np.array(list(owned), dtype=all_items.dtype)
        owned_arr.sort()
        features = compute_signals(entity, items_in_list, owned_arr)
        if signal_names is None:
            signal_names = features.signal_names
        # Positions 1..len in list order; positive is at position 1.
        positions = np.arange(1, len(items_in_list) + 1, dtype=np.int64)
        selected = np.zeros(len(items_in_list), dtype=np.int64)
        selected[0] = 1
        list_ids = np.full(len(items_in_list), list_id_counter, dtype=np.int64)
        list_id_counter += 1

        rows_signal.append(features.matrix)
        rows_selected.append(selected)
        rows_positions.append(positions)
        rows_list_ids.append(list_ids)
        n_outcomes_so_far += len(items_in_list)
        if n_outcomes_so_far >= cfg.max_outcomes:
            break

    if not rows_signal:
        return OutcomeBatch(
            signal_matrix=np.zeros((0, 0)),
            selected=np.zeros(0, dtype=np.int64),
            positions=np.zeros(0, dtype=np.int64),
            list_ids=np.zeros(0, dtype=np.int64),
        )

    return OutcomeBatch(
        signal_matrix=np.vstack(rows_signal),
        selected=np.concatenate(rows_selected),
        positions=np.concatenate(rows_positions),
        list_ids=np.concatenate(rows_list_ids),
    )
