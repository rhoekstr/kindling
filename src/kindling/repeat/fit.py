"""Build a RepeatProfileTable from timestamped interactions.

Pipeline per item:
1. Collect all inter-interaction intervals (seconds between successive
   interactions for the SAME entity with the SAME item).
2. Count unique users and repeat-interaction users -> repeat_rate.
3. If n_intervals >= min_observations_individual:
     detect_period on the item's own intervals.
4. Else (sparse): pool with K cooc-nearest neighbors' intervals.
5. Shape-classify the (optionally pooled) intervals against prototypes.
6. Compose a RepeatProfile with confidence = fit_quality *
   logistic(n_obs).

Explicit overrides (per-item and per-category) bypass inference entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from kindling.repeat.config import RepeatConfig
from kindling.repeat.period import detect_period
from kindling.repeat.pool import neighbors_by_cooccurrence
from kindling.repeat.profile import Pattern, RepeatProfile, RepeatProfileTable
from kindling.repeat.shape import classify_shape, dominant_pattern

if TYPE_CHECKING:
    from kindling.graph.item_graph import ItemGraph


def _confidence(
    n_obs_intervals: int,
    fit_quality: float,
    n_users: int,
    pattern_4_prob: float,
) -> float:
    """Pattern-aware confidence in [0, 1].

    Distributional patterns (1/2/3) need observed INTERVALS for KDE /
    prototype matching to be reliable. If we have few intervals, we
    trust the classification less.

    Pattern 4 (one-shot) is scored from the REPEAT RATE, which is
    reliable as soon as we've seen enough users (even if none of them
    repeated). A movie that 1000 users rated exactly once is a very
    high-confidence one-shot.

    We blend the two confidences by the pattern-4 probability: mostly-
    pattern-4 items get rate-based confidence, mostly-distributional
    items get interval-based confidence.
    """
    dist_conf = min(1.0, n_obs_intervals / 20.0) * fit_quality
    rate_conf = min(1.0, n_users / 10.0)
    return float(pattern_4_prob * rate_conf + (1.0 - pattern_4_prob) * dist_conf)


def fit_repeat_profiles(
    interactions: pd.DataFrame,
    item_graph: "ItemGraph | None" = None,
    config: RepeatConfig | None = None,
    default_period_seconds: float = 86400.0 * 30.0,
    default_refractory_multiplier: float = 3.0,
) -> RepeatProfileTable:
    """Build a per-item repeat profile table.

    Required input columns: ``entity_id``, ``item_id``, ``timestamp``.
    Timestamps must be convertible to pandas datetime / numpy datetime64.

    ``item_graph`` enables neighbor pooling for sparse items. Omit to
    skip pooling; sparse items then fall through to the default profile.

    ``config`` supplies thresholds, explicit overrides, and category
    mappings.
    """
    cfg = config or RepeatConfig()
    if "timestamp" not in interactions.columns:
        # Without timestamps the full pipeline can't operate. Return
        # an empty table; the engine will treat every item as the
        # (pattern=REPEAT, confidence=0) default -> no adjustment.
        return RepeatProfileTable()

    df = interactions[["entity_id", "item_id", "timestamp"]].copy()
    # Force nanosecond resolution to avoid pandas 2.x us/ms/s unit surprises,
    # then convert to integer seconds since the Unix epoch.
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    df["ts_s"] = df["timestamp"].astype("int64") // 1_000_000_000
    df = df.sort_values(["entity_id", "item_id", "ts_s"], kind="mergesort")

    # Compute per (entity, item) groups: the sorted timestamps.
    # Then the intervals between consecutive timestamps are that
    # entity's inter-interaction delays for that item.
    grouped = df.groupby(["entity_id", "item_id"], sort=False)["ts_s"]

    # Collect per-item interval lists + repeat-user counts.
    per_item_intervals: dict[object, list[float]] = {}
    per_item_n_users: dict[object, int] = {}
    per_item_n_repeat_users: dict[object, int] = {}

    for (_entity, item), ts_series in grouped:
        ts_arr = ts_series.to_numpy()
        n_users = per_item_n_users.get(item, 0) + 1
        per_item_n_users[item] = n_users
        if ts_arr.size >= 2:
            per_item_n_repeat_users[item] = per_item_n_repeat_users.get(item, 0) + 1
            intervals = np.diff(ts_arr).astype(np.float64)
            intervals = intervals[intervals > 0.0]
            if intervals.size > 0:
                per_item_intervals.setdefault(item, []).extend(intervals.tolist())

    # Pre-compute item indices if pooling is enabled.
    item_index: dict[object, int] | None = (
        item_graph.item_index if item_graph is not None else None
    )
    adj = item_graph.adjacency if item_graph is not None else None

    profiles: dict[object, RepeatProfile] = {}
    all_items = list(per_item_n_users.keys())

    for item in all_items:
        # Explicit overrides: skip inference entirely.
        if item in cfg.explicit_overrides:
            profiles[item] = cfg.explicit_overrides[item]
            continue
        cat = cfg.item_to_category.get(item)
        if cat is not None and cat in cfg.category_profiles:
            profiles[item] = cfg.category_profiles[cat]
            continue

        n_users = per_item_n_users.get(item, 0)
        n_repeat_users = per_item_n_repeat_users.get(item, 0)
        repeat_rate = n_repeat_users / max(n_users, 1)

        own_intervals = np.asarray(per_item_intervals.get(item, []), dtype=np.float64)
        pooled = False
        if own_intervals.size < cfg.min_observations_individual and adj is not None and item_index is not None:
            # Pool with K cooc-nearest neighbors' intervals.
            idx = item_index.get(item)
            if idx is not None:
                neighbor_indices = neighbors_by_cooccurrence(adj, idx, cfg.neighbor_pool_k)
                neighbor_item_ids = [item_graph.item_ids[i] for i in neighbor_indices]
                combined = own_intervals.tolist()
                for n_item in neighbor_item_ids:
                    combined.extend(per_item_intervals.get(n_item, []))
                if combined:
                    own_intervals = np.asarray(combined, dtype=np.float64)
                    pooled = True

        if own_intervals.size == 0:
            # No observation at all (only ever single interactions + no
            # neighbor data). Pattern-4 flag via repeat_rate will still
            # work; period defaults.
            period_s = default_period_seconds
            fit_quality = 0.1
        else:
            period_s, fit_quality = detect_period(own_intervals)
            if not np.isfinite(period_s) or period_s <= 0.0:
                period_s = default_period_seconds
                fit_quality = 0.1

        scaled = own_intervals / period_s if own_intervals.size else own_intervals
        probs = classify_shape(
            scaled_intervals=scaled,
            repeat_rate=repeat_rate,
            temperature=cfg.temperature,
            pattern_4_rate_threshold=cfg.pattern_4_rate_threshold,
        )
        pattern = dominant_pattern(probs)
        refractory = period_s * default_refractory_multiplier
        confidence = _confidence(
            n_obs_intervals=own_intervals.size,
            fit_quality=fit_quality,
            n_users=n_users,
            pattern_4_prob=probs[Pattern.ONE_SHOT],
        )

        profiles[item] = RepeatProfile(
            pattern=pattern,
            pattern_probs=probs,
            period_seconds=float(period_s),
            refractory_seconds=float(refractory),
            confidence=float(confidence),
            n_observations=int(own_intervals.size),
            pooled=bool(pooled),
            repeat_rate=float(repeat_rate),
        )

    return RepeatProfileTable(profiles=profiles)
