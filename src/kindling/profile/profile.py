"""DatasetProfile: auto-detected characteristics of training data.

Single dataclass capturing the size / density / time / sessions /
repeat / rating shape. Used by ``plan_layers`` to decide which
subsystems to build and which boosting layers to activate.

The profile is cheap to compute (small fraction of total fit cost)
and exposed via ``Engine.posterior_summary()`` so users see what
the engine concluded about their data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


# Human-readable density buckets for events per dimension.
DensityBucket = Literal["sparse", "moderate", "dense", "very_dense"]
# Session-depth buckets - drives session_cooccurrence + path_basket activation.
SessionDepth = Literal["none", "shallow", "moderate", "deep", "very_deep"]
# Time-use modes - drives temporal kernel decisions.
TimeUseMode = Literal[
    "no_timestamps",
    "rating_burst",       # GMM midpoint < 5 min - UI click bursts
    "session_consumption",# GMM midpoint 5 min - 24 hr - real shopping/browsing
    "long_horizon",       # midpoint > 24 hr - reviews, long-tail
    "weak_bimodality",    # GMM didn't pass LLR threshold; degenerate timestamps
]


def _bucket_density(events_per_unit: float) -> DensityBucket:
    """Map per-user or per-item event count to density bucket."""
    if events_per_unit < 5:
        return "sparse"
    if events_per_unit < 30:
        return "moderate"
    if events_per_unit < 200:
        return "dense"
    return "very_dense"


def _bucket_session_depth(median_size: float, deep_fraction: float) -> SessionDepth:
    """Map session depth to a bucket. ``median_size`` is items/session;
    ``deep_fraction`` is the fraction of sessions with 2+ items."""
    if deep_fraction < 0.05 or median_size < 1.5:
        return "none"
    if median_size < 3:
        return "shallow"
    if median_size < 6:
        return "moderate"
    if median_size < 15:
        return "deep"
    return "very_deep"


@dataclass(frozen=True)
class DatasetProfile:
    """Auto-detected shape of training data.

    Populated by ``profile_dataset(interactions)`` at engine fit time.
    Drives the layer plan.
    """

    # 1. Size
    n_users: int
    n_items: int
    n_interactions: int

    # User / item density
    avg_events_per_user: float
    avg_events_per_item: float
    user_density: DensityBucket
    item_density: DensityBucket

    # 2. Time
    has_timestamps: bool
    timestamp_span_days: float | None
    inter_event_delta_median_seconds: float | None
    session_strategy: str | None  # "explicit" / "gmm" / "manual_fallback"
    session_gap_seconds: float | None
    session_gof_llr: float | None
    time_use: TimeUseMode

    # 3. Sessions (from session_cooccurrence builder's diagnostics)
    has_explicit_sessions: bool
    n_sessions: int | None
    median_session_size: float | None
    deep_session_fraction: float | None
    session_depth: SessionDepth

    # 4. Repeat schema
    repeat_user_fraction: float    # fraction of users with at least one repeat
    median_repeat_count: float      # avg repeats per (user, item) pair where >1
    repeat_dataset: bool            # is repeat consumption a meaningful signal here?

    # 5. Ratings
    has_ratings: bool
    rating_min: float | None
    rating_max: float | None
    rating_mean: float | None

    # 6. Notes for diagnostics
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable one-paragraph summary of the dataset shape."""
        lines = [
            f"{self.n_interactions:,} interactions  "
            f"{self.n_users:,} users ({self.user_density}, {self.avg_events_per_user:.0f} events/user)  "
            f"{self.n_items:,} items ({self.item_density}, {self.avg_events_per_item:.0f} events/item)",
        ]
        if self.has_timestamps:
            span = f"{self.timestamp_span_days:.0f}d" if self.timestamp_span_days else "?"
            lines.append(
                f"timestamps over {span}  "
                f"time_use={self.time_use}  "
                f"session_gap={self.session_gap_seconds:.0f}s "
                f"(strategy={self.session_strategy}, llr={self.session_gof_llr})"
                if self.session_gap_seconds is not None
                else f"timestamps over {span}  time_use={self.time_use}"
            )
        else:
            lines.append("no timestamps")
        if self.session_depth != "none":
            lines.append(
                f"session_depth={self.session_depth} "
                f"(median {self.median_session_size:.1f} items/session, "
                f"deep_fraction={self.deep_session_fraction:.2f})"
                if self.median_session_size is not None
                else f"session_depth={self.session_depth}"
            )
        if self.repeat_dataset:
            lines.append(
                f"repeat_dataset=true  "
                f"{self.repeat_user_fraction:.0%} of users repeat items  "
                f"median {self.median_repeat_count:.1f} repeats per repeating pair"
            )
        if self.has_ratings:
            lines.append(
                f"ratings range [{self.rating_min:.1f}, {self.rating_max:.1f}] "
                f"mean {self.rating_mean:.2f}"
            )
        if self.notes:
            lines.append("notes: " + "; ".join(self.notes))
        return "\n  ".join(lines)


def profile_dataset(
    interactions: pd.DataFrame,
    session_inference=None,
    kernel_params=None,
    repeat_user_threshold: float = 0.10,
    deep_session_min: float = 0.20,
) -> DatasetProfile:
    """Compute the dataset profile from validated interactions + optional
    pre-computed session / kernel info.

    Parameters
    ----------
    interactions:
        Preprocessed/validated interactions DataFrame.
    session_inference:
        Pre-computed ``SessionInference`` from ``ingest.sessions``.
        When None, only the explicit-session column is read.
    kernel_params:
        Pre-computed ``KernelParams`` from
        ``graph.temporal_interaction.calibrate_kernel``. Used to
        classify time_use.
    repeat_user_threshold:
        Repeat-dataset gate: if more than this fraction of users have
        any repeat (user, item) pair, the dataset is flagged for the
        repeat module.
    deep_session_min:
        Minimum fraction of sessions with 2+ items required to
        classify session_depth above "none".
    """
    notes: list[str] = []

    # --- Size ---
    n_interactions = len(interactions)
    n_users = int(interactions["entity_id"].nunique()) if n_interactions else 0
    n_items = int(interactions["item_id"].nunique()) if n_interactions else 0

    avg_events_per_user = (n_interactions / n_users) if n_users else 0.0
    avg_events_per_item = (n_interactions / n_items) if n_items else 0.0

    user_density = _bucket_density(avg_events_per_user)
    item_density = _bucket_density(avg_events_per_item)

    # --- Time ---
    has_timestamps = "timestamp" in interactions.columns and n_interactions > 0
    timestamp_span_days: float | None = None
    inter_event_delta_median_seconds: float | None = None
    session_strategy: str | None = None
    session_gap_seconds: float | None = None
    session_gof_llr: float | None = None
    time_use: TimeUseMode = "no_timestamps"
    has_explicit_sessions = "session_id" in interactions.columns

    if has_timestamps:
        ts = pd.to_datetime(interactions["timestamp"], errors="coerce").dropna()
        if len(ts) >= 2:
            timestamp_span_days = float((ts.max() - ts.min()).total_seconds() / 86400.0)
        # Inter-event median for diagnostics.
        sorted_df = interactions.sort_values(["entity_id", "timestamp"], kind="mergesort")
        ts_sec = (sorted_df["timestamp"].astype("int64").to_numpy() // 10**9).astype(np.float64)
        ents = sorted_df["entity_id"].to_numpy()
        same = np.zeros(len(sorted_df), dtype=bool)
        same[1:] = ents[1:] == ents[:-1]
        deltas = np.zeros_like(ts_sec)
        deltas[1:] = np.where(same[1:], ts_sec[1:] - ts_sec[:-1], 0.0)
        deltas = deltas[same & (deltas > 0)]
        if deltas.size:
            inter_event_delta_median_seconds = float(np.median(deltas))

    # Pull session info from the inference object when provided.
    if session_inference is not None:
        session_strategy = getattr(session_inference, "strategy", None)
        session_gap_seconds = getattr(session_inference, "gap_threshold_seconds", None)
        session_gof_llr = getattr(session_inference, "gof_log_likelihood_ratio", None)

    # Map kernel calibration to time_use bucket.
    if kernel_params is not None:
        strategy = getattr(kernel_params, "strategy", "")
        midpoint = getattr(kernel_params, "midpoint_seconds", 0.0) or 0.0
        if strategy == "rating_burst_detected":
            time_use = "rating_burst"
        elif strategy == "pure_count":
            time_use = "no_timestamps" if not has_timestamps else "weak_bimodality"
        elif strategy == "gmm" or strategy == "manual_fallback":
            if midpoint < 86400:
                time_use = "session_consumption"
            else:
                time_use = "long_horizon"
        else:
            time_use = "weak_bimodality"
    elif has_explicit_sessions:
        time_use = "session_consumption"  # treat explicit as real
    else:
        time_use = "no_timestamps" if not has_timestamps else "weak_bimodality"

    # --- Session diagnostics (from explicit or implicit session_id) ---
    n_sessions: int | None = None
    median_session_size: float | None = None
    deep_session_fraction: float | None = None
    session_depth: SessionDepth = "none"

    session_col: pd.Series | None = None
    if has_explicit_sessions:
        session_col = interactions["session_id"]
    elif session_inference is not None:
        sids = getattr(session_inference, "session_ids", None)
        if sids is not None and len(sids) == n_interactions:
            session_col = pd.Series(sids, index=interactions.index)

    if session_col is not None and len(session_col) == n_interactions:
        sizes = session_col.groupby(session_col).size()
        n_sessions = int(len(sizes))
        if n_sessions > 0:
            median_session_size = float(sizes.median())
            deep_session_fraction = float((sizes >= 2).mean())
            session_depth = _bucket_session_depth(median_session_size, deep_session_fraction)

    # --- Repeat schema ---
    if n_interactions > 0:
        pair_counts = interactions.groupby(
            ["entity_id", "item_id"], sort=False
        ).size()
        repeating_pairs = pair_counts[pair_counts > 1]
        users_with_repeats = (
            interactions.loc[
                interactions.set_index(["entity_id", "item_id"]).index.isin(
                    repeating_pairs.index
                ),
                "entity_id",
            ].nunique()
            if len(repeating_pairs) > 0 and n_users > 0
            else 0
        )
        repeat_user_fraction = users_with_repeats / max(n_users, 1)
        median_repeat_count = (
            float(repeating_pairs.median()) if len(repeating_pairs) > 0 else 1.0
        )
    else:
        repeat_user_fraction = 0.0
        median_repeat_count = 1.0
    repeat_dataset = repeat_user_fraction >= repeat_user_threshold

    # --- Ratings ---
    has_ratings = "rating" in interactions.columns and n_interactions > 0
    rating_min: float | None = None
    rating_max: float | None = None
    rating_mean: float | None = None
    if has_ratings:
        ratings = pd.to_numeric(interactions["rating"], errors="coerce").dropna()
        if len(ratings):
            rating_min = float(ratings.min())
            rating_max = float(ratings.max())
            rating_mean = float(ratings.mean())

    # Notes
    if user_density == "sparse":
        notes.append("sparse user histories - rec quality bounded by signal density")
    if item_density == "sparse":
        notes.append("sparse item interactions - cold-start gauntlet")
    if time_use == "rating_burst":
        notes.append("timestamps reflect UI rating bursts, not real consumption")
    if session_depth == "none" and has_timestamps:
        notes.append("session structure too shallow for basket/session signals")
    if repeat_dataset:
        notes.append("repeat-friendly: repeat module expected to add value")

    return DatasetProfile(
        n_users=n_users,
        n_items=n_items,
        n_interactions=n_interactions,
        avg_events_per_user=avg_events_per_user,
        avg_events_per_item=avg_events_per_item,
        user_density=user_density,
        item_density=item_density,
        has_timestamps=has_timestamps,
        timestamp_span_days=timestamp_span_days,
        inter_event_delta_median_seconds=inter_event_delta_median_seconds,
        session_strategy=session_strategy,
        session_gap_seconds=session_gap_seconds,
        session_gof_llr=session_gof_llr,
        time_use=time_use,
        has_explicit_sessions=has_explicit_sessions,
        n_sessions=n_sessions,
        median_session_size=median_session_size,
        deep_session_fraction=deep_session_fraction,
        session_depth=session_depth,
        repeat_user_fraction=repeat_user_fraction,
        median_repeat_count=median_repeat_count,
        repeat_dataset=repeat_dataset,
        has_ratings=has_ratings,
        rating_min=rating_min,
        rating_max=rating_max,
        rating_mean=rating_mean,
        notes=notes,
    )
