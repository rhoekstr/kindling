"""LayerPlan: decide which subsystems and boost layers to activate from a profile.

Maps a ``DatasetProfile`` to a concrete plan that drives engine fit:

- which signal subsystems to build (item_graph always; session_cooc /
  temporal_cooc / repeat conditional on data shape)
- which boost layers participate in adaptive layered scoring
- whether the temporal kernel should be on (vs pure-count)
- whether the repeat module should be on
- per-layer activation rationale exposed to the user

The plan is the bridge between "what shape is the data" and "what
should kindling actually do at fit time."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kindling.profile.profile import DatasetProfile


# Signal subsystems we may or may not build during fit.
SubsystemName = Literal[
    "item_graph",            # always
    "session_cooc_graph",    # session-rich
    "temporal_graph",        # timestamps + meaningful kernel
    "repeat_module",         # repeat-dataset
    "path_basket",           # session-aware paths (basket index)
    "path_tail",             # always when timestamps; cheap
    "persona_index",         # caller-controlled
    "lightgcn",              # caller-controlled (very expensive on large data)
    "als",                   # caller-controlled
]

# Boost layers in the adaptive scorer. Always optional.
BoostLayer = Literal[
    "path_basket",
    "session_cooccurrence",
    "temporal_cooccurrence",
]


@dataclass(frozen=True)
class LayerPlan:
    """Decisions about what to build and what to boost with."""

    # Subsystems the engine should build.
    enabled_subsystems: tuple[SubsystemName, ...]
    # Boost layers participating in adaptive scoring at recommend time.
    enabled_boost_layers: tuple[BoostLayer, ...]
    # Whether the temporal kernel is meaningful (gmm) or collapsed
    # (pure_count / rating_burst). Drives temporal_cooccurrence's
    # behavior at scoring time even when the column is built.
    temporal_kernel_active: bool
    # Whether the repeat module should run (period detection + pattern
    # classification + post-rec filter).
    repeat_module_active: bool
    # Per-decision rationale - which heuristic in plan_layers fired.
    rationale: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """One-paragraph human-readable summary."""
        lines = [
            f"subsystems: {', '.join(self.enabled_subsystems)}",
            f"boost_layers: {', '.join(self.enabled_boost_layers) or '<none - cooc-only ranking>'}",
            f"temporal_kernel: {'on' if self.temporal_kernel_active else 'off'}",
            f"repeat_module: {'on' if self.repeat_module_active else 'off'}",
        ]
        if self.rationale:
            lines.append("rationale:")
            for k, v in self.rationale.items():
                lines.append(f"  {k}: {v}")
        return "\n  ".join(lines)


def plan_layers(profile: DatasetProfile) -> LayerPlan:
    """Map a profile to a concrete subsystem + boost-layer plan.

    Decisions are deliberately conservative and explainable - each
    "enable" or "skip" has a one-line rationale. The plan is exposed
    via ``Engine.posterior_summary()`` so users see WHY each decision
    was made.

    Decision tree (in order):

    1. item_graph + path_tail are always built (cheap, always useful).
    2. temporal_graph (substrate for temporal_cooccurrence): built
       whenever timestamps exist OR the dataset is large enough to
       benefit from the per-user-history-cap effect (>=20 events/user
       average).
    3. temporal_kernel_active: only when GMM detected real session
       structure (time_use ∈ {session_consumption, long_horizon}).
       The substrate is still useful with kernel off because the
       per-user history cap acts as implicit recency truncation.
    4. session_cooc_graph: built when sessions are deep enough
       (deep_session_fraction >= 20%) AND not rating-burst.
       Otherwise the graph degenerates.
    5. path_basket: built when session_depth >= "moderate" (median
       session size >= 3). Otherwise basket-level signal is too
       diluted to add value over cooc.
    6. repeat_module: enabled when ``repeat_dataset`` (>=10% of users
       have repeat (user, item) pairs).
    7. Boost layers: each enabled subsystem contributes its
       boost-layer column when the meaningfulness heuristic passes.
    """
    rationale: dict[str, str] = {}
    subsystems: list[SubsystemName] = ["item_graph", "path_tail"]
    boost_layers: list[BoostLayer] = []

    # Temporal substrate.
    use_temporal_graph = (
        profile.has_timestamps
        or profile.avg_events_per_user >= 20  # history-cap effect alone is worth it
    )
    if use_temporal_graph:
        subsystems.append("temporal_graph")
        rationale["temporal_graph"] = (
            "timestamps present"
            if profile.has_timestamps
            else f"avg events/user {profile.avg_events_per_user:.0f} >= 20 - history cap is useful"
        )
    else:
        rationale["temporal_graph"] = "no timestamps and sparse history - skip"

    # Temporal kernel mode.
    temporal_kernel_active = profile.time_use in (
        "session_consumption", "long_horizon"
    )
    rationale["temporal_kernel"] = (
        f"time_use={profile.time_use} -> kernel " +
        ("on" if temporal_kernel_active else "off (pure-count)")
    )

    # Session-aware subsystems.
    has_deep_sessions = (
        profile.session_depth in ("moderate", "deep", "very_deep")
        and profile.time_use != "rating_burst"
    )
    if has_deep_sessions:
        subsystems.append("session_cooc_graph")
        rationale["session_cooc_graph"] = (
            f"session_depth={profile.session_depth}, "
            f"time_use={profile.time_use} -> session structure usable"
        )
        if profile.session_depth in ("moderate", "deep", "very_deep"):
            subsystems.append("path_basket")
            rationale["path_basket"] = (
                f"session_depth={profile.session_depth} - basket signals viable"
            )
    else:
        if profile.time_use == "rating_burst":
            rationale["session_cooc_graph"] = "rating-burst guard - skip"
            rationale["path_basket"] = "rating-burst guard - skip"
        else:
            rationale["session_cooc_graph"] = (
                f"session_depth={profile.session_depth} - too shallow to bother"
            )
            rationale["path_basket"] = (
                f"session_depth={profile.session_depth} - too shallow to bother"
            )

    # Repeat module.
    repeat_module_active = profile.repeat_dataset
    rationale["repeat_module"] = (
        f"repeat_user_fraction={profile.repeat_user_fraction:.0%} "
        + (">= threshold - on" if repeat_module_active else "< 10% threshold - off")
    )
    if repeat_module_active:
        subsystems.append("repeat_module")

    # Boost layers - one per built subsystem that produces a per-candidate
    # signal column suitable for layered scoring.
    if "path_basket" in subsystems:
        boost_layers.append("path_basket")
    if "session_cooc_graph" in subsystems:
        boost_layers.append("session_cooccurrence")
    if "temporal_graph" in subsystems:
        boost_layers.append("temporal_cooccurrence")

    return LayerPlan(
        enabled_subsystems=tuple(subsystems),
        enabled_boost_layers=tuple(boost_layers),
        temporal_kernel_active=temporal_kernel_active,
        repeat_module_active=repeat_module_active,
        rationale=rationale,
    )
