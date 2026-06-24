"""Intelligent activation detection — the inspectable record of which
layers the engine turned on for a dataset, and why.

Every channel in the v2 scorer is gated by a measurable property of the
data (catalog size, timestamps, rating signal, rating-burst, history
length). Those gates run at ``fit()`` time and are recorded in the
engine's profile. ``ActivationPlan`` turns that raw profile into a
structured, self-explaining object: the regime that was detected, the
base scorer chosen, and each channel's on/off state with the reason.

The gating is deterministic by design — the experiment record
(``docs/EXPERIMENTS.md`` §4.4/§7.2) showed that *learned* per-dataset
calibration inverts between the internal holdout and the test slice and
does not deploy, while fixed cross-dataset gates transfer. So activation
is a regime classifier, not a learned model — and it can state and
defend its own configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LayerActivation:
    """One scoring layer's activation decision."""

    name: str
    active: bool
    weight: float | None  # blend weight when active (None for structural layers)
    reason: str  # the gate condition + the regime fact that drove it


@dataclass(frozen=True)
class ActivationPlan:
    """What the engine activated for the fitted dataset, and why."""

    # Detected regime
    n_users: int
    n_items: int
    median_history: float
    has_timestamps: bool
    has_sessions: bool
    rating_burst: bool
    rating_weighted: bool
    # Base scorer
    base_scorer: str  # "ease" | "wilson_cooc" | "cooc"
    ease_lambda: float | None
    # Layers + cold-start
    channels: list[LayerActivation]
    cold_start: dict[str, Any] = field(default_factory=dict)

    @property
    def active_channels(self) -> list[str]:
        return [c.name for c in self.channels if c.active]

    def summary(self) -> str:
        lines = [
            f"Regime: {self.n_users:,} users x {self.n_items:,} items, "
            f"median history {self.median_history:.0f}, "
            f"timestamps={self.has_timestamps}, sessions={self.has_sessions}, "
            f"rating_burst={self.rating_burst}",
            f"Base: {self.base_scorer}"
            + (f" (lambda={self.ease_lambda:.0f})" if self.ease_lambda else "")
            + (", rating-weighted" if self.rating_weighted else ""),
            "Channels:",
        ]
        for c in self.channels:
            mark = "ON " if c.active else "off"
            w = f" x{c.weight}" if (c.active and c.weight is not None) else ""
            lines.append(f"  [{mark}] {c.name}{w} — {c.reason}")
        if self.cold_start.get("cold_slots", 0) or self.cold_start.get("open_catalog"):
            lines.append(f"Cold-start: {self.cold_start}")
        return "\n".join(lines)


def build_activation_plan(engine: Any, profile: dict[str, Any]) -> ActivationPlan:
    """Construct the plan from a fitted engine + its profile dict."""
    base_used = profile.get("base_scorer_used", "cooc")
    transform = profile.get("cooc_base_transform", "raw")
    if base_used == "ease":
        base = "ease"
    elif transform and transform != "raw":
        base = f"{transform}_cooc"  # e.g. wilson_cooc
    else:
        base = "cooc"

    has_ts = bool(profile.get("has_timestamps", False))
    burst = bool(profile.get("rating_burst_detected", False))
    med_hist = float(profile.get("median_items_per_user", 0.0))
    gate = getattr(engine, "user_cf_history_gate", 20)
    on_ease = base == "ease"

    channels = [
        LayerActivation(
            "trend",
            has_ts,
            profile.get("trend_alpha"),
            "recent-window popularity; needs timestamps" + ("" if has_ts else " (absent → off)"),
        ),
        LayerActivation(
            "last_item",
            on_ease,
            getattr(engine, "last_item_alpha", 0.25) if on_ease else None,
            "EASE row of the newest item; reads structure not order (not burst-gated)"
            + ("" if on_ease else " — needs EASE base"),
        ),
        LayerActivation(
            "transitions",
            bool(profile.get("transition_channel_active", False)),
            profile.get("transition_alpha") if profile.get("transition_channel_active") else None,
            "directional last-k cooc; needs timestamps AND not rating-burst"
            + (" (burst detected → off)" if burst else ""),
        ),
        LayerActivation(
            "user_cf",
            bool(profile.get("user_cf_channel_active", False)),
            getattr(engine, "user_cf_alpha", 1.0)
            if profile.get("user_cf_channel_active")
            else None,
            f"k-NN user neighbours; sparse-history only (median {med_hist:.0f} "
            f"{'<=' if med_hist <= gate else '>'} gate {gate})",
        ),
        LayerActivation(
            "content",
            bool(profile.get("content_channel_active", False)),
            None,
            "item-metadata cosine, cold-gated; opt-in (default off on warm protocols)",
        ),
    ]

    cold = {
        "open_catalog": bool(getattr(engine, "open_catalog", False)),
        "cold_slots": int(getattr(engine, "cold_slots", 0)),
        "extension_items": int(profile.get("n_extension_items", 0)),
    }

    return ActivationPlan(
        n_users=int(profile.get("n_users", 0)),
        n_items=int(getattr(engine._state, "n_items", 0)),
        median_history=med_hist,
        has_timestamps=has_ts,
        has_sessions=bool(profile.get("has_sessions", False)),
        rating_burst=burst,
        rating_weighted=bool(profile.get("ease_weighted", False)),
        base_scorer=base,
        ease_lambda=profile.get("ease_lambda"),
        channels=channels,
        cold_start=cold,
    )
