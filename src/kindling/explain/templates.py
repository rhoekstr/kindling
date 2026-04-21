"""Explanation templates (PRD §8).

Phase 1 ships one default template per signal source. Full template override
and contextual-value override land in a later phase alongside the full signal
stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_DEFAULT_TEMPLATES: dict[str, str] = {
    "cooccurrence": "Often seen with items you've already interacted with.",
}


@dataclass(frozen=True)
class Explanation:
    """Plain-language explanation attached to a recommendation.

    Attributes
    ----------
    primary:
        Single-sentence explanation of the top-contributing signal.
    secondary:
        Optional second sentence when two signals both contributed
        meaningfully. ``None`` in Phase 1 (single-signal pipeline).
    debug_payload:
        Raw signal breakdown for the ``.debug()`` surface (PRD §6.5, §8.4).
        Keyed by signal name.
    """

    primary: str
    secondary: str | None = None
    debug_payload: dict[str, Any] = field(default_factory=dict)

    def debug(self) -> dict[str, Any]:
        return dict(self.debug_payload)


# The PRD calls these "confidence intervals" for familiarity, but they are
# Bayesian credible intervals derived from the posterior over blend weights.
# The plan's honesty commitment renames them to match.


def explain_from_source(source: str, score: float) -> Explanation:
    """Build an explanation for a Phase 1 single-source recommendation."""
    template = _DEFAULT_TEMPLATES.get(source, "Recommended based on your history.")
    return Explanation(
        primary=template,
        debug_payload={"signals": {source: {"score": score, "weight": 1.0}}},
    )
