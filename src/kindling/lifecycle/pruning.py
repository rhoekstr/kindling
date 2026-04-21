"""Density-aware pruning of derived structures (PRD §3.5).

Pruning is a complement to decay. Decay is applied at query time: every
stored entry contributes less as it ages, but it keeps contributing.
Pruning is a storage-time operation: entries whose decay-weighted
support falls below a threshold AND are older than a minimum retention
age are removed entirely.

PRD semantics:

    effective_support(entry) = raw_count(entry) * decay(age(entry))

    prune if: effective_support < support_threshold
              AND age > min_retention_age

Phase 6 simplification: kindling's current structures store decay-weighted
counts directly (the decay has already been applied at build time), so
``effective_support ≈ stored_weight``. Pruning drops entries whose stored
weight is below the configured threshold. The ``min_retention_age`` gate
lands in v1.x when per-entry age tracking is added; Phase 6 ships the
support-threshold half and a preserved-aggregate scaffold so posterior
variance accounting remains honest (plan §6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_SUPPORT_THRESHOLD = 0.01
DEFAULT_MIN_RETENTION_DAYS = 30


@dataclass(frozen=True)
class PruningConfig:
    """Knobs for the structure-level pruning operation.

    Attributes
    ----------
    enabled:
        When False, prune methods short-circuit and return zero-prune
        aggregates. Default True.
    schedule:
        ``"retrain"`` (run at Engine.fit / refit - the default and PRD
        recommendation), ``"scheduled"`` (run at a configurable cadence
        via user code), or ``"continuous"`` (run opportunistically during
        writes; reserved for v1.x).
    support_threshold:
        Entries with stored weight below this are candidates for removal.
        Default 0.01; units depend on the structure - for path/tail/
        basket indexes the weights are decay-weighted counts, for the
        item graph they are co-occurrence counts.
    min_retention_days:
        Grace period: entries younger than this are kept regardless of
        support. Phase 6 stub - the pruning functions accept and pass it
        through to the preserved-aggregate record, but the per-entry age
        gate is a v1.x feature that needs per-entry age tracking.
    retention_strategy:
        ``"decay"`` (default - derived from the decay function),
        ``"adaptive"`` (drift-informed, ``see drift.py``), or ``"fixed"``
        (max_age_days cap).
    """

    enabled: bool = True
    schedule: str = "retrain"
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD
    min_retention_days: int = DEFAULT_MIN_RETENTION_DAYS
    retention_strategy: str = "decay"


@dataclass
class PreservedAggregate:
    """Summary of what a prune pass removed. Feeds the posterior variance
    calculation so the Bayesian blend sees the right total data volume
    even when detail has been dropped (PRD §3.5, plan §6).

    Attributes
    ----------
    structure_name:
        Which structure produced this aggregate (``item_graph``,
        ``tail_index``, ``path_tree``, ``basket_index``, ``cost_graph``).
    n_pruned_entries:
        Count of entries removed.
    total_pruned_weight:
        Sum of the (decay-weighted) contributions that were removed.
        When the posterior uses support-based weighting, the pruned
        contribution enters the calibration denominator without adding
        to the numerator - i.e., the posterior knows the items existed
        but doesn't have evidence to update weights from them.
    """

    structure_name: str
    n_pruned_entries: int
    total_pruned_weight: float
    config: PruningConfig = field(default_factory=PruningConfig)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PreservedAggregate({self.structure_name}: "
            f"{self.n_pruned_entries} entries / {self.total_pruned_weight:.3f} weight)"
        )
