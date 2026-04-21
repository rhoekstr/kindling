"""Outcome reporting and logging (PRD §6.6).

Phase 5 ships:
- ``OutcomeLog``: SQLite-backed append-only store for both precise and
  simple outcome reports. Supports dedup via
  ``(entity_id, recommendation_id, item_id)`` and corrections that
  supersede prior rows with the same key.
- ``replay_to_batch``: reconstructs an ``OutcomeBatch`` from the log for
  Bayesian posterior refits.
"""

from kindling.outcomes.log import OutcomeLog, OutcomeRecord, ReportingMode
from kindling.outcomes.replay import replay_to_batch

__all__ = [
    "OutcomeLog",
    "OutcomeRecord",
    "ReportingMode",
    "replay_to_batch",
]
