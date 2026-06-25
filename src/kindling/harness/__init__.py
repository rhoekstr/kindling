"""kindling harness — reusable evaluation and serving.

Two production tools that ship inside the wheel:

* :mod:`kindling.harness.eval` — point the realistic-tier benchmark at your
  own interaction log (or a built-in reference dataset): fit → eval-by-warmth
  → report, with popularity / item-kNN / ALS / BPR baselines built in. The
  same protocol the project validates itself with, packaged for reuse.
* :mod:`kindling.serving` — load a saved engine and serve ``recommend`` over a
  small HTTP surface (FastAPI), with a batch endpoint and the new-user
  ``recommend_for_items`` path.

Both are reachable from the ``kindling`` console command (``kindling bench``,
``kindling serve``); see :mod:`kindling.cli`.
"""

from __future__ import annotations

from kindling.harness.data import chronological_split, load_interactions_csv, resolve_dataset
from kindling.harness.eval import (
    DEFAULT_BUCKETS,
    BucketResult,
    EvalReport,
    evaluate,
    format_report,
)

__all__ = [
    "DEFAULT_BUCKETS",
    "BucketResult",
    "EvalReport",
    "chronological_split",
    "evaluate",
    "format_report",
    "load_interactions_csv",
    "resolve_dataset",
]
