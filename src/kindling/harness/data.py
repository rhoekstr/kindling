"""Input handling for the eval harness.

Accepts either a *built-in reference dataset* (delegated to
``kindling.benchmarks``) or a *user-supplied interaction log* (CSV). A raw
log is split chronologically — the realistic tier never shuffles time — so
the evaluation mirrors how the model would actually be deployed: fit on the
past, score the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Columns kindling's contract understands. ``entity_id``/``item_id`` are
# required; ``timestamp`` activates the time-aware channels; ``rating``
# engages rating-weighted EASE when the values are true ratings.
_REQUIRED = ("entity_id", "item_id")
_OPTIONAL = ("timestamp", "rating", "weight")

# Common column aliases → canonical names, so a typical export Just Works.
_ALIASES = {
    "user_id": "entity_id",
    "user": "entity_id",
    "userid": "entity_id",
    "customer_id": "entity_id",
    "session_id": "entity_id",
    "product_id": "item_id",
    "item": "item_id",
    "itemid": "item_id",
    "article_id": "item_id",
    "sku": "item_id",
    "ts": "timestamp",
    "time": "timestamp",
    "date": "timestamp",
    "event_time": "timestamp",
    "score": "rating",
    "stars": "rating",
}


@dataclass(frozen=True)
class HarnessSplit:
    """A train/test split plus optional item metadata, harness-internal."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    items: pd.DataFrame | None


def read_csv_aliased(path: str | Path) -> pd.DataFrame:
    """Read a CSV and normalize its columns via :data:`_ALIASES` (no validation)."""
    df = pd.read_csv(path)
    return df.rename(columns={c: _ALIASES.get(str(c).lower(), str(c)) for c in df.columns})


def load_interactions_csv(path: str | Path) -> pd.DataFrame:
    """Load an interaction log from CSV, normalizing column names.

    Recognizes common aliases (``user_id`` → ``entity_id``,
    ``product_id`` → ``item_id``, ``ts``/``date`` → ``timestamp`` …) and
    parses ``timestamp`` to datetime when present. Raises a clear error if
    the required entity/item columns are absent after aliasing.
    """
    df = read_csv_aliased(path)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s) {missing}. "
            f"Need entity_id + item_id (aliases accepted: "
            f"{sorted(set(_ALIASES) | set(_REQUIRED))}). Found: {list(df.columns)}"
        )
    keep = [c for c in (*_REQUIRED, *_OPTIONAL) if c in df.columns]
    df = df[keep].copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def chronological_split(
    interactions: pd.DataFrame, test_fraction: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: the latest ``test_fraction`` of events become the test set.

    Falls back to a deterministic row-order tail split when there is no
    ``timestamp`` column (the only non-temporal option that stays leakage-free
    for a single-pass log).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if "timestamp" in interactions.columns and interactions["timestamp"].notna().any():
        ordered = interactions.sort_values("timestamp", kind="stable")
        cut = ordered["timestamp"].quantile(1.0 - test_fraction)
        train = ordered[ordered["timestamp"] <= cut].copy()
        test = ordered[ordered["timestamp"] > cut].copy()
        if len(test) == 0 or len(train) == 0:  # degenerate (ties at the cut)
            n_test = max(1, int(len(ordered) * test_fraction))
            train, test = ordered.iloc[:-n_test].copy(), ordered.iloc[-n_test:].copy()
        return train, test
    n_test = max(1, int(len(interactions) * test_fraction))
    return interactions.iloc[:-n_test].copy(), interactions.iloc[-n_test:].copy()


def resolve_dataset(
    source: str | Path,
    *,
    test_fraction: float = 0.1,
    metadata: str | Path | None = None,
) -> HarnessSplit:
    """Resolve ``source`` to a train/test split.

    ``source`` is either a *built-in dataset name* (e.g. ``movielens-1m``,
    ``synthetic-grocery`` — anything the benchmark registry knows) or a *path
    to a CSV* interaction log. A CSV is split chronologically here; a built-in
    name arrives pre-split from its loader. ``metadata`` (CSV path) is only
    consulted for the CSV branch and feeds the cold-slot / open-catalog path.
    """
    src = str(source)
    is_path = Path(src).exists() and Path(src).is_file()
    if not is_path:
        # Built-in reference dataset — delegate to the benchmark registry.
        try:
            from kindling.benchmarks.comparison import _load_dataset
        except ImportError as exc:  # pragma: no cover - benchmarks always ship
            raise ValueError(
                f"{src!r} is not a file and the benchmark loaders are unavailable ({exc})."
            ) from exc
        try:
            split = _load_dataset(src, test_fraction=test_fraction)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Unknown dataset {src!r}. Pass a CSV path, or a built-in name "
                f"such as 'synthetic-grocery' / 'movielens-1m'. ({exc})"
            ) from exc
        return HarnessSplit(name=split.name, train=split.train, test=split.test, items=split.items)

    interactions = load_interactions_csv(src)
    train, test = chronological_split(interactions, test_fraction)
    items = read_csv_aliased(metadata) if metadata is not None else None
    return HarnessSplit(name=Path(src).stem, train=train, test=test, items=items)
