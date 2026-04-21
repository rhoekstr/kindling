"""SQLite-backed outcome log (PRD §6.6).

The outcome log is kindling's working record of user feedback. The user
retains their own source of truth separately; the log is what drives
posterior refits.

Two reporting modes:

1. **Precise**: the caller knows which items were shown at which position,
   which were selected, and which were rejected. Populates all fields.
2. **Simple**: the caller only reports observed interactions, not
   impressions. Populates entity/item/action/rating/timestamp; position
   defaults to 1, shown defaults to True. Calibration quality degrades
   because position-bias correction can't apply; the engine surfaces a
   warning via ``posterior_summary()``.

Plan-closed gap: dedup via ``(entity_id, recommendation_id, item_id)``
primary key; corrections supersede prior rows with the same key through
``report_outcome_correction``. Late-arriving outcomes are accepted and
reverse the timeout assumption at the next posterior refit.
"""

from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# SQLite timestamp precision is microsecond; we store ISO 8601 strings for
# portability across sqlite versions.
_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S.%f"


class ReportingMode(StrEnum):
    """Which reporting API produced this row."""

    PRECISE = "precise"
    SIMPLE = "simple"


@dataclass(frozen=True)
class OutcomeRecord:
    """One outcome observation.

    Attributes
    ----------
    entity_id:
        The entity the recommendation was for.
    recommendation_id:
        Unique id for the recommendation list this outcome belongs to.
        For simple reporting (no list context), empty string.
    item_id:
        The item this observation is about.
    shown:
        True if the item was actually displayed to the entity.
    selected:
        True if the entity selected the item.
    rejected:
        True if the entity explicitly rejected the item.
    rating:
        Optional numeric rating when available.
    position:
        1-indexed display position. 0 if not known.
    reporting_mode:
        Which API produced this row.
    timestamp:
        When the event was observed.
    """

    entity_id: object
    recommendation_id: str
    item_id: object
    shown: bool
    selected: bool
    rejected: bool
    rating: float | None
    position: int
    reporting_mode: ReportingMode
    timestamp: datetime


class OutcomeLog:
    """Persistent outcome store backed by SQLite.

    Use ``":memory:"`` for an ephemeral in-process log (tests); use a file
    path for durable storage. The schema version is tracked in a
    ``_metadata`` table so future migrations can be safe.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path) if path != ":memory:" else path
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._initialize_schema()

    def __del__(self) -> None:
        # Ensure the SQLite connection closes cleanly on GC, otherwise
        # pytest's unraisable-exception hook escalates the ResourceWarning
        # to an error under our strict filterwarnings setting.
        conn = getattr(self, "_conn", None)
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()

    # ---- lifecycle --------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> OutcomeLog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS _metadata (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS outcomes (
                entity_id         TEXT NOT NULL,
                recommendation_id TEXT NOT NULL,
                item_id           TEXT NOT NULL,
                shown             INTEGER NOT NULL,
                selected          INTEGER NOT NULL,
                rejected          INTEGER NOT NULL,
                rating            REAL,
                position          INTEGER NOT NULL,
                reporting_mode    TEXT NOT NULL,
                timestamp         TEXT NOT NULL,
                inserted_at       TEXT NOT NULL,
                PRIMARY KEY (entity_id, recommendation_id, item_id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_entity ON outcomes(entity_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_mode ON outcomes(reporting_mode)")
        cur.execute(
            "INSERT OR IGNORE INTO _metadata (key, value) VALUES (?, ?)",
            ("schema_version", str(self.SCHEMA_VERSION)),
        )

    # ---- writes -----------------------------------------------------------

    def report_precise(
        self,
        *,
        entity_id: object,
        recommendation_id: str,
        shown_items: list[object],
        selected_items: set[object] | list[object] | None = None,
        rejected_items: set[object] | list[object] | None = None,
        positions: list[int] | None = None,
        timestamp: datetime | None = None,
    ) -> int:
        """Log a precise-mode report. Each shown item becomes one row.

        Returns the number of rows inserted (duplicates dedup out).
        """
        if timestamp is None:
            timestamp = _now()
        selected = set(selected_items or ())
        rejected = set(rejected_items or ())
        if positions is None:
            positions = list(range(1, len(shown_items) + 1))
        if len(positions) != len(shown_items):
            raise ValueError(
                f"positions length {len(positions)} != shown_items length {len(shown_items)}"
            )
        rows: list[tuple[object, ...]] = []
        for item, pos in zip(shown_items, positions, strict=True):
            rows.append(
                (
                    str(entity_id),
                    recommendation_id,
                    str(item),
                    1,  # shown
                    1 if item in selected else 0,
                    1 if item in rejected else 0,
                    None,
                    int(pos),
                    ReportingMode.PRECISE.value,
                    timestamp.strftime(_TIMESTAMP_FMT),
                    _now().strftime(_TIMESTAMP_FMT),
                )
            )
        return self._insert(rows)

    def report_simple(
        self,
        *,
        entity_id: object,
        item_id: object,
        action: str,
        rating: float | None = None,
        timestamp: datetime | None = None,
    ) -> int:
        """Log a simple-mode report (interaction-only).

        ``action`` is one of ``"selected"``, ``"rejected"``, ``"rated"``.
        A synthetic ``recommendation_id`` is generated so the primary key
        stays unique per event. Users who want to overwrite an earlier
        row should call ``report_correction`` instead.
        """
        if timestamp is None:
            timestamp = _now()
        if action not in {"selected", "rejected", "rated"}:
            raise ValueError(f"action must be one of selected/rejected/rated, got {action!r}")
        rec_id = f"simple:{timestamp.strftime(_TIMESTAMP_FMT)}"
        rows = [
            (
                str(entity_id),
                rec_id,
                str(item_id),
                1,
                1 if action == "selected" else 0,
                1 if action == "rejected" else 0,
                rating,
                1,
                ReportingMode.SIMPLE.value,
                timestamp.strftime(_TIMESTAMP_FMT),
                _now().strftime(_TIMESTAMP_FMT),
            )
        ]
        return self._insert(rows)

    def report_correction(
        self,
        *,
        entity_id: object,
        recommendation_id: str,
        item_id: object,
        shown: bool,
        selected: bool,
        rejected: bool = False,
        rating: float | None = None,
        position: int = 1,
        timestamp: datetime | None = None,
    ) -> None:
        """Supersede a prior row with the same
        ``(entity_id, recommendation_id, item_id)`` key.

        Calibration uses the latest observation for a given tuple, so
        corrections are consumed at the next ``refit_posterior`` call.
        """
        if timestamp is None:
            timestamp = _now()
        self._conn.execute(
            """
            INSERT INTO outcomes (
                entity_id, recommendation_id, item_id,
                shown, selected, rejected, rating, position,
                reporting_mode, timestamp, inserted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_id, recommendation_id, item_id) DO UPDATE SET
                shown = excluded.shown,
                selected = excluded.selected,
                rejected = excluded.rejected,
                rating = excluded.rating,
                position = excluded.position,
                timestamp = excluded.timestamp,
                inserted_at = excluded.inserted_at
            """,
            (
                str(entity_id),
                recommendation_id,
                str(item_id),
                1 if shown else 0,
                1 if selected else 0,
                1 if rejected else 0,
                rating,
                int(position),
                ReportingMode.PRECISE.value,
                timestamp.strftime(_TIMESTAMP_FMT),
                _now().strftime(_TIMESTAMP_FMT),
            ),
        )

    def _insert(self, rows: Sequence[tuple[object, ...]]) -> int:
        if not rows:
            return 0
        cur = self._conn.cursor()
        cur.executemany(
            """
            INSERT OR IGNORE INTO outcomes (
                entity_id, recommendation_id, item_id,
                shown, selected, rejected, rating, position,
                reporting_mode, timestamp, inserted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return int(cur.rowcount)

    # ---- reads ------------------------------------------------------------

    def __len__(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM outcomes")
        return int(cur.fetchone()[0])

    def count_by_mode(self) -> dict[ReportingMode, int]:
        cur = self._conn.execute(
            "SELECT reporting_mode, COUNT(*) FROM outcomes GROUP BY reporting_mode"
        )
        return {ReportingMode(row[0]): int(row[1]) for row in cur.fetchall()}

    def iter_records(self) -> Iterator[OutcomeRecord]:
        """Yield all records in insertion order."""
        cur = self._conn.execute(
            """
            SELECT entity_id, recommendation_id, item_id,
                   shown, selected, rejected, rating, position,
                   reporting_mode, timestamp
            FROM outcomes
            ORDER BY inserted_at
            """
        )
        for row in cur:
            yield OutcomeRecord(
                entity_id=row[0],
                recommendation_id=row[1],
                item_id=row[2],
                shown=bool(row[3]),
                selected=bool(row[4]),
                rejected=bool(row[5]),
                rating=row[6],
                position=int(row[7]),
                reporting_mode=ReportingMode(row[8]),
                timestamp=datetime.strptime(row[9], _TIMESTAMP_FMT),
            )

    def has_simple_mode_records(self) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM outcomes WHERE reporting_mode = ? LIMIT 1",
            (ReportingMode.SIMPLE.value,),
        )
        return cur.fetchone() is not None
