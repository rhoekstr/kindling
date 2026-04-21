"""Outcome-log replay into an OutcomeBatch.

Given a fitted engine and an ``OutcomeLog``, builds an ``OutcomeBatch``
suitable for ``BayesianBlend.fit_posterior``. Items that aren't known to
the fitted engine (e.g., new items that arrived after the last fit) are
skipped with a warning.

Replay is deterministic: same log state + same engine state -> same
batch.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from kindling.blend.likelihoods import OutcomeBatch
from kindling.outcomes.log import OutcomeLog, OutcomeRecord

SignalRowBuilder = Callable[[object, object], "np.ndarray | None"]


def replay_to_batch(
    log: OutcomeLog,
    compute_signal_row: SignalRowBuilder,
) -> OutcomeBatch:
    """Reconstruct an OutcomeBatch from an outcome log.

    ``compute_signal_row(entity_id, item_id) -> np.ndarray | None`` is a
    callback the engine supplies. It returns the signal vector for the
    given (entity, item), or ``None`` when either is unknown to the
    engine. The replay skips unknown combinations.

    Rows grouped by ``(entity_id, recommendation_id)`` become list ids
    for pairwise/multinomial likelihoods. Simple-mode rows each form
    their own single-item list (downweighted by the likelihood).
    """
    by_list: dict[tuple[object, str], list[OutcomeRecord]] = defaultdict(list)
    for rec in log.iter_records():
        if not rec.shown:
            continue
        by_list[(rec.entity_id, rec.recommendation_id)].append(rec)

    if not by_list:
        return OutcomeBatch(
            signal_matrix=np.zeros((0, 0)),
            selected=np.zeros(0, dtype=np.int64),
            positions=np.zeros(0, dtype=np.int64),
            list_ids=np.zeros(0, dtype=np.int64),
        )

    signal_rows: list[np.ndarray] = []
    selected: list[int] = []
    positions: list[int] = []
    list_ids: list[int] = []
    next_list_id = 0
    missing = 0

    for _key, recs in by_list.items():
        # Build one list id per (entity, recommendation_id) group.
        list_rows: list[np.ndarray] = []
        list_sel: list[int] = []
        list_pos: list[int] = []
        for rec in recs:
            row = compute_signal_row(rec.entity_id, rec.item_id)
            if row is None:
                missing += 1
                continue
            list_rows.append(row)
            list_sel.append(1 if rec.selected else 0)
            list_pos.append(max(rec.position, 1))
        if not list_rows:
            continue
        signal_rows.extend(list_rows)
        selected.extend(list_sel)
        positions.extend(list_pos)
        list_ids.extend([next_list_id] * len(list_rows))
        next_list_id += 1

    if missing:
        warnings.warn(
            f"Replay skipped {missing} outcomes whose (entity, item) pair is "
            "not known to the currently fitted engine.",
            stacklevel=2,
        )

    if not signal_rows:
        return OutcomeBatch(
            signal_matrix=np.zeros((0, 0)),
            selected=np.zeros(0, dtype=np.int64),
            positions=np.zeros(0, dtype=np.int64),
            list_ids=np.zeros(0, dtype=np.int64),
        )

    return OutcomeBatch(
        signal_matrix=np.vstack(signal_rows),
        selected=np.asarray(selected, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.int64),
        list_ids=np.asarray(list_ids, dtype=np.int64),
    )
