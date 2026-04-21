"""Tail index (PRD §6.1.1 mechanism 2): Markovian next-step given the most
recent item.

Stores, for every consecutive pair ``(a, b)`` observed in training sessions,
the count weighted by the decay factor evaluated at the observation's age.
At query time, given the entity's last item ``a``, the signal for candidate
``d`` is ``count(a, d) / sum_d count(a, d)``.

Data prerequisite: ordered interactions of length >= 1 per session. Sessions
of length 1 contribute nothing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import cast

import numpy as np

from kindling._native import NATIVE_AVAILABLE, kindling_native
from kindling.lifecycle.decay import DecayProtocol, NoDecay
from kindling.path._sessions import SessionSequence

_SECONDS_PER_DAY = 86400.0


@dataclass
class TailIndex:
    """Mapping from anchor-item to weighted next-item counts.

    Attributes
    ----------
    counts:
        ``counts[anchor][next_item]`` is the decay-weighted count of the pair
        ``(anchor -> next_item)`` across training.
    row_totals:
        ``row_totals[anchor]`` = ``sum(counts[anchor].values())``. Cached for
        constant-time probability queries.
    """

    counts: dict[object, dict[object, float]] = field(default_factory=dict)
    row_totals: dict[object, float] = field(default_factory=dict)

    def score(self, candidate: object, last_item: object | None) -> float:
        """Return ``P(candidate | last_item)`` from the stored distribution."""
        if last_item is None:
            return 0.0
        row = self.counts.get(last_item)
        if not row:
            return 0.0
        total = self.row_totals.get(last_item, 0.0)
        if total <= 0.0:
            return 0.0
        return row.get(candidate, 0.0) / total

    def score_many(self, candidates: Iterable[object], last_item: object | None) -> np.ndarray:
        """Vectorized variant for a list of candidates.

        Routes through ``kindling_native.tail_score_many`` when the Rust
        extension is present; the Rust path trades a Python dict-lookup
        loop for a FxHashMap scan.
        """
        cand_list = list(candidates)
        if last_item is None:
            return np.zeros(len(cand_list), dtype=np.float64)
        row = self.counts.get(last_item, {})
        total = self.row_totals.get(last_item, 0.0) or 1.0

        if (
            NATIVE_AVAILABLE
            and kindling_native is not None
            and row
            and all(isinstance(c, int | np.integer) for c in cand_list)
            and all(isinstance(k, int | np.integer) for k in row)
        ):
            # Rust path: requires integer item ids. Fall back to Python
            # for string / mixed-type ids.
            row_items: list[tuple[int, float]] = [
                (int(k), v)  # type: ignore[call-overload]
                for k, v in row.items()
            ]
            cand_ints: list[int] = [int(c) for c in cand_list]  # type: ignore[call-overload]
            return np.asarray(
                kindling_native.tail_score_many(row_items, float(total), cand_ints),
                dtype=np.float64,
            )
        return np.array(
            [row.get(c, 0.0) / total for c in cand_list],
            dtype=np.float64,
        )

    @property
    def n_anchors(self) -> int:
        return len(self.counts)

    @property
    def n_pairs(self) -> int:
        return sum(len(row) for row in self.counts.values())

    def prune_below(self, support_threshold: float) -> tuple[int, float]:
        """Remove (anchor, successor) entries whose stored weight is below
        ``support_threshold``. Returns ``(n_pruned, total_pruned_weight)``
        for preserved-aggregate bookkeeping.

        Empty rows (anchors whose entire successor list was pruned) are
        dropped, and row_totals is refreshed to stay consistent.
        """
        if support_threshold <= 0.0 or not self.counts:
            return 0, 0.0
        pruned_count = 0
        pruned_weight = 0.0
        for anchor, row in list(self.counts.items()):
            keep: dict[object, float] = {}
            for item, weight in row.items():
                if weight < support_threshold:
                    pruned_count += 1
                    pruned_weight += weight
                else:
                    keep[item] = weight
            if keep:
                self.counts[anchor] = keep
                self.row_totals[anchor] = sum(keep.values())
            else:
                del self.counts[anchor]
                self.row_totals.pop(anchor, None)
        return pruned_count, pruned_weight


def build_tail_index(
    sessions: Iterable[SessionSequence],
    decay: DecayProtocol | None = None,
    reference_timestamp: float | None = None,
) -> TailIndex:
    """Build a ``TailIndex`` from training sessions.

    Parameters
    ----------
    sessions:
        Iterable of ordered sequences.
    decay:
        Decay function applied to each observed pair based on session age.
        Defaults to ``NoDecay`` - callers that want temporal weighting must
        pass an explicit decay function and ``reference_timestamp``.
    reference_timestamp:
        Unix-seconds timestamp used as the "now" point when computing age.
        Required when ``decay`` is time-sensitive and sessions carry
        timestamps; ignored otherwise.
    """
    decay_fn: DecayProtocol = decay if decay is not None else cast(DecayProtocol, NoDecay())
    counts: dict[object, dict[object, float]] = defaultdict(lambda: defaultdict(float))

    for session in sessions:
        items = session.items
        if len(items) < 2:
            continue
        weight = _session_weight(session, decay_fn, reference_timestamp)
        for a, b in pairwise(items):
            if a == b:
                # Tail signal treats self-loops as noise - a user re-rating
                # the same item twice in a row tells us nothing about what
                # item comes next.
                continue
            counts[a][b] += weight

    row_totals = {anchor: sum(row.values()) for anchor, row in counts.items()}
    return TailIndex(
        counts={anchor: dict(row) for anchor, row in counts.items()},
        row_totals=row_totals,
    )


def _session_weight(
    session: SessionSequence,
    decay: DecayProtocol,
    reference_timestamp: float | None,
) -> float:
    """Age of this session in days -> decay weight."""
    if session.end_timestamp is None or reference_timestamp is None:
        return 1.0
    age_days = max(0.0, (reference_timestamp - session.end_timestamp) / _SECONDS_PER_DAY)
    return float(decay(age_days))
