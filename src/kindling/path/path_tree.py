"""Full-path tree (PRD §6.1.1 mechanism 1): directional sequence likelihood
given an exact prefix match.

This is the sparsest of the three path mechanisms and carries the strongest
signal when it fires. Phase 2 stores prefixes of length ``[2, max_prefix]``
(default ``max_prefix=3``) and their next-step distributions. At query time,
given the entity's recent trajectory, we try the longest matching prefix
first and back off to shorter prefixes; if none match, signal is 0.

Storage: a dict of ``prefix_tuple -> {next_item: decay_weighted_count}``
plus cached per-prefix totals. A plain dict-of-dicts is small enough for the
Phase 2 scale (ML-1M: ~4k items, tens of thousands of unique prefixes).
A compact trie / radix tree lands in Phase 8 if profiling says we need it.

The invariant captured in the property test:

    count(p) = sum_d count(p -> d) + terminal_count(p)

where ``terminal_count(p)`` is the number of observations where the
trajectory ended exactly at ``p`` with no next step. We don't store
terminal counts separately in Phase 2 because they aren't needed for
next-step prediction; this is noted here so Phase 3's Bayesian likelihood
(which may want them for calibration) knows to add them back.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from kindling._native import NATIVE_AVAILABLE, kindling_native
from kindling.lifecycle.decay import DecayProtocol, NoDecay
from kindling.path._sessions import SessionSequence
from kindling.path.tail_index import _session_weight

DEFAULT_MAX_PREFIX = 3


@dataclass
class PathTree:
    """Full-prefix next-step distribution with back-off.

    Attributes
    ----------
    counts:
        ``counts[prefix][next_item]`` = decay-weighted count.
    row_totals:
        ``row_totals[prefix]`` = ``sum(counts[prefix].values())``.
    max_prefix:
        Longest prefix stored. Prefix length 1 is the tail index's domain; we
        skip it here and let ``TailIndex`` own that mechanism exclusively.
    """

    counts: dict[tuple[object, ...], dict[object, float]] = field(default_factory=dict)
    row_totals: dict[tuple[object, ...], float] = field(default_factory=dict)
    max_prefix: int = DEFAULT_MAX_PREFIX

    def score(self, candidate: object, history: tuple[object, ...]) -> float:
        """Return ``P(candidate | longest matching prefix of history)``.

        Backs off from ``history[-max_prefix:]`` down to prefixes of length 2.
        Returns 0 if no prefix >=2 is known.
        """
        upper = min(len(history), self.max_prefix)
        for length in range(upper, 1, -1):
            prefix = tuple(history[-length:])
            row = self.counts.get(prefix)
            if not row:
                continue
            total = self.row_totals.get(prefix, 0.0)
            if total <= 0.0:
                continue
            return row.get(candidate, 0.0) / total
        return 0.0

    def score_many(
        self,
        candidates: Iterable[object],
        history: tuple[object, ...],
    ) -> np.ndarray:
        """Vectorized score for a candidate list, single history.

        Prefix back-off stays in Python; the per-candidate gather over
        the resolved prefix row routes through the Rust extension when
        available.
        """
        cand_list = list(candidates)
        upper = min(len(history), self.max_prefix)
        for length in range(upper, 1, -1):
            prefix = tuple(history[-length:])
            row = self.counts.get(prefix)
            if not row:
                continue
            total = self.row_totals.get(prefix, 0.0) or 1.0
            if (
                NATIVE_AVAILABLE
                and kindling_native is not None
                and all(isinstance(c, int | np.integer) for c in cand_list)
                and all(isinstance(k, int | np.integer) for k in row)
            ):
                row_items: list[tuple[int, float]] = [
                    (int(k), v)  # type: ignore[call-overload]
                    for k, v in row.items()
                ]
                cand_ints: list[int] = [int(c) for c in cand_list]  # type: ignore[call-overload]
                return np.asarray(
                    kindling_native.path_tree_score_many(row_items, float(total), cand_ints),
                    dtype=np.float64,
                )
            return np.array([row.get(c, 0.0) / total for c in cand_list], dtype=np.float64)
        return np.zeros(len(cand_list), dtype=np.float64)

    @property
    def n_prefixes(self) -> int:
        return len(self.counts)

    def prune_below(self, support_threshold: float) -> tuple[int, float]:
        """Remove (prefix, successor) entries whose stored weight is
        below ``support_threshold``. Returns ``(n_pruned, pruned_weight)``."""
        if support_threshold <= 0.0 or not self.counts:
            return 0, 0.0
        pruned_count = 0
        pruned_weight = 0.0
        for prefix, row in list(self.counts.items()):
            keep: dict[object, float] = {}
            for item, weight in row.items():
                if weight < support_threshold:
                    pruned_count += 1
                    pruned_weight += weight
                else:
                    keep[item] = weight
            if keep:
                self.counts[prefix] = keep
                self.row_totals[prefix] = sum(keep.values())
            else:
                del self.counts[prefix]
                self.row_totals.pop(prefix, None)
        return pruned_count, pruned_weight


def build_path_tree(
    sessions: Iterable[SessionSequence],
    max_prefix: int = DEFAULT_MAX_PREFIX,
    decay: DecayProtocol | None = None,
    reference_timestamp: float | None = None,
) -> PathTree:
    """Build the path tree from training sessions.

    Sessions shorter than 3 items contribute nothing to a ``max_prefix=3``
    tree because they can't form a prefix->successor pair with prefix >= 2.
    """
    if max_prefix < 2:
        raise ValueError(f"max_prefix must be >= 2, got {max_prefix}")
    decay_fn: DecayProtocol = decay if decay is not None else cast(DecayProtocol, NoDecay())
    counts: dict[tuple[object, ...], dict[object, float]] = defaultdict(lambda: defaultdict(float))

    for session in sessions:
        items = session.items
        if len(items) < 3:
            continue
        weight = _session_weight(session, decay_fn, reference_timestamp)
        # For each position k, for each prefix length L in [2, max_prefix],
        # record the (prefix, successor) pair.
        for k in range(1, len(items) - 1):
            successor = items[k + 1]
            if items[k] == successor:
                continue  # same reasoning as TailIndex - skip duplicate emission
            for length in range(2, max_prefix + 1):
                start = k - length + 1
                if start < 0:
                    continue
                prefix = tuple(items[start : k + 1])
                counts[prefix][successor] += weight

    row_totals = {prefix: sum(row.values()) for prefix, row in counts.items()}
    return PathTree(
        counts={prefix: dict(row) for prefix, row in counts.items()},
        row_totals=row_totals,
        max_prefix=max_prefix,
    )
