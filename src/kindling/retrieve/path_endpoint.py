"""Path-endpoint retriever (PRD §5.2).

Returns items that commonly follow the entity's most recent item(s) in the
tail and path structures. Operates on the directional, time-decayed
next-step distribution - pulls the top-scoring continuations from the
longest matching prefix in the path tree, backing off to the tail index.
"""

from __future__ import annotations

from kindling.path.path_tree import PathTree
from kindling.path.tail_index import TailIndex
from kindling.retrieve.protocol import Candidate

_DEFAULT_HISTORY_LEN = 3


class PathEndpointRetriever:
    """Retrieve candidates from the entity's recent trajectory."""

    name = "path_endpoint"

    def __init__(
        self,
        path_tree: PathTree,
        tail_index: TailIndex,
        budget_fraction: float = 1.0,
        history_length: int = _DEFAULT_HISTORY_LEN,
    ) -> None:
        self.path_tree = path_tree
        self.tail_index = tail_index
        self.budget_fraction = budget_fraction
        self.history_length = history_length

    def retrieve(
        self,
        recent_history: tuple[object, ...],
        budget: int,
        exclude: set[object],
    ) -> list[Candidate]:
        """Return up to ``budget`` candidates.

        ``recent_history`` is the tail of the entity's trajectory, oldest to
        newest. ``exclude`` is the set of items already owned - we never
        recommend an item the entity already has.

        Note: This retriever's signature differs from the base protocol
        because it requires an ordered history rather than an unordered set.
        The Engine adapts between them.
        """
        if budget <= 0 or not recent_history:
            return []

        history = tuple(recent_history[-self.history_length :])
        scored: dict[object, float] = {}

        # Pull top candidates from the path tree (longest matching prefix).
        tree_row: dict[object, float] | None = None
        for length in range(len(history), 1, -1):
            prefix = history[-length:]
            row = self.path_tree.counts.get(prefix)
            if row:
                tree_row = row
                break
        if tree_row:
            total = sum(tree_row.values()) or 1.0
            for item, count in tree_row.items():
                if item in exclude:
                    continue
                scored[item] = max(scored.get(item, 0.0), count / total)

        # Back off / augment with tail distribution from the most recent item.
        tail_row = self.tail_index.counts.get(history[-1])
        if tail_row:
            total = self.tail_index.row_totals.get(history[-1], 0.0) or 1.0
            for item, count in tail_row.items():
                if item in exclude:
                    continue
                # Tail contributes a lower-specificity signal - weight it
                # half to keep path-tree candidates on top when they exist.
                adjusted = 0.5 * count / total
                scored[item] = max(scored.get(item, 0.0), adjusted)

        if not scored:
            return []
        ranked = sorted(scored.items(), key=lambda kv: -kv[1])[:budget]
        return [Candidate(item_id=item, score=score, source=self.name) for item, score in ranked]
