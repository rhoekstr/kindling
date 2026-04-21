"""Path-based signal structures (PRD §6.1.1).

Three mechanisms:

- ``TailIndex`` - Markovian next-step given the most recent item.
- ``PathTree`` - full-prefix next-step (trigrams in v1, extensible).
- ``BasketIndex`` - next-add given the composition of the current held set.

Each exposes ``build_from_sessions(...)`` taking a list of ordered session
item sequences, and ``score(candidate, context, ...)`` returning a signal in
``[0, 1]`` for a single candidate given the entity's context.

The three are deliberately independent structures, not views on a shared
trie. The Bayesian blend in Phase 3 learns their relative weights from data;
building them separately is what enables the decorrelation step to produce
the interpretable per-mechanism weights described in PRD §6.2.
"""

from kindling.path.basket_index import BasketIndex, BasketSimilarity, build_basket_index
from kindling.path.path_tree import PathTree, build_path_tree
from kindling.path.tail_index import TailIndex, build_tail_index

__all__ = [
    "BasketIndex",
    "BasketSimilarity",
    "PathTree",
    "TailIndex",
    "build_basket_index",
    "build_path_tree",
    "build_tail_index",
]
