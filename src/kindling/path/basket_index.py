"""Basket index (PRD §6.1.1 mechanism 3): next-add distribution given the
composition of the held set, ignoring order.

For each session ``(i_1, i_2, ..., i_n)`` and each position ``k`` in it, we
record a training observation ``(basket B_k = {i_1, ..., i_k}, next d_k =
i_{k+1}, weight)``. At query time, given a query basket ``Q``, we compute

    P_basket(d | Q) = sum_h [ w(Q, B_h) * 1{d_h = d} ] / sum_h w(Q, B_h)

where ``w(Q, B_h)`` is the configured basket-similarity function. Efficient
retrieval uses an inverted index: for each item ``i``, the list of training
observations whose basket contains ``i``. For a query ``Q``, the candidate
observation set is the union of posting lists over items in ``Q``.

Similarities implemented (PRD §6.1.2):
- ``coverage`` (default): ``|Q & B_h| / |Q|``
- ``jaccard``:             ``|Q & B_h| / |Q | B_h|``
- ``idf_coverage``:        IDF-weighted variant of coverage
- ``exact``:               ``1 if B_h == Q else 0``
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

import numpy as np

from kindling.lifecycle.decay import DecayProtocol, NoDecay
from kindling.path._sessions import SessionSequence
from kindling.path.tail_index import _session_weight

BasketSimFunc = Callable[[frozenset[object], frozenset[object]], float]


class BasketSimilarity(StrEnum):
    """Which basket-similarity formula to use."""

    COVERAGE = "coverage"
    JACCARD = "jaccard"
    IDF_COVERAGE = "idf_coverage"
    EXACT = "exact"


@dataclass(frozen=True)
class _Observation:
    """A single (basket, next-add) training record."""

    basket: frozenset[object]
    next_item: object
    weight: float


@dataclass
class BasketIndex:
    """Inverted-index-backed basket mechanism store.

    Attributes
    ----------
    observations:
        Training records in insertion order.
    postings:
        ``postings[item]`` = list of observation indices whose basket contains
        ``item``. Enables O(|Q| * avg_postings) candidate enumeration.
    idf:
        Log-scaled inverse document frequency per item, for the
        ``idf_coverage`` similarity.
    """

    observations: list[_Observation] = field(default_factory=list)
    postings: dict[object, list[int]] = field(default_factory=dict)
    idf: dict[object, float] = field(default_factory=dict)

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    @property
    def n_items_indexed(self) -> int:
        return len(self.postings)

    def prune_below(self, support_threshold: float) -> tuple[int, float]:
        """Remove low-weight observations and rebuild the posting lists.
        Returns ``(n_pruned_observations, total_pruned_weight)``."""
        if support_threshold <= 0.0 or not self.observations:
            return 0, 0.0
        kept: list[_Observation] = []
        keep_indices: list[int] = []
        pruned_count = 0
        pruned_weight = 0.0
        for idx, obs in enumerate(self.observations):
            if obs.weight < support_threshold:
                pruned_count += 1
                pruned_weight += obs.weight
                continue
            keep_indices.append(idx)
            kept.append(obs)
        if pruned_count == 0:
            return 0, 0.0
        # Rebuild postings from the surviving observations.
        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        new_postings: dict[object, list[int]] = {}
        for item, posting_list in self.postings.items():
            new_list = [old_to_new[old] for old in posting_list if old in old_to_new]
            if new_list:
                new_postings[item] = new_list
        self.observations = kept
        self.postings = new_postings
        return pruned_count, pruned_weight

    def score(
        self,
        candidate: object,
        query_basket: frozenset[object] | set[object] | tuple[object, ...],
        similarity: BasketSimilarity = BasketSimilarity.COVERAGE,
    ) -> float:
        """Scalar basket signal for a single candidate given ``query_basket``."""
        return float(
            self.score_many(
                candidates=[candidate],
                query_basket=query_basket,
                similarity=similarity,
            )[0]
        )

    def score_many(
        self,
        candidates: Iterable[object],
        query_basket: frozenset[object] | set[object] | tuple[object, ...],
        similarity: BasketSimilarity = BasketSimilarity.COVERAGE,
    ) -> np.ndarray:
        """Vectorized basket signal for a list of candidates."""
        cand_list = list(candidates)
        cand_to_idx = {c: i for i, c in enumerate(cand_list)}
        out = np.zeros(len(cand_list), dtype=np.float64)
        query = frozenset(query_basket)
        if not query or not self.observations:
            return out

        # Collect observation indices whose basket overlaps the query.
        overlap_ids: set[int] = set()
        for item in query:
            overlap_ids.update(self.postings.get(item, ()))
        if not overlap_ids:
            return out

        sim_fn = _similarity_fn(similarity, self.idf)
        total_weight = 0.0
        for obs_idx in overlap_ids:
            obs = self.observations[obs_idx]
            w = sim_fn(query, obs.basket) * obs.weight
            if w <= 0.0:
                continue
            total_weight += w
            cand_slot = cand_to_idx.get(obs.next_item)
            if cand_slot is not None:
                out[cand_slot] += w

        if total_weight > 0:
            out /= total_weight
        return out


def build_basket_index(
    sessions: Iterable[SessionSequence],
    decay: DecayProtocol | None = None,
    reference_timestamp: float | None = None,
    max_basket_size: int = 50,
) -> BasketIndex:
    """Build a ``BasketIndex`` from ordered sessions.

    For each session of length ``n``, we add ``n - 1`` observations, one per
    prefix ``(k, k+1) -> d_k``. Sessions shorter than 2 items contribute
    nothing.

    ``max_basket_size`` caps the size of the stored basket. Large baskets are
    truncated to the most recent ``max_basket_size`` items. This prevents a
    single entity with an enormous historical session from dominating memory
    and posting lists.
    """
    decay_fn: DecayProtocol = decay if decay is not None else cast(DecayProtocol, NoDecay())
    observations: list[_Observation] = []
    postings: dict[object, list[int]] = defaultdict(list)
    item_doc_count: dict[object, int] = defaultdict(int)

    for session in sessions:
        items = session.items
        if len(items) < 2:
            continue
        weight = _session_weight(session, decay_fn, reference_timestamp)
        if weight <= 0.0:
            continue
        for k in range(len(items) - 1):
            next_item = items[k + 1]
            if next_item in items[: k + 1]:
                # Already in the basket - not a next-add event. Skip.
                continue
            basket_items = items[: k + 1]
            if len(basket_items) > max_basket_size:
                basket_items = basket_items[-max_basket_size:]
            basket = frozenset(basket_items)
            obs_idx = len(observations)
            observations.append(_Observation(basket=basket, next_item=next_item, weight=weight))
            for item in basket:
                postings[item].append(obs_idx)
                item_doc_count[item] += 1

    # IDF with log-smoothing: log(1 + N / df). The "+1" keeps singletons
    # informative without going negative.
    n_docs = max(len(observations), 1)
    idf = {item: math.log(1.0 + n_docs / max(df, 1)) for item, df in item_doc_count.items()}
    return BasketIndex(
        observations=observations,
        postings=dict(postings),
        idf=idf,
    )


def _similarity_fn(
    similarity: BasketSimilarity,
    idf: dict[object, float],
) -> BasketSimFunc:
    if similarity is BasketSimilarity.COVERAGE:
        return _coverage
    if similarity is BasketSimilarity.JACCARD:
        return _jaccard
    if similarity is BasketSimilarity.EXACT:
        return _exact
    if similarity is BasketSimilarity.IDF_COVERAGE:
        return _IdfCoverage(idf)
    raise ValueError(f"Unknown similarity: {similarity!r}")


def _coverage(q: frozenset[object], b: frozenset[object]) -> float:
    if not q:
        return 0.0
    return len(q & b) / len(q)


def _jaccard(q: frozenset[object], b: frozenset[object]) -> float:
    union = q | b
    if not union:
        return 0.0
    return len(q & b) / len(union)


def _exact(q: frozenset[object], b: frozenset[object]) -> float:
    return 1.0 if q == b else 0.0


class _IdfCoverage:
    """Weighted coverage where each matching item contributes its IDF weight."""

    def __init__(self, idf: dict[object, float]) -> None:
        self._idf = idf

    def __call__(self, q: frozenset[object], b: frozenset[object]) -> float:
        denom = sum(self._idf.get(i, 0.0) for i in q)
        if denom <= 0.0:
            return 0.0
        return sum(self._idf.get(i, 0.0) for i in q & b) / denom
