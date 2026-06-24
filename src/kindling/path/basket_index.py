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

from kindling._native import NATIVE_AVAILABLE, kindling_native
from kindling.lifecycle.decay import DecayProtocol, NoDecay
from kindling.path._sessions import SessionSequence
from kindling.path.tail_index import _session_weight

BasketSimFunc = Callable[[frozenset[object], frozenset[object]], float]


def _all_ints(xs: Iterable[object]) -> bool:
    return all(isinstance(x, int | np.integer) for x in xs)


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
    # Pair postings: for integer-id baskets, ``pair_postings[(i, j)]`` (with
    # ``i < j``) lists observation indices whose basket contains both items.
    # Empty on non-integer or disabled indexes. Dramatically shrinks the
    # per-query overlap set on session-rich data - popular items' posting
    # lists are huge, popular *pairs* are sparse.
    pair_postings: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    idf: dict[object, float] = field(default_factory=dict)
    # Lazily populated CSR pack of (basket, next, weight) for the native
    # coverage kernel. Reset whenever observations change (prune_below).
    _csr_cache: tuple[list[int], list[int], list[int], list[int], list[float]] | None = field(
        default=None, repr=False, compare=False
    )

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
        new_pair_postings: dict[tuple[int, int], list[int]] = {}
        for pair, posting_list in self.pair_postings.items():
            new_list = [old_to_new[old] for old in posting_list if old in old_to_new]
            if new_list:
                new_pair_postings[pair] = new_list
        self.observations = kept
        self.postings = new_postings
        self.pair_postings = new_pair_postings
        self._csr_cache = None
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
        scan_cap: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Vectorized basket signal for a list of candidates.

        ``scan_cap``: when set, if the per-query observation overlap exceeds
        this value, uniformly subsample to ``scan_cap``. The weighted-mean
        estimator converges as O(1/sqrt(N)), so 10k samples of a 200k-obs
        overlap preserves the signal to within ~1% error while bounding
        latency. ``rng`` seeds the subsample for reproducibility.
        """
        cand_list = list(candidates)
        cand_to_idx = {c: i for i, c in enumerate(cand_list)}
        out = np.zeros(len(cand_list), dtype=np.float64)
        query = frozenset(query_basket)
        if not query or not self.observations:
            return out

        # Collect observation indices whose basket overlaps the query.
        # Pair-index fast path: when we have pair postings AND the query has
        # >=2 integer items, enumerate the C(|Q|, 2) pairs and union their
        # tiny posting lists. Gives overlap_ids only for observations whose
        # basket shares >=2 items with Q - the single-overlap "long tail"
        # contributes at most 1/|Q| weight per obs, i.e. negligible on large
        # queries. For |Q| <= 2 or non-int items, fall back to item postings.
        overlap_ids: set[int] = set()
        if self.pair_postings and len(query) >= 2 and _all_ints(query):
            qlist = sorted(int(q) for q in query)  # type: ignore[call-overload]
            for ai in range(len(qlist)):
                for bi in range(ai + 1, len(qlist)):
                    posting = self.pair_postings.get((qlist[ai], qlist[bi]))
                    if posting:
                        overlap_ids.update(posting)
        else:
            for item in query:
                overlap_ids.update(self.postings.get(item, ()))
        if not overlap_ids:
            return out

        # Scan cap: when the overlap set is huge (popular pairs on
        # ratings-style data), uniformly subsample to bound the kernel
        # scan. MC weighted-mean estimator converges at O(1/sqrt(N)).
        if scan_cap is not None and len(overlap_ids) > scan_cap:
            rng_use = rng if rng is not None else np.random.default_rng(0)
            overlap_arr = np.fromiter(overlap_ids, dtype=np.int64, count=len(overlap_ids))
            sampled = rng_use.choice(overlap_arr, size=scan_cap, replace=False)
            overlap_ids = set(int(i) for i in sampled)

        # Native coverage kernel: 10-30x speedup on hot paths. Gated on
        # integer ids and the default COVERAGE similarity - IDF/Jaccard/
        # exact stay Python-only for now.
        if (
            NATIVE_AVAILABLE
            and kindling_native is not None
            and similarity is BasketSimilarity.COVERAGE
            and _all_ints(cand_list)
            and _all_ints(query)
        ):
            csr = self._get_or_build_int_csr()
            if csr is not None:
                starts, lens, items_flat, next_items, weights = csr
                return np.asarray(
                    kindling_native.basket_score_many(
                        starts,
                        lens,
                        items_flat,
                        next_items,
                        weights,
                        sorted(overlap_ids),
                        [int(q) for q in query],  # type: ignore[call-overload]
                        [int(c) for c in cand_list],  # type: ignore[call-overload]
                    ),
                    dtype=np.float64,
                )

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

    def _get_or_build_int_csr(
        self,
    ) -> tuple[list[int], list[int], list[int], list[int], list[float]] | None:
        """Build (and cache) the int-keyed CSR view of observations used by
        the native coverage kernel. Returns None if any observation has a
        non-integer basket item or next_item (the kernel is int-only)."""
        if self._csr_cache is not None:
            return self._csr_cache
        starts: list[int] = []
        lens: list[int] = []
        items_flat: list[int] = []
        next_items: list[int] = []
        weights: list[float] = []
        for obs in self.observations:
            if not isinstance(obs.next_item, int | np.integer):
                return None
            if not _all_ints(obs.basket):
                return None
            start = len(items_flat)
            basket_ints = sorted(int(i) for i in obs.basket)  # type: ignore[call-overload]
            items_flat.extend(basket_ints)
            starts.append(start)
            lens.append(len(basket_ints))
            next_items.append(int(obs.next_item))
            weights.append(float(obs.weight))
        self._csr_cache = (starts, lens, items_flat, next_items, weights)
        return self._csr_cache


def build_basket_index(
    sessions: Iterable[SessionSequence],
    decay: DecayProtocol | None = None,
    reference_timestamp: float | None = None,
    max_basket_size: int = 50,
    build_pair_index: bool = True,
    distinctiveness_weighting: bool = True,
) -> BasketIndex:
    """Build a ``BasketIndex`` from ordered sessions.

    For each session of length ``n``, we add ``n - 1`` observations, one per
    prefix ``(k, k+1) -> d_k``. Sessions shorter than 2 items contribute
    nothing.

    ``max_basket_size`` caps the size of the stored basket. Large baskets are
    truncated to the most recent ``max_basket_size`` items. This prevents a
    single entity with an enormous historical session from dominating memory
    and posting lists.

    ``build_pair_index`` (default True) adds a second index keyed by item
    pairs. On session data the per-query overlap set shrinks by 10-100x
    because popular *pairs* are much rarer than popular *items*. Memory
    cost is O(|observations| * avg_basket_size^2). Disabled automatically
    if basket items aren't integers.

    ``distinctiveness_weighting`` (default True) divides each observation's
    effective weight by the global frequency of its ``next_item`` being a
    next-add (plus Laplace smoothing). This turns the basket signal into a
    *lift* over popularity: items whose next-add rate is elevated *in the
    context of this basket* rank higher than items that are next-added
    everywhere. Without this, a session-rich workload tends to surface
    popularity (milk, bread) even when the basket is highly specific
    (salsa, avocado, cilantro). The global popularity component is already
    captured by the cooccurrence + cost_population signals, so the basket
    signal shouldn't duplicate it.
    """
    decay_fn: DecayProtocol = decay if decay is not None else cast(DecayProtocol, NoDecay())
    observations: list[_Observation] = []
    postings: dict[object, list[int]] = defaultdict(list)
    pair_postings: dict[tuple[int, int], list[int]] = defaultdict(list)
    item_doc_count: dict[object, int] = defaultdict(int)
    next_item_weight_sum: dict[object, float] = defaultdict(float)
    all_items_int = True

    for session in sessions:
        items = session.items
        if len(items) < 2:
            continue
        base_weight = _session_weight(session, decay_fn, reference_timestamp)
        if base_weight <= 0.0:
            continue
        item_w = session.item_weights
        for k in range(len(items) - 1):
            next_item = items[k + 1]
            if next_item in items[: k + 1]:
                # Already in the basket - not a next-add event. Skip.
                continue
            # Weight observation by destination item's rating weight so
            # highly-rated next-adds contribute more than low-rated ones.
            dest_w = float(item_w[k + 1]) if k + 1 < len(item_w) else 1.0
            weight = base_weight * dest_w
            if weight <= 0.0:
                # Zero-rating destination contributes nothing to path_basket.
                continue
            basket_items = items[: k + 1]
            if len(basket_items) > max_basket_size:
                basket_items = basket_items[-max_basket_size:]
            basket = frozenset(basket_items)
            obs_idx = len(observations)
            observations.append(_Observation(basket=basket, next_item=next_item, weight=weight))
            next_item_weight_sum[next_item] += weight
            for item in basket:
                postings[item].append(obs_idx)
                item_doc_count[item] += 1
            if build_pair_index and all_items_int:
                if _all_ints(basket):
                    basket_ints = sorted(int(i) for i in basket)  # type: ignore[call-overload]
                    for ai in range(len(basket_ints)):
                        for bi in range(ai + 1, len(basket_ints)):
                            pair_postings[(basket_ints[ai], basket_ints[bi])].append(obs_idx)
                else:
                    # Non-int id encountered; drop any partial pair index to
                    # keep the invariant "pair_postings empty unless complete".
                    all_items_int = False
                    pair_postings.clear()

    # Distinctiveness: rewrite each observation's weight to w / baseline(d),
    # where baseline(d) is the total weight of observations whose next_item
    # is d, normalized so the mean baseline is 1.0. This makes the signal a
    # lift over popularity: items whose next-add rate is elevated in this
    # basket's context win, and generic-everywhere items don't.
    if distinctiveness_weighting and observations:
        total_weight = sum(next_item_weight_sum.values()) or 1.0
        n_distinct_next = max(len(next_item_weight_sum), 1)
        mean_baseline = total_weight / n_distinct_next
        eps = mean_baseline * 0.1  # Laplace-style smoothing, 10% of mean.
        rewritten: list[_Observation] = []
        for obs in observations:
            baseline = next_item_weight_sum.get(obs.next_item, 0.0) + eps
            new_weight = obs.weight * mean_baseline / baseline
            rewritten.append(
                _Observation(basket=obs.basket, next_item=obs.next_item, weight=new_weight)
            )
        observations = rewritten

    # IDF with log-smoothing: log(1 + N / df). The "+1" keeps singletons
    # informative without going negative.
    n_docs = max(len(observations), 1)
    idf = {item: math.log(1.0 + n_docs / max(df, 1)) for item, df in item_doc_count.items()}
    return BasketIndex(
        observations=observations,
        postings=dict(postings),
        pair_postings=dict(pair_postings) if all_items_int and build_pair_index else {},
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
