"""The kindling engine — closed-form base + auto-gated channels via ``kindling_core``.

A fused score per (user, candidate) built from a closed-form base and a
set of counting-statistic channels, each activated by a measurable
property of the dataset. Numerics run in the Rust core; this module is the
orchestration + profiling shell. See ``docs/REFERENCE.md`` for the full
architecture and ``engine.activation_plan`` for the runtime gate decisions.

    fit(interactions):
        1. ingest + preprocess
        2. profile the data → regime (catalog size, timestamps, rating
           signal, rating-burst, history length, sessions)
        3. build the base: rating-weighted EASE (catalog ≤ ease_max_items)
           or wilson-normalized cooccurrence (above), auto-selected
        4. build the active channels (trend / last-item / transitions /
           user-CF) — each gated on the regime
        5. (open_catalog) extend the catalog with metadata-only items;
           reserve cold_slots for cold-item exposure

    recommend(entity_id, n):
        score = z(base) + 0.5·z(trend) + 0.25·z(last_item)
              + 0.25·z(transitions) + 1.0·z(user_cf)   [active channels only]
        → top-N, with reserved cold slots ranked by content similarity.

    recommend_for_items(seed_item_ids, n):
        serve a brand-new / anonymous user from ad-hoc seeds with no
        per-user training; empty/all-unknown seeds → popularity fallback.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from kindling._native import CORE_AVAILABLE, kindling_core
from kindling.explain import Explanation
from kindling.graph.cooc_transform import apply_cooc_transform, resolve_cooc_transform
from kindling.ingest.contract import canonicalize, validate_interactions
from kindling.ingest.sessions import infer_sessions
from kindling.path._sessions import sessions_from_interactions
from kindling.path.basket_index import BasketIndex, build_basket_index
from kindling.path.tail_index import TailIndex, build_tail_index
from kindling.preprocess import preprocess_interactions, weights_of

if TYPE_CHECKING:
    from pathlib import Path

    from kindling.activation import ActivationPlan


@dataclass(frozen=True)
class Recommendation:
    """Single output row: item + composite score + per-layer contributions."""

    item_id: object
    score: float
    base_kind: str  # "cooc" | "ease" | "cooc_fused" | "cold_popularity" | "cold_content"
    explanation: Explanation | None = None


@dataclass
class EngineState:
    """Everything the v2 recommend path reads."""

    # Catalog
    item_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    item_to_idx: dict[object, int] = field(default_factory=dict)
    n_items: int = 0
    # Train-subspace size: interaction structures (EASE B, cooc CSRs,
    # transitions) cover indices [0, n_train_items); open-catalog
    # extension items occupy [n_train_items, n_items).
    n_train_items: int = 0
    # User → owned items (sparse)
    owned_by_entity: dict[object, np.ndarray] = field(default_factory=dict)
    entity_to_user_idx: dict[object, int] = field(default_factory=dict)
    n_users: int = 0
    # Plan decisions
    kernel: str = "pure_count"
    half_life_days: float = 30.0
    enabled_boost_layers: list[str] = field(default_factory=list)
    # Base layer: global cooc CSR (raw, not popularity-corrected).
    # We tested cosine-as-base; on popularity-biased test sets like
    # amazon-beauty it over-suppressed popular items and crashed
    # hit/recall/NDCG. Reverting to raw cooc as the base; bumping
    # retrieval_budget so tail-favored items survive into the pool.
    cooc_data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    cooc_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    cooc_indptr: np.ndarray = field(default_factory=lambda: np.array([0], dtype=np.int32))
    # Boost layers (per-layer adjacency / scoring state)
    # layer_name → CSR triple for the layer's cooc-shaped adjacency
    boost_layer_adjacencies: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    # Path-family signals (not cooc-shaped — separate Python objects).
    tail_index: TailIndex | None = None
    basket_index: BasketIndex | None = None
    history_by_entity: dict[object, tuple[object, ...]] = field(default_factory=dict)
    # EASE base scorer: dense item-item weight matrix B (n_items × n_items,
    # f32, zero diagonal). None when the cooc base is active.
    ease_b: np.ndarray | None = None
    base_scorer_used: str = "cooc"  # "cooc" | "ease"
    # Trend signal: z-normalized recent-window item popularity (length
    # n_items). None when timestamps are absent or trend_alpha == 0.
    trend_z: np.ndarray | None = None
    trend_alpha: float = 0.0
    # Per-item train popularity (interaction counts, length n_items). The
    # zero-seed cold-start fallback for brand-new users: recommend_for_items([])
    # returns top popularity (the benchmark's cold-data champion).
    item_popularity: np.ndarray | None = None
    # Sequential transition channel: directional cooc CSR (item → item),
    # user-level timestamp-ordered. None when gated off (no timestamps /
    # rating bursts / transition_alpha == 0).
    trans_data: np.ndarray | None = None
    trans_indices: np.ndarray | None = None
    trans_indptr: np.ndarray | None = None
    transition_alpha: float = 0.0
    transition_last_k: int = 5
    transition_decay: float = 0.7
    # Content channel: generic item features (item_features.ItemFeatures)
    # + per-item coldness ∈ [0, 1] (1 = no train interactions). The
    # channel contribution is cold-gated: content_alpha · coldness ·
    # z(content) — content only speaks where interaction signal is
    # data-starved. None when metadata absent or content_alpha == 0.
    content_features: Any = None
    content_coldness: np.ndarray | None = None
    content_alpha: float = 0.0
    # Release recency per catalog item (exp-decayed days since release,
    # schema-inferred datetime column). Used only inside the cold slot.
    cold_recency: np.ndarray | None = None
    cold_recency_beta: float = 0.0
    # Last-item context channel: z(B[last_item, :]) — the EASE row of
    # the user's most recent item as a current-taste signal. Only
    # active on the EASE path (needs B).
    last_item_alpha: float = 0.0
    # User-user CF channel: item→user inverted CSR + per-user degree.
    # Otsuka-Ochiai k-NN over interaction vectors; gated to sparse-
    # history datasets. None when gated off.
    uu_users_data: np.ndarray | None = None  # concatenated user ids
    uu_users_indptr: np.ndarray | None = None  # per-item offsets
    uu_user_deg: np.ndarray | None = None
    user_row_items: dict[int, np.ndarray] = field(default_factory=dict)
    user_cf_alpha: float = 0.0
    user_cf_k: int = 100
    # Rating-signal classification (gates the EASE rating-weighting path).
    signal_kind: str = "unknown"  # "binary" | "counts" | "ratings"
    # Calibrated scoring config
    z_threshold: float = 2.5
    boost_multiplier: float = 3.0
    # Diagnostics
    fit_seconds: float = 0.0
    profile: dict[str, Any] = field(default_factory=dict)


class Engine:
    """The recommender engine. Fit on an interaction frame, then ``recommend``."""

    def __init__(
        self,
        retrieval_budget: int = 500,
        random_state: int = 0,
        # Base scorer selection. "cooc" is the legacy raw-count sum;
        # "ease" is the closed-form EASE linear model (Steck 2019) —
        # an inverse-Gram reweighting of the same co-occurrence signal
        # that subtracts popularity/redundancy structure. The gap-
        # decomposition diagnostic (2026-06) showed raw cooc scores
        # degenerate toward popularity ranking; EASE is the fix.
        #   "auto" — EASE when n_items <= ease_max_items (the O(n³)
        #            inversion gate), else cooc.
        # When EASE is active it powers BOTH retrieval and base scoring;
        # boost layers stack on top unchanged.
        base_scorer: str = "auto",
        # EASE L2 regularization. None = auto: λ = 20 × (n_obs / n_items),
        # i.e. ~20× the mean Gram diagonal. Empirically tracks the best
        # fixed λ on both ml1m (dense; wanted ~8k) and amazon-beauty
        # (sparse; wanted ~250) — λ must grow with item-count density or
        # the inverse under-regularizes the popular-item rows.
        ease_lambda: float | None = None,
        ease_max_items: int = 20_000,
        # Weight transform for the cooc base scorer — applies ONLY on the
        # cooc path (n_items > ease_max_items; <=20k uses EASE and is left
        # untouched). Raw co-counts row-sum ≈ popularity on large catalogs;
        # a popularity-normalized weight lifts amazon-book-academic +68%
        # NDCG@20. "auto" → wilson (book winner + safest vs popular-item
        # over-suppression). "raw" restores prior behavior. See
        # graph/cooc_transform.py.
        cooc_base_transform: str = "auto",
        # Trend signal: z-normalized item popularity within the most
        # recent `trend_window_fraction` of the training time span,
        # blended additively into the (z-normalized) EASE base:
        #   score = z(ease) + trend_alpha · z(recent_popularity)
        # Motivated by the chronological eval splits: held-out events
        # come from the final time window, where global popularity
        # drift is a first-order signal the pairwise model can't see
        # (a bare trending-items list beat every persona variant on
        # ml1m). Gated by timestamp availability. 0.0 disables.
        trend_alpha: float = 0.5,
        trend_window_fraction: float = 0.10,
        # Sequential transition channel: directional cooc D[i→j] built
        # from each user's timestamp-ordered history; at recommend time
        # the last `transition_last_k` owned items vote with exponential
        # decay:
        #   trans = Σ_j decay^j · D[last_j, :]
        #   score += transition_alpha · z(trans)
        # Gated by: timestamps present AND NOT rating_burst_detected.
        # On burst datasets (ml1m: users rate dozens of movies in one
        # sitting) within-burst order is meaningless and this channel
        # measurably hurts; on purchase streams (amazon) it lifts both
        # NDCG and recall. 0.0 disables.
        transition_alpha: float = 0.25,
        transition_last_k: int = 5,
        transition_decay: float = 0.7,
        # Content channel (opt-in): generic item-feature similarity from
        # `item_metadata` (schema-inferring extractor — categorical /
        # multi-categorical / numeric / text columns all handled; see
        # item_features.py). Contribution is COLD-GATED per item:
        #   + content_alpha · clip(1 − train_count/warmth_threshold, 0, 1) · z(content)
        # so it only speaks for items the interaction channels are
        # data-starved on, never diluting warm ranking.
        #
        # Measured 2026-06 (ml1m, 100% metadata coverage): un-gated
        # blending HURTS warm ranking (0.2841 → 0.2755 at α=0.5);
        # cold-gated blending is harmless but unrewarded because the
        # canonical protocol's held-out items are never cold. Default
        # 0.0 (off) until a cold-start protocol shows lift. amazon-beauty
        # note: its 2023 metadata matches only 0.17% of the 2014 review
        # catalog — the channel is inert there regardless.
        content_alpha: float = 0.0,
        content_warmth_threshold: int = 20,
        # Metadata smoothing (default off): augment the active base item-item
        # matrix (cooc or EASE) with a metadata-kNN graph whose edge weights are
        # the fitted prediction of the base's own weight from metadata
        # similarity. Self-scaling (per base) and self-gating (dead metadata ⇒
        # no-op). 'auto' picks the link from the data (poisson for repeat/count,
        # logistic for no-repeat binary). See graph/metadata_smoothing.py.
        metadata_smoothing: str = "off",
        metadata_smoothing_family: str = "auto",
        metadata_smoothing_topk: int = 20,
        # Dose: edge weight = sim · cap · base_max (a fixed fraction of the
        # base's strongest edge). Empirical optimum ~0.05–0.1, dataset-
        # dependent; the grounded prediction (cap=0) under-doses badly.
        metadata_smoothing_cap: float = 0.1,
        # Open catalog: when item_metadata is provided, extend the
        # recommendable catalog with metadata-only items (items that
        # never appear in train). Interaction channels score them zero
        # by construction; the cold-gated content channel is their only
        # voice. This is the cold-START serving capability — without it
        # the engine can only ever re-rank its training history (steam:
        # 13% of test events were structurally unreachable).
        open_catalog: bool = True,
        # Hard cap on metadata-only catalog extension items. None = auto
        # (memory-aware: spend the RAM headroom under 80% of physical
        # memory, after reserving the estimated interaction-fit peak).
        # Set an int to pin it regardless of RAM.
        open_catalog_max_extension: int | None = None,
        # Reserved cold slots per recommendation list. 0 = pure-accuracy
        # ranking (default). 1 recommended for serving deployments that
        # value new-item discoverability: the last slot goes to the best
        # cold-content candidate (requires item_metadata + content
        # features). The aggregate-NDCG cost is small and explicit; the
        # cold-coverage gain is not achievable by blend weights at all.
        cold_slots: int = 0,
        # Release-recency weight inside the cold slot. Cold purchases
        # skew heavily toward NEW releases; ranking cold candidates by
        # z(content) + beta·exp(−days_since_release/180) lifted steam
        # cold recovery 6.0% → 8.5% at zero warm cost. The release date
        # is schema-inferred: any item_metadata column whose values are
        # majority-parseable as datetimes. 0 disables.
        cold_recency_beta: float = 2.0,
        # Cold-slot ranking mechanism: the reserved cold slots (cold_slots>0)
        # rank metadata-only / barely-seen items by content-space cosine to the
        # user's owned items (item_features). Embedding-imputation cold ranking
        # was tried and removed — validated standalone but did not transfer
        # through the production EASE path (scale mismatch). See
        # docs/EXPERIMENTS.md §4.9.
        # New-user popularity-shrinkage (recommend_for_items only). A brand-new
        # user's seed-based score is noisy when seeds are few; this adds a
        # popularity-prior term  (pop_prior · z(log popularity))  with
        # pop_prior = cold_user_pop_prior / n_seeds, so the ranking leans on the
        # popularity prior when evidence is thin and on the personalized signal
        # as seeds accumulate (empirical-Bayes shrinkage). Removes the 1-seed
        # dip below popularity on popularity-heavy catalogs (onboarding curve)
        # without retraining. Scoped to the new-user path — `recommend` (known
        # users with full history) passes pop_prior=0, so warm ranking and all
        # warm/cold-user benchmarks are unchanged. Default 8.0 (tuned on the
        # onboarding curve): closes the 1-seed dip on ml1m, mostly on steam, is
        # harmless on sparse catalogs (their popularity is too flat to pull),
        # and preserves the high-seed wins (decays as c/n_seeds). 0 disables.
        cold_user_pop_prior: float = 8.0,
        # Last-item context channel: blends z(B[last_item, :]) — the
        # EASE row of the user's most recent item — emphasizing the
        # current taste neighborhood over the whole-history average.
        # Unlike the directional transition channel this is burst-robust
        # (it reads co-occurrence structure, not within-burst order), so
        # it is NOT burst-gated and helps on both ml1m (+1.3% NDCG,
        # +1.7% MRR) and beauty (+8.8% recall, +6.7% HR). β=0.5
        # overshoots on both; profile-wide recency decay was also
        # measured and rejected (forgetting full history always hurt).
        last_item_alpha: float = 0.25,
        # User-user CF channel: k-NN over user interaction vectors
        # (Otsuka-Ochiai cosine), neighbors vote for their items.
        # Complementary geometry to item-item: connects a sparse-history
        # user to items none of their own items co-occur with.
        #
        # Gated by MEDIAN USER HISTORY ≤ user_cf_history_gate: on sparse-
        # history data (beauty, median ≈ 7) it is the largest single
        # post-EASE lift (+5% NDCG, +6% HR at k=100, α=1.0); on dense-
        # history data (ml1m, median ≈ 100) EASE already encodes the
        # user-user structure and the crude k-NN measurably hurts
        # (0.2879 → 0.2814 at α=0.5). k=200 dilutes; α beyond 1.0 flat.
        user_cf_alpha: float = 1.0,
        user_cf_k: int = 100,
        user_cf_history_gate: int = 20,
        # Rating-weighted EASE Gram: when the dataset carries true
        # ratings (signal_kind == "ratings"), weight X by the rating
        # (mean-normalized so the Gram scale and λ semantics stay
        # comparable to the binary case) instead of binarizing.
        # "auto" = on for ratings data only; "on"/"off" force.
        ease_use_weights: str = "auto",
    ):
        if not CORE_AVAILABLE:
            raise ImportError(
                "kindling_core extension not available; build with "
                "`maturin build` in native/kindling_core/"
            )
        self.retrieval_budget = retrieval_budget
        self.random_state = random_state
        if base_scorer not in ("auto", "cooc", "ease"):
            raise ValueError(f"base_scorer must be 'auto', 'cooc', or 'ease'; got {base_scorer!r}")
        self.base_scorer = base_scorer
        if ease_lambda is not None and ease_lambda <= 0.0:
            raise ValueError(f"ease_lambda must be > 0 or None (auto); got {ease_lambda!r}")
        self.ease_lambda = ease_lambda
        if ease_max_items < 1:
            raise ValueError(f"ease_max_items must be >= 1; got {ease_max_items!r}")
        self.ease_max_items = ease_max_items
        resolve_cooc_transform(cooc_base_transform)  # validate at construction
        self.cooc_base_transform = cooc_base_transform
        if trend_alpha < 0.0:
            raise ValueError(f"trend_alpha must be >= 0; got {trend_alpha!r}")
        self.trend_alpha = trend_alpha
        if not (0.0 < trend_window_fraction <= 1.0):
            raise ValueError(
                f"trend_window_fraction must be in (0, 1]; got {trend_window_fraction!r}"
            )
        self.trend_window_fraction = trend_window_fraction
        if transition_alpha < 0.0:
            raise ValueError(f"transition_alpha must be >= 0; got {transition_alpha!r}")
        self.transition_alpha = transition_alpha
        if transition_last_k < 1:
            raise ValueError(f"transition_last_k must be >= 1; got {transition_last_k!r}")
        self.transition_last_k = transition_last_k
        if not (0.0 < transition_decay <= 1.0):
            raise ValueError(f"transition_decay must be in (0, 1]; got {transition_decay!r}")
        self.transition_decay = transition_decay
        if content_alpha < 0.0:
            raise ValueError(f"content_alpha must be >= 0; got {content_alpha!r}")
        self.content_alpha = content_alpha
        if metadata_smoothing not in ("off", "on", "auto", "cooc", "ease"):
            raise ValueError(
                f"metadata_smoothing must be off|on|auto|cooc|ease; got {metadata_smoothing!r}"
            )
        self.metadata_smoothing = metadata_smoothing
        if metadata_smoothing_family not in ("auto", "ols", "poisson", "logistic"):
            raise ValueError(
                "metadata_smoothing_family must be auto|ols|poisson|logistic; "
                f"got {metadata_smoothing_family!r}"
            )
        self.metadata_smoothing_family = metadata_smoothing_family
        if metadata_smoothing_topk < 1:
            raise ValueError(
                f"metadata_smoothing_topk must be >= 1; got {metadata_smoothing_topk!r}"
            )
        self.metadata_smoothing_topk = metadata_smoothing_topk
        if metadata_smoothing_cap < 0.0:
            raise ValueError(f"metadata_smoothing_cap must be >= 0; got {metadata_smoothing_cap!r}")
        self.metadata_smoothing_cap = metadata_smoothing_cap
        if content_warmth_threshold < 1:
            raise ValueError(
                f"content_warmth_threshold must be >= 1; got {content_warmth_threshold!r}"
            )
        self.content_warmth_threshold = content_warmth_threshold
        self.open_catalog = bool(open_catalog)
        self.open_catalog_max_extension = open_catalog_max_extension
        if cold_slots < 0:
            raise ValueError(f"cold_slots must be >= 0; got {cold_slots!r}")
        self.cold_slots = int(cold_slots)
        if cold_recency_beta < 0:
            raise ValueError(f"cold_recency_beta must be >= 0; got {cold_recency_beta!r}")
        self.cold_recency_beta = float(cold_recency_beta)
        if cold_user_pop_prior < 0.0:
            raise ValueError(f"cold_user_pop_prior must be >= 0; got {cold_user_pop_prior!r}")
        self.cold_user_pop_prior = float(cold_user_pop_prior)
        if last_item_alpha < 0.0:
            raise ValueError(f"last_item_alpha must be >= 0; got {last_item_alpha!r}")
        self.last_item_alpha = last_item_alpha
        if user_cf_alpha < 0.0:
            raise ValueError(f"user_cf_alpha must be >= 0; got {user_cf_alpha!r}")
        self.user_cf_alpha = user_cf_alpha
        if user_cf_k < 1:
            raise ValueError(f"user_cf_k must be >= 1; got {user_cf_k!r}")
        self.user_cf_k = user_cf_k
        if user_cf_history_gate < 0:
            raise ValueError(f"user_cf_history_gate must be >= 0; got {user_cf_history_gate!r}")
        self.user_cf_history_gate = user_cf_history_gate
        if ease_use_weights not in ("auto", "on", "off"):
            raise ValueError(
                f"ease_use_weights must be 'auto' | 'on' | 'off'; got {ease_use_weights!r}"
            )
        self.ease_use_weights = ease_use_weights
        self._state: EngineState | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    @staticmethod
    def detect_rating_signal(values: np.ndarray) -> str:
        """Classify a rating/weight column as 'binary' | 'counts' | 'ratings'.

        Heuristic:
          - single value (or coefficient of variation < 5%): 'binary'
          - p99/median > 10 (heavy right tail): 'counts'
          - bounded narrow range with ≥3 distinct values: 'ratings'
          - default fallback: 'counts'

        Only 'ratings' data drives the EASE rating-weighted Gram
        (``ease_use_weights='auto'``); 'binary'/'counts' binarize.
        """
        vals = np.asarray(values, dtype=np.float64)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return "binary"
        unique = np.unique(vals)
        if len(unique) <= 1:
            return "binary"
        cv = vals.std() / max(abs(vals.mean()), 1e-9)
        if cv < 0.05:
            return "binary"
        median = np.median(vals)
        p99 = np.percentile(vals, 99)
        if p99 / max(median, 1e-9) > 10.0:
            return "counts"
        if unique.max() <= 10.0 and len(unique) >= 3:
            return "ratings"
        return "counts"

    # Interaction-fit peak model: bytes ≈ A·n_obs + B·n_train_items,
    # calibrated to the two largest fits measured (steam 14k items / 7M
    # obs ≈ 3.4 GB; book-chrono 357k items / 8M obs ≈ 17.4 GB). The
    # per-item term dominates because the cooc CSR + its build hashmap
    # scale with the train catalog, not interaction count.
    _PEAK_BYTES_PER_OBS = 400
    _PEAK_BYTES_PER_TRAIN_ITEM = 39_800
    # Each extension item costs catalog-vector slots (allocated per
    # recommend over n_items_ext) + a retained metadata row. Generous;
    # tuned so the book OOM (357k train + 200k ext) caps to a safe size.
    _EXTENSION_BYTES_PER_ITEM = 30_000
    # Fixed headroom reserved for the OS + Python runtime working set. The
    # jetsam/OOM wall sits below 100% of physical RAM because the kernel and
    # other processes need resident memory; on a loaded 24 GB box a fit
    # estimated at ~19 GB (80%) was OOM-killed. Reserving an absolute floor
    # (not just a percentage) makes the cap fail safe on smaller machines:
    # ceiling = min(0.80·total, total − reserve), so it only ever tightens.
    _OS_RESERVE_BYTES = 6 * 1024**3

    def _open_catalog_extension_cap(self, n_obs: int, n_train_items: int) -> int:
        """Max metadata-only extension items that fit under the RAM budget.

        Reserves the estimated interaction-fit peak plus an OS/runtime
        floor, then spends the remaining headroom on the extension. Returns
        0 when the interaction fit alone already exceeds budget — the engine
        then runs catalog-only rather than risking an OOM.
        """
        if self.open_catalog_max_extension is not None:
            return max(0, int(self.open_catalog_max_extension))
        # Physical RAM via POSIX sysconf (Linux/macOS) — no third-party dep.
        # Conservative 8 GB fallback where sysconf is unavailable (e.g. Windows).
        try:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            total = 8 * 1024**3
        # Budget against PHYSICAL RAM (macOS swap absorbs some overrun, so
        # the wall sits near physical, not `available`), but take the more
        # conservative of 80% and (total − OS reserve). On 24 GB the latter
        # (~18 GB) is tighter than 0.80×24 (~19.2 GB), which is what the
        # book OOM showed was needed.
        ceiling = min(0.80 * total, float(total - self._OS_RESERVE_BYTES))
        interaction_peak = (
            self._PEAK_BYTES_PER_OBS * n_obs + self._PEAK_BYTES_PER_TRAIN_ITEM * n_train_items
        )
        headroom = ceiling - interaction_peak
        if headroom <= 0:
            return 0
        return int(headroom / self._EXTENSION_BYTES_PER_ITEM)

    def fit(
        self,
        interactions: pd.DataFrame,
        item_metadata: pd.DataFrame | None = None,
    ) -> Engine:
        t0 = time.perf_counter()
        # Same contract as v1: validate → canonicalize → preprocess.
        schema = validate_interactions(interactions)
        canonical = canonicalize(interactions, schema)
        canonical, _ctx = preprocess_interactions(canonical, use_ratings=None)
        interactions = canonical
        weights = weights_of(interactions).astype(np.float32)
        # Classify the rating signal — gates the EASE rating-weighted Gram.
        signal_kind = self.detect_rating_signal(weights)
        # Build catalogs.
        item_ids = pd.Index(interactions["item_id"].unique())
        item_to_idx = {item: i for i, item in enumerate(item_ids)}
        n_items = len(item_ids)
        entity_ids = pd.Index(interactions["entity_id"].unique())
        entity_to_user_idx = {e: i for i, e in enumerate(entity_ids)}
        n_users = len(entity_ids)

        item_idx = interactions["item_id"].map(item_to_idx).to_numpy(dtype=np.int64)
        user_idx = interactions["entity_id"].map(entity_to_user_idx).to_numpy(dtype=np.int64)
        # ── Open catalog: metadata-only items get catalog indices ≥
        # n_items. All interaction-derived structures (cooc, EASE,
        # transitions, user-CF) stay in the train subspace
        # [0, n_items); only catalog-length vectors (trend, content,
        # blend output) span the extension.
        n_items_ext = n_items
        extension_capped: dict[str, object] | None = None
        if self.open_catalog and item_metadata is not None and "item_id" in item_metadata.columns:
            extra = pd.Index(item_metadata["item_id"].dropna().unique()).difference(item_ids)
            # Memory-aware cap. Extension items cost catalog-length-vector
            # slots + retained metadata rows but add NOTHING to the cooc
            # CSR (interaction structures live in [0, n_items); cooc uses
            # n_items, not n_items_ext). The fit's memory peak is driven by
            # the interaction fit, which an uncapped extension can tip over
            # (book: 357k train + naive 200k extension OOM'd a 24GB box).
            # Estimate the interaction peak from a two-term model
            # calibrated to the steam (14k items / 7M obs ≈ 3.4GB) and
            # book-chrono (357k items / 8M obs ≈ 17.4GB) fits, then spend
            # only the headroom under 80% of physical RAM on the extension.
            n_keep = len(extra)
            if n_keep:
                cap = self._open_catalog_extension_cap(n_obs=len(user_idx), n_train_items=n_items)
                if n_keep > cap:
                    # `profile` doesn't exist yet — stash for write below.
                    extension_capped = {"requested": int(n_keep), "kept": int(cap)}
                    # Keep the first `cap` in metadata order — callers sort
                    # metadata by importance (the book loader by salesRank).
                    extra = extra[:cap]
                    n_keep = cap
            if n_keep:
                item_ids = item_ids.append(pd.Index(extra))
                for j, it in enumerate(extra):
                    item_to_idx[it] = n_items + j
                n_items_ext = len(item_ids)
        timestamps_col = (
            interactions["timestamp"].to_numpy(dtype=np.float64)
            if "timestamp" in interactions.columns
            else None
        )

        # owned_by_entity + history (timestamp-ordered) per entity.
        # Vectorized: one stable lexsort over (entity, time) + boundary
        # split. The per-user pandas groupby this replaces dominated fit
        # time on large datasets (2.3M users on steam → 10+ minutes).
        # entity_ids is first-appearance ordered, so ascending user_idx
        # preserves the original dict insertion order.
        owned_by_entity: dict[object, np.ndarray] = {}
        history_by_entity: dict[object, tuple[object, ...]] = {}
        if timestamps_col is not None:
            order = np.lexsort((timestamps_col, user_idx))
        else:
            order = np.lexsort((np.arange(len(user_idx)), user_idx))
        su = user_idx[order]
        si = item_idx[order]
        item_ids_arr = np.asarray(item_ids, dtype=object)
        if len(su):
            boundaries = np.flatnonzero(np.diff(su)) + 1
            starts = np.concatenate(([0], boundaries))
            for start, items_arr in zip(starts, np.split(si, boundaries)):
                ent = entity_ids[int(su[start])]
                owned_by_entity[ent] = items_arr
                history_by_entity[ent] = tuple(item_ids_arr[items_arr])

        # ── Profile + Plan decisions.
        profile = self._profile(interactions, weights, n_users, n_items)
        plan = self._plan(profile)
        # Open-catalog bookkeeping (computed above, before `profile` existed).
        profile["n_extension_items"] = int(n_items_ext - n_items)
        if extension_capped is not None:
            profile["open_catalog_extension_capped"] = extension_capped

        # ── Base cooc (raw, not popularity-corrected).
        cooc_data, cooc_indices, cooc_indptr = kindling_core.build_cooccurrence(
            user_idx,
            item_idx,
            weights,
            n_users=n_users,
            n_items=n_items,
            kernel=plan["kernel"],
            alpha=plan["alpha"],
            half_life_days=plan["half_life_days"],
            timestamps=timestamps_col,
        )
        cooc_data = np.asarray(cooc_data, dtype=np.float32)
        cooc_indices = np.asarray(cooc_indices, dtype=np.int32)
        cooc_indptr = np.asarray(cooc_indptr, dtype=np.int32)

        # ── EASE base (if selected). Closed-form inverse-Gram reweighting
        # of the co-occurrence signal; replaces raw-count cooc for both
        # retrieval and base scoring. Gated by catalog size: the O(n³)
        # Cholesky inversion is feasible to ~20k items.
        ease_b: np.ndarray | None = None
        base_scorer_used = "cooc"
        ease_eligible = self.base_scorer == "ease" or (
            self.base_scorer == "auto" and n_items <= self.ease_max_items
        )
        transition_gate_open = (
            timestamps_col is not None
            and len(timestamps_col) > 0
            and not profile.get("rating_burst_detected", False)
        )
        # Effective hyper-params: the configured/auto values.
        eff_trend_alpha = self.trend_alpha
        eff_trans_alpha = self.transition_alpha if transition_gate_open else 0.0
        if ease_eligible:
            eff_lambda = (
                self.ease_lambda
                if self.ease_lambda is not None
                else 20.0 * len(user_idx) / max(n_items, 1)
            )
            use_w = self.ease_use_weights == "on" or (
                self.ease_use_weights == "auto" and signal_kind == "ratings"
            )
            ease_weights = None
            if use_w:
                w_mean = float(weights.mean()) if len(weights) else 1.0
                if w_mean > 0:
                    ease_weights = (weights / w_mean).astype(np.float32)
            profile["ease_weighted"] = ease_weights is not None
            t_ease = time.perf_counter()
            ease_b = np.asarray(
                kindling_core.fit_ease_py(
                    user_idx,
                    item_idx,
                    n_users=n_users,
                    n_items=n_items,
                    lambda_=eff_lambda,
                    weights=ease_weights,
                ),
                dtype=np.float32,
            )
            base_scorer_used = "ease"
            profile["ease_fit_seconds"] = time.perf_counter() - t_ease
            profile["ease_lambda"] = eff_lambda
        profile["base_scorer_used"] = base_scorer_used

        # ── Cooc weight transform (cooc path only; EASE path keeps raw cooc
        # so the <=20k datasets are byte-for-byte unchanged). Popularity-
        # normalizes the co-counts so the base stops degenerating toward a
        # popularity ranker on large catalogs.
        if base_scorer_used == "cooc" and self.cooc_base_transform != "raw":
            transform = resolve_cooc_transform(self.cooc_base_transform)
            item_counts = np.bincount(item_idx, minlength=n_items).astype(np.float64)
            cooc_data = apply_cooc_transform(
                cooc_data, cooc_indices, cooc_indptr, item_counts, n_users, transform
            )
            profile["cooc_base_transform"] = transform

        # ── Trend signal (timestamp-gated). Item interaction counts in
        # the most recent window of the training span, z-normalized
        # across the catalog.
        trend_z: np.ndarray | None = None
        if eff_trend_alpha > 0.0 and timestamps_col is not None and len(timestamps_col):
            t_hi = float(np.max(timestamps_col))
            t_lo = float(np.min(timestamps_col))
            if t_hi > t_lo:
                cut = t_hi - (t_hi - t_lo) * self.trend_window_fraction
                recent_mask = timestamps_col >= cut
                counts = np.bincount(item_idx[recent_mask], minlength=n_items_ext).astype(
                    np.float64
                )
                std = counts.std()
                if std > 0:
                    trend_z = (counts - counts.mean()) / std
                    profile["trend_window_fraction"] = self.trend_window_fraction
                    profile["trend_alpha"] = eff_trend_alpha

        # ── Sequential transition channel (timestamp-gated AND burst-
        # gated). Directional cooc over each user's timestamp-ordered
        # history; rating-burst datasets are excluded because within-
        # burst order carries no sequence information.
        trans_data: np.ndarray | None = None
        trans_indices: np.ndarray | None = None
        trans_indptr: np.ndarray | None = None
        if eff_trans_alpha > 0.0 and transition_gate_open:
            td, ti, tp = kindling_core.build_directional_cooc(
                user_idx,
                item_idx,
                weights,
                n_sessions=n_users,
                n_items=n_items,
                timestamps=timestamps_col,
            )
            trans_data = np.asarray(td, dtype=np.float32)
            trans_indices = np.asarray(ti, dtype=np.int32)
            trans_indptr = np.asarray(tp, dtype=np.int32)
            profile["transition_channel_active"] = True
            profile["transition_alpha"] = eff_trans_alpha
        else:
            profile["transition_channel_active"] = False

        # ── Content channel (metadata-gated, opt-in). Generic schema-
        # inferring feature extraction; contribution is cold-gated per
        # item at blend time so warm ranking is never diluted.
        # Built whenever EITHER the warm blend (content_alpha>0) OR the
        # reserved cold slots (cold_slots>0) need it — cold slots rank
        # their candidates by content similarity, so `cold_slots=1` with
        # `content_alpha=0` must still build features (else the slot has
        # no signal and silently no-ops).
        content_features = None
        content_coldness: np.ndarray | None = None
        _need_features = (
            self.content_alpha > 0.0 or self.cold_slots > 0 or self.metadata_smoothing != "off"
        )
        if _need_features and item_metadata is not None:
            from kindling.item_features import ItemFeatureExtractor

            content_features = ItemFeatureExtractor().fit_transform(
                item_metadata, item_to_idx, n_items_ext
            )
            item_counts = np.bincount(item_idx, minlength=n_items_ext).astype(np.float64)
            content_coldness = np.clip(
                1.0 - item_counts / float(self.content_warmth_threshold), 0.0, 1.0
            )
            profile["content_channel_active"] = content_features.n_features > 0
            profile["content_n_features"] = content_features.n_features
            profile["content_coverage"] = content_features.coverage
            profile["content_specs"] = [
                f"{s.column}:{s.kind}({s.n_features})" for s in content_features.specs
            ]
        else:
            profile["content_channel_active"] = False

        # ── Metadata smoothing of the active base (optional, default off).
        # Augment the base item-item matrix (cooc CSR or EASE) with a
        # metadata-kNN graph whose weights are the fitted prediction of the
        # base's own weight from metadata similarity. Self-scaling per base,
        # self-gating on dead metadata. Fit-time cost scales with the catalog.
        if (
            self.metadata_smoothing != "off"
            and content_features is not None
            and content_features.n_features > 0
        ):
            import scipy.sparse as _sp

            from kindling.graph.metadata_smoothing import smoothing_graph

            target_base = "ease" if base_scorer_used == "ease" else "cooc"
            if self.metadata_smoothing in ("on", "auto") or self.metadata_smoothing == target_base:
                feat = _sp.csr_matrix(
                    (content_features.data, content_features.indices, content_features.indptr),
                    shape=(n_items_ext, content_features.n_features),
                )[:n_items]
                is_repeat = signal_kind == "counts"
                t_sm = time.perf_counter()
                sm_cap = self.metadata_smoothing_cap
                if target_base == "ease" and ease_b is not None:
                    eb = ease_b
                    m_sm, sm_info = smoothing_graph(
                        feat,
                        lambda ei, ej: eb[ei, ej],
                        n_items,
                        topk=self.metadata_smoothing_topk,
                        family=self.metadata_smoothing_family,
                        is_repeat=is_repeat,
                        cap=sm_cap,
                        base_max=float(eb.max()) if eb.size else 0.0,
                    )
                    if sm_info["applied"]:
                        mco = m_sm.tocoo()
                        ease_b[mco.row, mco.col] += mco.data.astype(np.float32)
                else:
                    cooc_csr = _sp.csr_matrix(
                        (cooc_data, cooc_indices, cooc_indptr), shape=(n_items, n_items)
                    )
                    m_sm, sm_info = smoothing_graph(
                        feat,
                        lambda ei, ej: np.asarray(cooc_csr[ei, ej]).ravel(),
                        n_items,
                        topk=self.metadata_smoothing_topk,
                        family=self.metadata_smoothing_family,
                        is_repeat=is_repeat,
                        cap=sm_cap,
                        base_max=float(cooc_data.max()) if cooc_data.size else 0.0,
                    )
                    if sm_info["applied"]:
                        aug = (cooc_csr + m_sm).tocsr()
                        cooc_data = aug.data.astype(np.float32)
                        cooc_indices = aug.indices.astype(np.int32)
                        cooc_indptr = aug.indptr.astype(np.int32)
                sm_info["base"] = target_base
                sm_info["seconds"] = round(time.perf_counter() - t_sm, 2)
                profile["metadata_smoothing"] = sm_info

        # ── Cold-slot release recency (schema-inferred). Find the first
        # metadata column whose values are majority-parseable datetimes;
        # recency = exp(−days_before_train_end / 180), 0 when unknown.
        cold_recency: np.ndarray | None = None
        if (self.cold_slots > 0 or self.cold_recency_beta > 0) and item_metadata is not None:
            # `errors="coerce"` + scalar→Series so an out-of-range epoch
            # (bad/0/huge timestamp) yields NaT rather than raising.
            ref_end = None
            if timestamps_col is not None and len(timestamps_col):
                _re = pd.to_datetime(
                    pd.Series([float(np.max(timestamps_col))]),
                    unit="s",
                    errors="coerce",
                ).iloc[0]
                ref_end = None if pd.isna(_re) else _re
            for col_name in item_metadata.columns:
                if col_name == "item_id":
                    continue
                col = item_metadata[col_name]
                if pd.api.types.is_numeric_dtype(col):
                    continue
                sample = col.dropna().iloc[:200]
                if len(sample) < 10:
                    continue
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().mean() < 0.5:
                    continue
                all_parsed = pd.to_datetime(
                    item_metadata[col_name], errors="coerce", format="mixed"
                )
                ref = ref_end if ref_end is not None else all_parsed.max()
                cold_recency = np.zeros(n_items_ext)
                for iid, rd in zip(item_metadata["item_id"], all_parsed):
                    ix = item_to_idx.get(iid)
                    if ix is None or pd.isna(rd):
                        continue
                    days = (ref - rd).days
                    if days >= -30:  # tolerate slight clock skew; drop bad parses
                        cold_recency[ix] = np.exp(-max(days, 0) / 180.0)
                profile["cold_recency_column"] = col_name
                break

        # ── User-user CF channel (history-gated). Build the item→user
        # inverted CSR once at fit; neighbors are computed per recommend.
        uu_users_data: np.ndarray | None = None
        uu_users_indptr: np.ndarray | None = None
        uu_user_deg: np.ndarray | None = None
        user_counts_for_gate = np.bincount(user_idx, minlength=n_users)
        median_history = (
            float(np.median(user_counts_for_gate[user_counts_for_gate > 0]))
            if (user_counts_for_gate > 0).any()
            else 0.0
        )
        profile["median_items_per_user"] = median_history
        user_cf_open = self.user_cf_alpha > 0.0 and median_history <= self.user_cf_history_gate
        if user_cf_open:
            # Binarize (unique user-item pairs), bucket users by item.
            pair_key = user_idx.astype(np.int64) * n_items + item_idx.astype(np.int64)
            uniq = np.unique(pair_key)
            uu_u = (uniq // n_items).astype(np.int64)
            uu_i = (uniq % n_items).astype(np.int64)
            order = np.argsort(uu_i, kind="stable")
            uu_users_data = uu_u[order]
            uu_users_indptr = np.zeros(n_items + 1, dtype=np.int64)
            np.add.at(uu_users_indptr, uu_i + 1, 1)
            uu_users_indptr = np.cumsum(uu_users_indptr)
            uu_user_deg = np.bincount(uu_u, minlength=n_users).astype(np.float64)
            # user-row → unique item indices (for neighbor voting).
            u_order = np.argsort(uu_u, kind="stable")
            sorted_u = uu_u[u_order]
            sorted_i = uu_i[u_order]
            bounds = np.searchsorted(sorted_u, np.arange(n_users + 1))
            user_row_items = {
                u: sorted_i[bounds[u] : bounds[u + 1]]
                for u in range(n_users)
                if bounds[u + 1] > bounds[u]
            }
        else:
            user_row_items = {}
        profile["user_cf_channel_active"] = bool(user_cf_open)

        # ── Boost layers. Each gets its own cooc-shaped adjacency —
        # i.e., a full DUPLICATE of the item-item CSR. On very large
        # catalogs that duplication is multi-GB for layers that have
        # never shown lift beyond the fused channels; gate them off.
        # (24GB machine + 360k-item amazon-book: base cooc + temporal
        # cooc + session structures together exceed physical RAM.)
        boost_layers_size_ok = n_items <= 100_000
        if not boost_layers_size_ok:
            profile["boost_layers_skipped"] = (
                f"catalog too large ({n_items} items > 100k): duplicate adjacency builds gated off"
            )
        boost_adj: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # temporal_cooccurrence is just cooc with hybrid_temporal kernel — only
        # built when timestamps present and not rating-burst.
        if (
            boost_layers_size_ok
            and "temporal_cooccurrence" in plan["enabled_boost_layers"]
            and timestamps_col is not None
        ):
            td, ti, tp = kindling_core.build_cooccurrence(
                user_idx,
                item_idx,
                weights,
                n_users=n_users,
                n_items=n_items,
                kernel="hybrid_temporal",
                alpha=1.0,
                half_life_days=plan["half_life_days"],
                timestamps=timestamps_col,
            )
            boost_adj["temporal_cooccurrence"] = (
                np.asarray(td, dtype=np.float32),
                np.asarray(ti, dtype=np.int32),
                np.asarray(tp, dtype=np.int32),
            )
        # session_cooccurrence is built on session_id → item bipartite.
        if (
            boost_layers_size_ok
            and "session_cooccurrence" in plan["enabled_boost_layers"]
            and "session_id" in interactions.columns
        ):
            session_ids = pd.Index(interactions["session_id"].unique())
            session_to_idx = {s: i for i, s in enumerate(session_ids)}
            session_idx = interactions["session_id"].map(session_to_idx).to_numpy(dtype=np.int64)
            sd, si, spt = kindling_core.build_session_cooccurrence(
                session_idx,
                item_idx,
                weights,
                n_sessions=len(session_ids),
                n_items=n_items,
                kernel="pure_count",
            )
            boost_adj["session_cooccurrence"] = (
                np.asarray(sd, dtype=np.float32),
                np.asarray(si, dtype=np.int32),
                np.asarray(spt, dtype=np.int32),
            )

        # NOTE: item_cosine is no longer a boost layer — it IS the v2 base
        # (built earlier in this fit). Boost layers are now reserved for
        # *refinements* on different axes (temporal, session, path).

        # ── path_tail + path_basket: infer sessions, build indices.
        # Build is plan-aware: skip basket on rating-burst datasets, etc.
        # The Rust score_many kernels run at recommend time; here we just
        # feed the Python orchestrators that own session walking.
        tail_index: TailIndex | None = None
        basket_index: BasketIndex | None = None
        try:
            sess_inf = infer_sessions(interactions)
            sessions = list(sessions_from_interactions(interactions, sess_inf.session_ids))
            if sessions:
                tail_index = build_tail_index(sessions)
                # Skip basket_index when sessions are too shallow — it's
                # the heavyweight build and produces noise on rating-
                # burst datasets.
                deep_session_fraction = profile.get("deep_session_fraction", 0.0)
                if deep_session_fraction >= 0.30:
                    basket_index = build_basket_index(sessions)
        except Exception as exc:  # pragma: no cover — defensive; sessions are optional
            import warnings

            warnings.warn(
                f"path-family fit skipped ({exc!r}); v2 falls back to cooc-only retrieval.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._state = EngineState(
            item_ids=np.asarray(item_ids, dtype=object),
            item_to_idx=item_to_idx,
            n_items=n_items_ext,
            n_train_items=n_items,
            owned_by_entity=owned_by_entity,
            entity_to_user_idx=entity_to_user_idx,
            n_users=n_users,
            tail_index=tail_index,
            basket_index=basket_index,
            history_by_entity=history_by_entity,
            kernel=plan["kernel"],
            half_life_days=plan["half_life_days"],
            enabled_boost_layers=list(boost_adj.keys()),
            cooc_data=cooc_data,
            cooc_indices=cooc_indices,
            cooc_indptr=cooc_indptr,
            ease_b=ease_b,
            base_scorer_used=base_scorer_used,
            trend_z=trend_z,
            trend_alpha=eff_trend_alpha,
            item_popularity=np.bincount(item_idx, minlength=n_items_ext).astype(np.float64),
            trans_data=trans_data,
            trans_indices=trans_indices,
            trans_indptr=trans_indptr,
            transition_alpha=eff_trans_alpha,
            content_features=content_features,
            content_coldness=content_coldness,
            cold_recency=cold_recency,
            cold_recency_beta=self.cold_recency_beta,
            content_alpha=self.content_alpha,
            last_item_alpha=self.last_item_alpha,
            uu_users_data=uu_users_data,
            uu_users_indptr=uu_users_indptr,
            uu_user_deg=uu_user_deg,
            user_row_items=user_row_items,
            user_cf_alpha=self.user_cf_alpha if user_cf_open else 0.0,
            user_cf_k=self.user_cf_k,
            transition_last_k=self.transition_last_k,
            transition_decay=self.transition_decay,
            signal_kind=signal_kind,
            boost_layer_adjacencies=boost_adj,
            z_threshold=2.5,
            boost_multiplier=3.0,
            fit_seconds=time.perf_counter() - t0,
            profile=profile,
        )
        return self

    @property
    def activation_plan(self) -> ActivationPlan:
        """Which layers were activated for this dataset, and why.

        A self-explaining record of the regime-based gating (base scorer,
        per-channel on/off with reasons, cold-start). Inspect after fit:
        ``print(engine.activation_plan.summary())``.
        """
        if self._state is None:
            raise RuntimeError("Engine not fitted. Call .fit(interactions) first.")
        from kindling.activation import build_activation_plan

        return build_activation_plan(self, self._state.profile)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist this fitted engine to ``path``. See ``kindling.persist``."""
        from kindling.persist import save_engine

        save_engine(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Engine:
        """Load an engine previously written by :meth:`save`."""
        from kindling.persist import load_engine

        return load_engine(path)

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    def recommend(self, entity_id: object, n: int = 10) -> list[Recommendation]:
        if self._state is None:
            raise RuntimeError("Engine not fitted. Call .fit(interactions) first.")
        owned = self._state.owned_by_entity.get(entity_id)
        if owned is None or owned.size == 0:
            return []
        return self._recommend_core(owned, entity_id, n)

    def recommend_for_items(
        self, seed_item_ids: Iterable[object], n: int = 10
    ) -> list[Recommendation]:
        """Serve a NEW / anonymous user from ad-hoc seed items — no per-user
        training, no stored history required. The closed-form base (EASE/cooc)
        scores from *any* seed set, so a brand-new user who just interacted
        with a few items gets personalized recommendations immediately. Seeds
        not in the catalog are dropped; an empty/all-unknown seed set falls back
        to popularity (`_cold_recommend`) — the benchmark's cold-data champion.
        This is the cold-USER serving path (§7.4); cf. `recommend` for known
        entities.
        """
        if self._state is None:
            raise RuntimeError("Engine not fitted. Call .fit(interactions) first.")
        st = self._state
        seen: set[int] = set()
        owned_list: list[int] = []
        for it in seed_item_ids:
            ix = st.item_to_idx.get(it)
            if ix is not None and ix not in seen:
                seen.add(ix)
                owned_list.append(ix)
        if not owned_list:
            return self._cold_recommend(n)
        # Empirical-Bayes shrinkage: lean on the popularity prior when seeds are
        # few, the personalized signal as they accumulate.
        pop_prior = self.cold_user_pop_prior / len(owned_list)
        return self._recommend_core(
            np.asarray(owned_list, dtype=np.int64), None, n, pop_prior=pop_prior
        )

    def _cold_recommend(self, n: int) -> list[Recommendation]:
        """Zero-history fallback for brand-new users with no seeds: top-n by
        all-time item popularity (the warming benchmark's cold-data champion —
        it beats recent-trend for a zero-info user), falling back to the trend
        vector only when popularity is unavailable. Popularity is unbeatable in
        the data-starved regime, so it is the right zero-seed prior."""
        st = self._state
        assert st is not None
        scores = st.item_popularity if st.item_popularity is not None else st.trend_z
        if scores is None or scores.size == 0:
            return []
        n_eff = min(n, scores.size)
        top = np.argpartition(-scores, n_eff - 1)[:n_eff]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [
            Recommendation(
                item_id=st.item_ids[int(c)],
                score=float(scores[c]),
                base_kind="cold_popularity",
            )
            for c in top
        ]

    def _recommend_core(
        self, owned: np.ndarray, entity_id: object, n: int, pop_prior: float = 0.0
    ) -> list[Recommendation]:
        st = self._state
        assert st is not None

        # New-user popularity shrinkage: a length-n_items addend
        # pop_prior · z(log popularity), added to the blended full-catalog
        # score so a thin-seed ranking leans on the popularity prior. Scalar 0
        # (no-op) for known users (recommend passes pop_prior=0).
        pop_addend: np.ndarray | float = 0.0
        if pop_prior > 0.0 and st.item_popularity is not None:
            p = np.log1p(st.item_popularity.astype(np.float64))
            sd = p.std()
            pop_addend = pop_prior * ((p - p.mean()) / sd) if sd > 0 else 0.0

        # ── 1. Retrieve candidate pool. When EASE is the base scorer it
        # powers retrieval too — the gap-decomposition diagnostic showed
        # the raw-cooc retrieval/scoring tautology caps achievable
        # quality, so the better signal must drive the pool as well.
        ease_scores_full: np.ndarray | None = None
        if st.ease_b is not None:
            base_vec = st.ease_b[owned].sum(axis=0, dtype=np.float64)
            if base_vec.size < st.n_items:
                # Open-catalog extension items: no EASE evidence → 0.
                base_vec = np.concatenate([base_vec, np.zeros(st.n_items - base_vec.size)])
            ease_scores_full = self._blend_channels(
                st,
                owned,
                base_vec,
                user_row=st.entity_to_user_idx.get(entity_id, -1),
            )
            ease_scores_full = ease_scores_full + pop_addend
            ease_scores_full[owned] = -np.inf
            budget = min(self.retrieval_budget, ease_scores_full.size)
            top = np.argpartition(-ease_scores_full, budget - 1)[:budget]
            top = top[np.argsort(-ease_scores_full[top], kind="stable")]
            cand_ids = [int(c) for c in top if np.isfinite(ease_scores_full[c])]
            base_kind = "ease"
        elif (st.trend_z is not None and st.trend_alpha > 0.0) or (
            st.trans_data is not None and st.transition_alpha > 0.0
        ):
            # Fused cooc path (large catalogs where the EASE inversion is
            # gated off). Same channel blend over the full-catalog cooc
            # vector. Because cooc is SPARSE — zero score for any item
            # the history has no co-occurrence edge to — taking the pool
            # from the blended scores is genuine retrieval fusion: trend/
            # transition channels can promote items into the pool that
            # the cooc retriever alone would never surface.
            cooc_full = np.zeros(st.n_items, dtype=np.float64)
            for item in owned.tolist():
                s_ = int(st.cooc_indptr[item])
                e_ = int(st.cooc_indptr[item + 1])
                if e_ > s_:
                    cooc_full[st.cooc_indices[s_:e_]] += st.cooc_data[s_:e_]
            ease_scores_full = self._blend_channels(
                st,
                owned,
                cooc_full,
                user_row=st.entity_to_user_idx.get(entity_id, -1),
            )
            ease_scores_full = ease_scores_full + pop_addend
            ease_scores_full[owned] = -np.inf
            budget = min(self.retrieval_budget, ease_scores_full.size)
            top = np.argpartition(-ease_scores_full, budget - 1)[:budget]
            top = top[np.argsort(-ease_scores_full[top], kind="stable")]
            cand_ids = [int(c) for c in top if np.isfinite(ease_scores_full[c])]
            base_kind = "cooc_fused"
        else:
            cand_ids, _scores = kindling_core.cooccurrence_retrieve(
                st.cooc_data,
                st.cooc_indices,
                st.cooc_indptr,
                owned_indices=owned.tolist(),
                budget=self.retrieval_budget,
                include_owned=False,
            )
            cand_ids = list(cand_ids)
            base_kind = ""
        if not cand_ids:
            return []

        # ── 2. Base scores. Fused paths (ease / cooc_fused) carry their
        # own pool scores; otherwise fall back to the cooc base.
        if ease_scores_full is not None:
            base = ease_scores_full[np.asarray(cand_ids, dtype=np.int64)]
        else:
            base_kind, base = self._compute_base(entity_id, owned, cand_ids)

        # ── 3. Layered scoring.
        layer_specs = self._build_layer_specs(entity_id, owned, cand_ids)
        composite = kindling_core.layered_score_py(
            base,
            layer_specs,
            z_threshold=st.z_threshold,
            boost_multiplier=st.boost_multiplier,
        )
        composite = np.asarray(composite)

        # ── 4. Top-N (skip repeat module — not yet ported).
        # The positivity filter encodes cooc semantics (score 0 = no
        # co-occurrence evidence). EASE weights are signed, so a small
        # negative composite can still be the best available candidate.
        order = np.argsort(-composite)[:n]
        out: list[Recommendation] = []
        for rank, idx in enumerate(order):
            if composite[idx] <= 0.0 and base_kind not in ("ease", "cooc_fused"):
                continue
            cid = cand_ids[idx]
            out.append(
                Recommendation(
                    item_id=st.item_ids[cid],
                    score=float(composite[idx]),
                    base_kind=base_kind,
                )
            )

        # ── Reserved cold slots ("new releases shelf"). Cold items can
        # never out-z the warm army in one blended ranking no matter the
        # content weight (steam: rank-136-of-20k cold candidates vs 500
        # warm EASE candidates for 10 slots). Reserving the final slots
        # for the top cold-content candidates trades a sliver of warm
        # accuracy for cold-start coverage — steam: cold-event recovery
        # 0% → 5.1% for −0.6% aggregate NDCG at cold_slots=1.
        if (
            self.cold_slots > 0
            and st.content_features is not None
            and st.content_coldness is not None
            and st.content_features.n_features > 0
        ):
            # Content-space ranker: cosine to the user's owned-item content.
            from kindling.item_features import content_scores

            cs = content_scores(st.content_features, owned)
            std = cs.std()
            if std > 0:
                cs = (cs - cs.mean()) / std
            if st.cold_recency is not None and st.cold_recency_beta > 0:
                # New releases dominate real cold purchases; recency
                # reorders only within the cold slot (warm untouched).
                cs = cs + st.cold_recency_beta * st.cold_recency
            cs[st.content_coldness < 0.75] = -np.inf  # warm items excluded
            cs[owned] = -np.inf
            kept_ids = {st.item_to_idx.get(r.item_id, -1) for r in out}
            cold_picks: list[int] = []
            for i in np.argsort(-cs):
                if not np.isfinite(cs[i]):
                    break
                if int(i) not in kept_ids:
                    cold_picks.append(int(i))
                    if len(cold_picks) >= self.cold_slots:
                        break
            if cold_picks:
                out = out[: max(n - len(cold_picks), 0)]
                for i in cold_picks:
                    out.append(
                        Recommendation(
                            item_id=st.item_ids[i],
                            score=float(cs[i]),
                            base_kind="cold_content",
                        )
                    )
        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _blend_channels(
        self,
        st: EngineState,
        owned: np.ndarray,
        scores_full: np.ndarray,
        user_row: int = -1,
    ) -> np.ndarray:
        """z-normalize a full-catalog base vector and add the active
        channels (trend / last-item / transitions / content / user-CF).

        Channels are independent of which base produced `scores_full`
        (EASE or cooc), so both paths share this blend. `user_row` is
        the engine user index, needed only by the user-CF channel for
        self-exclusion.
        """
        trend_on = st.trend_z is not None and st.trend_alpha > 0.0
        trans_on = st.trans_data is not None and st.transition_alpha > 0.0
        content_on = (
            st.content_features is not None
            and st.content_alpha > 0.0
            and st.content_features.n_features > 0
        )
        last_on = st.ease_b is not None and st.last_item_alpha > 0.0 and owned.size > 0
        uu_on = st.uu_users_data is not None and st.user_cf_alpha > 0.0 and owned.size > 0
        if not (trend_on or trans_on or content_on or last_on or uu_on):
            return scores_full
        std = scores_full.std()
        if std > 0:
            scores_full = (scores_full - scores_full.mean()) / std
        if trend_on:
            assert st.trend_z is not None  # narrowed by trend_on
            scores_full = scores_full + st.trend_alpha * st.trend_z
        if uu_on:
            # Otsuka-Ochiai k-NN: overlap counts via the inverted index,
            # normalized by sqrt(deg_u · deg_v); top-k neighbors vote
            # for their items, weighted by similarity.
            assert st.uu_user_deg is not None
            assert st.uu_users_indptr is not None
            assert st.uu_users_data is not None
            n_users_total = st.uu_user_deg.shape[0]
            counts = np.zeros(n_users_total, dtype=np.float64)
            for i in owned.tolist():
                s_ = int(st.uu_users_indptr[i])
                e_ = int(st.uu_users_indptr[i + 1])
                if e_ > s_:
                    counts[st.uu_users_data[s_:e_]] += 1.0
            if 0 <= user_row < n_users_total:
                counts[user_row] = 0.0
            nz = np.nonzero(counts)[0]
            if nz.size > 0:
                sims = counts[nz] / (np.sqrt(st.uu_user_deg[nz]) * np.sqrt(max(owned.size, 1)))
                # Deterministic top-k: similarity desc, ties broken by
                # ascending user row (nz is ascending, and a stable sort
                # preserves that order among equal sims). argpartition's tie
                # order is unspecified and not portable, so it cannot be
                # byte-matched by the Rust core; a stable secondary key makes
                # the neighbor set reproducible across implementations.
                if nz.size > st.user_cf_k:
                    keep = np.argsort(-sims, kind="stable")[: st.user_cf_k]
                else:
                    keep = np.arange(nz.size)
                # Neighbors vote: accumulate their item sets. Inverted
                # again via the per-user owned arrays would need a
                # second index; instead vote through the entity map.
                uu_vec = np.zeros(st.n_items, dtype=np.float64)
                neighbor_rows = nz[keep]
                neighbor_sims = sims[keep]
                for v_row, sim in zip(neighbor_rows.tolist(), neighbor_sims.tolist()):
                    v_items = st.user_row_items.get(int(v_row))
                    if v_items is not None:
                        uu_vec[v_items] += sim
                u_std = uu_vec.std()
                if u_std > 0:
                    scores_full = scores_full + st.user_cf_alpha * (
                        (uu_vec - uu_vec.mean()) / u_std
                    )
        if last_on:
            assert st.ease_b is not None  # narrowed by last_on
            last_row = st.ease_b[int(owned[-1])].astype(np.float64)
            if last_row.size < st.n_items:
                last_row = np.concatenate([last_row, np.zeros(st.n_items - last_row.size)])
            l_std = last_row.std()
            if l_std > 0:
                scores_full = scores_full + st.last_item_alpha * (
                    (last_row - last_row.mean()) / l_std
                )
        if content_on:
            from kindling.item_features import content_scores

            cs = content_scores(st.content_features, owned)
            c_std = cs.std()
            if c_std > 0:
                cz = (cs - cs.mean()) / c_std
                coldness = st.content_coldness if st.content_coldness is not None else 1.0
                scores_full = scores_full + st.content_alpha * coldness * cz
        if trans_on:
            assert st.trans_indptr is not None
            assert st.trans_indices is not None
            assert st.trans_data is not None
            trans = np.zeros(st.n_items, dtype=np.float64)
            recent = owned[::-1][: st.transition_last_k]
            for j, item in enumerate(recent):
                s_ = int(st.trans_indptr[item])
                e_ = int(st.trans_indptr[item + 1])
                if e_ > s_:
                    trans[st.trans_indices[s_:e_]] += (st.transition_decay**j) * st.trans_data[
                        s_:e_
                    ]
            t_std = trans.std()
            if t_std > 0:
                scores_full = scores_full + st.transition_alpha * ((trans - trans.mean()) / t_std)
        return scores_full

    def _compute_base(
        self, entity_id: object, owned: np.ndarray, cand_ids: list[int]
    ) -> tuple[str, np.ndarray]:
        """Cooc base scores over the candidate pool."""
        st = self._state
        assert st is not None
        base = kindling_core.cooccurrence_signal(
            st.cooc_data,
            st.cooc_indices,
            st.cooc_indptr,
            owned_indices=owned.tolist(),
            candidate_indices=cand_ids,
        )
        return "cooc", np.asarray(base)

    def _build_layer_specs(
        self,
        entity_id: object,
        owned: np.ndarray,
        cand_ids: list[int],
    ) -> list[tuple[np.ndarray, str]]:
        """Build (layer_scores, z_mode) tuples for the layered scorer.

        Pinned to v1 layered's canonical set
        [path_basket, session_cooccurrence, temporal_cooccurrence] so v2
        and v1-with-layered_scoring are an apples-to-apples comparison.
        Each layer is sparse / "nonzero" z-mode.

        path_tail and item_cosine were experimented with as additional
        layers; both produced no measurable lift over the canonical set
        and are intentionally excluded here so consolidation parity is
        clean.
        """
        st = self._state
        assert st is not None
        out: list[tuple[np.ndarray, str]] = []

        # Cooc-shaped sparse layers (temporal_cooc, session_cooc).
        for layer_name in st.enabled_boost_layers:
            adj = st.boost_layer_adjacencies.get(layer_name)
            if adj is None:
                continue
            data, indices, indptr = adj
            scores = kindling_core.cooccurrence_signal(
                data,
                indices,
                indptr,
                owned_indices=owned.tolist(),
                candidate_indices=cand_ids,
            )
            out.append((np.asarray(scores), "nonzero"))

        # path_basket: sparse, queries against the user's recent history.
        if st.basket_index is not None and st.basket_index.observations:
            history = st.history_by_entity.get(entity_id, ())
            if history:
                # Use the most recent ~50 items as the query basket.
                query_basket = frozenset(history[-50:])
                cand_external = [st.item_ids[ci] for ci in cand_ids]
                basket_scores = st.basket_index.score_many(cand_external, query_basket=query_basket)
                out.append((np.asarray(basket_scores, dtype=np.float64), "nonzero"))

        return out

    def _profile(
        self,
        interactions: pd.DataFrame,
        weights: np.ndarray,
        n_users: int,
        n_items: int,
    ) -> dict[str, Any]:
        density = (
            float(len(interactions)) / max(n_users * n_items, 1) if n_users and n_items else 0.0
        )
        has_timestamps = "timestamp" in interactions.columns
        has_sessions = "session_id" in interactions.columns
        avg_per_user = float(len(interactions)) / max(n_users, 1)

        # Crude rating-burst detection: median inter-event delta < 300s ⇒ burst.
        rating_burst = False
        median_delta_seconds = None
        if has_timestamps:
            diffs: list[float] = []
            for entity, grp in interactions.groupby("entity_id", sort=False):
                ts = grp["timestamp"].to_numpy(dtype=np.float64)
                if ts.size > 1:
                    diffs.extend(np.diff(np.sort(ts)).tolist())
            if diffs:
                median_delta_seconds = float(np.median(diffs))
                rating_burst = median_delta_seconds < 300.0
        # Deep-session check.
        deep_session_fraction = 0.0
        if has_sessions:
            session_ids = pd.Index(interactions["session_id"].unique())
            session_to_idx = {s: i for i, s in enumerate(session_ids)}
            sidx = interactions["session_id"].map(session_to_idx).to_numpy(dtype=np.int64)
            iidx = (
                interactions["item_id"]
                .map({k: i for i, k in enumerate(pd.Index(interactions["item_id"].unique()))})
                .to_numpy(dtype=np.int64)
            )
            deep_session_fraction = float(
                kindling_core.deep_session_fraction(sidx.tolist(), iidx.tolist())
            )

        return {
            "n_users": n_users,
            "n_items": n_items,
            "n_interactions": len(interactions),
            "density": density,
            "has_timestamps": has_timestamps,
            "has_sessions": has_sessions,
            "avg_per_user": avg_per_user,
            "rating_burst_detected": rating_burst,
            "median_delta_seconds": median_delta_seconds,
            "deep_session_fraction": deep_session_fraction,
        }

    def _plan(self, profile: dict[str, Any]) -> dict[str, Any]:
        """LayerPlan: decisions from profile (PRD §"Profile → Plan contract")."""
        # Single decay knob: half-life decays with time-density.
        # Default ~30 days for moderate density; longer for sparser data.
        half_life_days = 30.0
        # Kernel choice.
        kernel = (
            "pure_count"
            if profile["rating_burst_detected"]
            else ("hybrid_temporal" if profile["has_timestamps"] else "pure_count")
        )
        # Boost layer enable/disable per profile.
        enabled = []
        if profile["has_timestamps"] and not profile["rating_burst_detected"]:
            enabled.append("temporal_cooccurrence")
        if profile["has_sessions"] and profile["deep_session_fraction"] >= 0.30:
            enabled.append("session_cooccurrence")
        # path_tail / path_basket / interaction_network / cosine /
        # lightgcn deferred until their builders / scorers are wired here.
        return {
            "kernel": kernel,
            "alpha": 1.0,
            "half_life_days": half_life_days,
            "enabled_boost_layers": enabled,
        }

    # Diagnostics / introspection
    def fit_summary(self) -> dict[str, Any]:
        if self._state is None:
            return {"fitted": False}
        st = self._state
        return {
            "fitted": True,
            "fit_seconds": st.fit_seconds,
            "n_users": st.n_users,
            "n_items": st.n_items,
            "kernel": st.kernel,
            "half_life_days": st.half_life_days,
            "enabled_boost_layers": st.enabled_boost_layers,
            "z_threshold": st.z_threshold,
            "boost_multiplier": st.boost_multiplier,
            "profile": st.profile,
            "signal_kind": st.signal_kind,
            "base_scorer_used": st.base_scorer_used,
            "base_scorer_setting": self.base_scorer,
        }
