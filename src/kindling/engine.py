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

import math
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
    # Repeat module: per-user CSR of reorder items re-surfaced via the timing
    # multiplier on repeat-regime datasets (gated by repeat_rate). Active only
    # when repeat_active; the native recommend exempts these from the owned-mask.
    repeat_active: bool = False
    repeat_rate: float = 0.0
    repeat_indptr: np.ndarray | None = None
    repeat_items: np.ndarray | None = None
    repeat_counts: np.ndarray | None = None
    repeat_last_ts: np.ndarray | None = None
    repeat_periods: np.ndarray | None = None
    repeat_quality: np.ndarray | None = None
    repeat_now_ts: float = float("nan")
    repeat_freq_alpha: float = 0.0
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
        # Opt-in held-out λ search (leave-last-out, recall@10) over a grid around
        # the heuristic. Default OFF: investigation (docs/EASE-LAMBDA.md) found
        # the heuristic is already EASE-optimal — a held-out search independently
        # lands on ~1× across last-1/3/10% held-outs and recall/NDCG — so the
        # search *confirms* rather than improves it, and defaulting it on would
        # triple fit time for no gain. The earlier "4× too high" was a
        # random-split-protocol artifact (that regime wants ~0.2× the heuristic).
        # Useful as opt-in for off-distribution data where the heuristic may be
        # off; needs timestamps, gated by catalog size for cost. "auto" =
        # search when affordable (≤8k items); True/False force it.
        ease_lambda_search: bool | str = False,
        # EASE+ denoising (EDLAE): δ adds a popularity-proportional penalty
        # δ·diag(G) to the EASE ridge, correcting the dropout-free autoencoder's
        # train/serve mismatch. Default 0.0 = canonical EASE (opt-in only). EASE+
        # is non-universal — δ=0.5 lifts ml1m/beauty/tafeng (+1.2–5%) but regresses
        # steam (−1.7%), and the held-out "auto" δ search can't reliably tell them
        # apart (leave-last-out recall is a noisy proxy for the chronological
        # NDCG eval), so it ships off. Set a fixed float for EASE+ on data where
        # it helps, or "auto" for the (caveated) held-out δ search. Gated by size.
        ease_denoise: float | str = 0.0,
        # Held-out channel-activation gate: backward-eliminate a blend channel
        # (trend / user_cf / last_item / transition) when removing it strictly
        # improves a leave-last-out held-out (recall@10). Channels are tuned for
        # temporal/sequential signal; on data without it they can be net-negative
        # (a random-split ml1m ablation cost −0.017 NDCG). The gate auto-disables
        # the offenders out of the box — and never fires on data where the
        # channels help (strict-improvement criterion), so it can't regress the
        # temporal benchmarks. "auto" = gate when affordable (≤20k items);
        # True/False force it. EASE base only.
        channel_gate: bool | str = "auto",
        # Catalog-size gate for the EASE base: above this, "auto" falls back to
        # wilson-cooc. The wall is the dense n×n Gram + inverse (O(n²) memory,
        # O(n³) compute) — ~3GB at 20k, ~20GB at 50k, ~63GB at 88k. 20k is a safe
        # default for a typical box; raise it on a larger-memory machine to push
        # EASE onto bigger catalogs (e.g. ease_max_items=50_000 ≈ 20GB Gram).
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
        # Repeat-consumption recommendation. On repeat-regime datasets (grocery,
        # replenishment), the seen-item mask hides the user's reorders — exactly
        # what they'll buy next. "auto" builds the reorder profile when the fit's
        # repeat_rate (fraction of duplicate user-item interactions) exceeds the
        # low repeat_min_rate pre-filter, then a held-out repeat gate keeps it
        # only if recommending reorders strictly improves a leave-last-fraction
        # repeat-aware NDCG@10 — so it auto-declines on fake-repeat data (e.g.
        # steam: re-logs are duplicates but freq-reordering ranks worse than the
        # base), which a repeat-rate threshold alone can't tell from true
        # repurchase. True/False force it (True skips the gate). Re-surfaces via
        # the personal-frequency layer + timing multiplier (REPLENISH).
        repeat_recommend: bool | str = "auto",
        repeat_min_rate: float = 0.05,
        # Personal-frequency layer for the repeat path. When the repeat gate
        # fires, each reorder candidate is lifted by repeat_freq_alpha·log1p(count)
        # (count = how often the user bought it), timing-modulated — so frequently
        # bought items rise like the "buy it again" baseline. 0 = affinity-only
        # (old behavior). "auto" picks a robust default when the gate is on.
        repeat_freq_alpha: float | str = "auto",
        # Boost-layer (z_threshold, boost_multiplier) auto-calibration. When the
        # fit has cooc-shaped boost layers (temporal/session cooc), "auto" runs a
        # held-out (leave-last-out) grid sweep and adopts the (z, boost) cell that
        # maximizes held-out NDCG — falling back to the 2.5/3.0 defaults if no
        # cell beats them (so it never regresses). True/False force it.
        calibrate_boost: bool | str = "auto",
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
        if ease_lambda_search not in (True, False, "auto"):
            raise ValueError(
                f"ease_lambda_search must be True | False | 'auto'; got {ease_lambda_search!r}"
            )
        self.ease_lambda_search = ease_lambda_search
        if ease_denoise != "auto" and (
            not isinstance(ease_denoise, (int, float)) or ease_denoise < 0
        ):
            raise ValueError(f"ease_denoise must be >= 0 or 'auto'; got {ease_denoise!r}")
        self.ease_denoise = ease_denoise
        if channel_gate not in (True, False, "auto"):
            raise ValueError(f"channel_gate must be True | False | 'auto'; got {channel_gate!r}")
        self.channel_gate = channel_gate
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
        if repeat_recommend not in (True, False, "auto"):
            raise ValueError(
                f"repeat_recommend must be True | False | 'auto'; got {repeat_recommend!r}"
            )
        self.repeat_recommend = repeat_recommend
        if not 0.0 <= repeat_min_rate <= 1.0:
            raise ValueError(f"repeat_min_rate must be in [0, 1]; got {repeat_min_rate!r}")
        self.repeat_min_rate = float(repeat_min_rate)
        if repeat_freq_alpha != "auto" and (
            not isinstance(repeat_freq_alpha, (int, float)) or repeat_freq_alpha < 0
        ):
            raise ValueError(f"repeat_freq_alpha must be >= 0 or 'auto'; got {repeat_freq_alpha!r}")
        self.repeat_freq_alpha = repeat_freq_alpha
        if calibrate_boost not in (True, False, "auto"):
            raise ValueError(f"calibrate_boost must be True | False | 'auto'; got {calibrate_boost!r}")
        self.calibrate_boost = calibrate_boost
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
        # Native Rust recommend engine, lazily built per fit and cached. It is a
        # non-picklable PyO3 object, so it is dropped on pickle and rebuilt.
        self._native: Any = None
        self._native_built: bool = False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_native"] = None
        state["_native_built"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

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

        # ── Repeat module. repeat_rate = fraction of duplicate (entity, item)
        # interactions — the reorders the seen-mask would otherwise hide. On a
        # repeat-regime dataset, build the per-user reorder profile (count ≥ 2,
        # last_ts, repurchase period) so the native recommend can re-surface
        # due items via the timing multiplier.
        repeat_active = False
        repeat_rate = (
            float(interactions.duplicated(["entity_id", "item_id"]).mean())
            if len(interactions)
            else 0.0
        )
        repeat_indptr = repeat_items = repeat_last_ts = None
        repeat_periods = repeat_quality = repeat_counts = None
        repeat_now_ts = float("nan")
        want_repeat = self.repeat_recommend is True or (
            self.repeat_recommend == "auto" and repeat_rate >= self.repeat_min_rate
        )
        if want_repeat and hasattr(kindling_core, "fit_repeat_profile"):
            ts_arg = (
                timestamps_col
                if timestamps_col is not None and timestamps_col.size and timestamps_col.max() > timestamps_col.min()
                else None
            )
            rp = kindling_core.fit_repeat_profile(user_idx, item_idx, ts_arg, n_users, 2)
            repeat_indptr = np.asarray(rp[0], np.int64)
            repeat_items = np.asarray(rp[1], np.int64)
            repeat_counts = np.asarray(rp[2], np.float64)
            repeat_last_ts = np.asarray(rp[3], np.float64)
            repeat_periods = np.asarray(rp[4], np.float64)
            repeat_quality = np.asarray(rp[5], np.float64)
            repeat_active = repeat_items.size > 0
            if ts_arg is not None:
                repeat_now_ts = float(ts_arg.max())
        # Resolve the personal-frequency layer strength (auto → robust default).
        eff_freq_alpha = 0.0
        if repeat_active:
            # auto=50: frequency dominates the reorder ranking (the "buy it again"
            # signal), the timing multiplier modulates. Swept on dunnhumby/tafeng
            # (both beat the personal-frequency baseline at 50).
            eff_freq_alpha = (
                50.0 if self.repeat_freq_alpha == "auto" else float(self.repeat_freq_alpha)
            )

        # owned_by_entity + history (timestamp-ordered) per entity.
        # Vectorized: one stable lexsort over (entity, time) + boundary
        # split. The per-user pandas groupby this replaces dominated fit
        # time on large datasets (2.3M users on steam → 10+ minutes).
        # entity_ids is first-appearance ordered, so ascending user_idx
        # preserves the original dict insertion order.
        owned_by_entity: dict[object, np.ndarray] = {}
        if timestamps_col is not None:
            order = np.lexsort((timestamps_col, user_idx))
        else:
            order = np.lexsort((np.arange(len(user_idx)), user_idx))
        su = user_idx[order]
        si = item_idx[order]
        if len(su):
            boundaries = np.flatnonzero(np.diff(su)) + 1
            starts = np.concatenate(([0], boundaries))
            for start, items_arr in zip(starts, np.split(si, boundaries)):
                owned_by_entity[entity_ids[int(su[start])]] = items_arr

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
            use_w = self.ease_use_weights == "on" or (
                self.ease_use_weights == "auto" and signal_kind == "ratings"
            )
            ease_weights = None
            if use_w:
                w_mean = float(weights.mean()) if len(weights) else 1.0
                if w_mean > 0:
                    ease_weights = (weights / w_mean).astype(np.float32)
            profile["ease_weighted"] = ease_weights is not None
            heuristic_lambda = 20.0 * len(user_idx) / max(n_items, 1)
            fixed_delta = 0.0 if self.ease_denoise == "auto" else float(self.ease_denoise)
            if self.ease_lambda is not None:
                eff_lambda, eff_delta = self.ease_lambda, fixed_delta
            else:
                eff_lambda, eff_delta = self._resolve_ease_hparams(
                    user_idx, item_idx, timestamps_col, n_users, n_items,
                    ease_weights, heuristic_lambda, fixed_delta, profile,
                )
            profile["ease_delta"] = eff_delta
            t_ease = time.perf_counter()
            ease_b = np.asarray(
                kindling_core.fit_ease_py(
                    user_idx,
                    item_idx,
                    n_users=n_users,
                    n_items=n_items,
                    lambda_=eff_lambda,
                    weights=ease_weights,
                    delta=eff_delta,
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
        # *refinements* on different axes (temporal, session) — all cooc-shaped
        # and served by the native engine. The path-family (path_tail /
        # path_basket) was a Python-per-recommend layer incompatible with the
        # native batch path and is no longer built.

        # Auto-calibrate the boost layer's (z_threshold, boost_multiplier) by
        # held-out lift; no-op (defaults) without boost layers.
        eff_z, eff_boost = self._calibrate_boost(
            ease_b,
            (cooc_data, cooc_indices, cooc_indptr) if ease_b is None else None,
            boost_adj,
            user_idx,
            item_idx,
            timestamps_col,
            profile,
        )

        self._state = EngineState(
            item_ids=np.asarray(item_ids, dtype=object),
            item_to_idx=item_to_idx,
            n_items=n_items_ext,
            n_train_items=n_items,
            owned_by_entity=owned_by_entity,
            entity_to_user_idx=entity_to_user_idx,
            n_users=n_users,
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
            repeat_active=repeat_active,
            repeat_rate=repeat_rate,
            repeat_indptr=repeat_indptr,
            repeat_items=repeat_items,
            repeat_counts=repeat_counts,
            repeat_last_ts=repeat_last_ts,
            repeat_periods=repeat_periods,
            repeat_quality=repeat_quality,
            repeat_now_ts=repeat_now_ts,
            repeat_freq_alpha=eff_freq_alpha,
            boost_layer_adjacencies=boost_adj,
            z_threshold=eff_z,
            boost_multiplier=eff_boost,
            fit_seconds=time.perf_counter() - t0,
            profile=profile,
        )
        # Invalidate the native engine cache — rebuilt lazily for this fit.
        self._native = None
        self._native_built = False
        self._apply_channel_gate()
        self._apply_repeat_gate(user_idx, item_idx, timestamps_col)
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
        st = self._state
        owned = st.owned_by_entity.get(entity_id)
        if owned is None or owned.size == 0:
            return []
        user_row = int(st.entity_to_user_idx.get(entity_id, -1))
        return self._native_recs(self._require_native(), [int(x) for x in owned], user_row, n, 0.0)

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
        # Anonymous: user_row=-1 (no user-CF); native applies the pop_prior.
        return self._native_recs(self._require_native(), owned_list, -1, n, pop_prior)

    def _require_native(self) -> Any:
        """The native Rust recommend engine for this fit, built lazily and
        cached (the EASE-matrix copy amortizes over a batch). Raises if the
        engine can't be built — there is no Python recommend fallback."""
        if not getattr(self, "_native_built", False):
            from kindling._native_engine import build_native_engine

            self._native = build_native_engine(self)
            self._native_built = True
        if self._native is None:
            raise RuntimeError(
                "Native recommend engine unavailable — the kindling_core "
                "extension is not built, or no base scorer was fitted. Build "
                "the extension (maturin develop) or check the fit."
            )
        return self._native

    def _native_recs(
        self, native: Any, owned: list[int], user_row: int, n: int, pop_prior: float
    ) -> list[Recommendation]:
        """Map a native ``EngineState.recommend`` result to ``Recommendation``s."""
        st = self._state
        assert st is not None
        items, scores, kinds = native.recommend(owned, user_row, n, pop_prior)
        return [
            Recommendation(item_id=st.item_ids[i], score=float(s), base_kind=k)
            for i, s, k in zip(items, scores, kinds)
        ]

    def recommend_batch(
        self, entity_ids: Iterable[object], n: int = 10
    ) -> list[list[Recommendation]]:
        """Recommend for many known entities at once — the fast path.

        The per-user EASE/cooc sums, channel blends, and retrievals run
        concurrently in Rust with the GIL released — several× the per-user loop
        on the reference datasets. Entities with no history yield ``[]``,
        matching :meth:`recommend`.
        """
        if self._state is None:
            raise RuntimeError("Engine not fitted. Call .fit(interactions) first.")
        st = self._state
        ids = list(entity_ids)
        native = self._require_native()
        results: list[list[Recommendation]] = [[] for _ in ids]
        owneds: list[list[int]] = []
        user_rows: list[int] = []
        positions: list[int] = []
        for k, ent in enumerate(ids):
            owned = st.owned_by_entity.get(ent)
            if owned is None or owned.size == 0:
                continue
            owneds.append([int(x) for x in owned])
            user_rows.append(int(st.entity_to_user_idx.get(ent, -1)))
            positions.append(k)
        if owneds:
            batch = native.recommend_batch(owneds, user_rows, n, 0.0)
            for j, k in enumerate(positions):
                items, scores, kinds = batch[j]
                results[k] = [
                    Recommendation(item_id=st.item_ids[i], score=float(s), base_kind=kk)
                    for i, s, kk in zip(items, scores, kinds)
                ]
        return results

    def _resolve_ease_hparams(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        timestamps_col: np.ndarray | None,
        n_users: int,
        n_items: int,
        ease_weights: np.ndarray | None,
        heuristic: float,
        fixed_delta: float,
        profile: dict[str, Any],
    ) -> tuple[float, float]:
        """Pick EASE (λ, δ) by leave-last-out held-out (recall@10). λ defaults to
        the heuristic (20× mean Gram diagonal — already EASE-optimal); the
        ease_lambda_search opt-in sweeps a multiplicative grid. δ (EASE+/EDLAE
        denoising) defaults to a held-out search over {0, 0.5, 1.0} at the chosen
        λ, keeping δ=0 (plain EASE) unless δ>0 strictly improves — never-regress.
        Each candidate is one EASE inversion; gated by catalog size for cost."""
        profile["ease_lambda_heuristic"] = float(heuristic)
        auto_delta = self.ease_denoise == "auto"
        search_lambda = self.ease_lambda_search is True or (
            self.ease_lambda_search == "auto" and n_items <= 8000
        )
        # δ search runs on the EASE-feasible range (held-out builds are the same
        # size as the real fit); λ search keeps its tighter, cheaper gate.
        want_search = (search_lambda or auto_delta) and n_items <= self.ease_max_items
        if not want_search or timestamps_col is None or len(timestamps_col) == 0:
            profile["ease_searched"] = False
            return heuristic, fixed_delta

        # Leave-last-out: hold out each multi-interaction user's latest item.
        order = np.lexsort((timestamps_col, user_idx))  # by user, then time asc
        su, si = user_idx[order], item_idx[order]
        sw = ease_weights[order] if ease_weights is not None else None
        is_last = np.empty(su.shape, dtype=bool)
        is_last[-1] = True
        is_last[:-1] = su[:-1] != su[1:]
        counts = np.bincount(su, minlength=n_users)
        last_pos = np.flatnonzero(is_last)
        multi = counts[su[last_pos]] >= 2
        drop_pos = last_pos[multi]
        if drop_pos.size < 50:
            profile["ease_searched"] = False
            return heuristic, fixed_delta
        held_user = su[drop_pos]
        held_item = si[drop_pos]
        keep = np.ones(su.shape, dtype=bool)
        keep[drop_pos] = False
        tu, ti = su[keep], si[keep]
        tw = sw[keep] if sw is not None else None

        # Sample held users for cheap scoring; build their train history.
        rng = np.random.default_rng(self.random_state)
        if held_user.size > 2000:
            sel = rng.choice(held_user.size, 2000, replace=False)
            held_user, held_item = held_user[sel], held_item[sel]
        held_set = set(held_user.tolist())
        hist: dict[int, list[int]] = {u: [] for u in held_set}
        for u, i in zip(tu.tolist(), ti.tolist()):
            h = hist.get(u)
            if h is not None:
                h.append(i)
        hu, hi = held_user.tolist(), held_item.tolist()

        def recall(lam: float, delta: float) -> float:
            b = np.asarray(
                kindling_core.fit_ease_py(
                    tu, ti, n_users=n_users, n_items=n_items,
                    lambda_=lam, weights=tw, delta=delta,
                ),
                dtype=np.float32,
            )
            hits = 0
            for u, t in zip(hu, hi):
                h = hist.get(u)
                if not h:
                    continue
                scores = b[h].sum(axis=0)
                scores[h] = -np.inf
                if t in np.argpartition(-scores, 10)[:10]:
                    hits += 1
            return hits / max(int(held_user.size), 1)

        base_delta = 0.0 if auto_delta else fixed_delta
        # λ: heuristic, or the opt-in multiplicative grid.
        lam_grid = [heuristic * m for m in (1.0, 2.0, 4.0)] if search_lambda else [heuristic]
        best_lam, best_sc = heuristic, -1.0
        for lam in lam_grid:
            sc = recall(lam, base_delta)
            if sc > best_sc:
                best_sc, best_lam = sc, lam
        # δ: keep base_delta unless a positive δ strictly beats it (never-regress).
        best_delta = base_delta
        if auto_delta:
            for delta in (0.5, 1.0):
                sc = recall(best_lam, delta)
                if sc > best_sc:
                    best_sc, best_delta = sc, delta
        profile["ease_searched"] = True
        profile["ease_lambda_search_mult"] = round(best_lam / max(heuristic, 1e-9), 3)
        return best_lam, best_delta

    def _apply_channel_gate(self) -> None:
        """Backward-eliminate net-negative non-recency channels (user_cf,
        last_item) on a leave-last-fraction held-out, scored by NDCG@10. Drops a
        channel's α only when removing it *strictly* improves held-out NDCG, so
        it disables channels mis-firing on non-temporal data without regressing
        data where they help."""
        st = self._state
        assert st is not None
        if st.base_scorer_used != "ease" or st.ease_b is None:
            return
        do_gate = self.channel_gate is True or (
            self.channel_gate == "auto" and st.n_items <= 20_000
        )
        if not do_gate:
            return
        names = ("trend", "user_cf", "last_item", "transition")
        alphas0 = [
            float(st.trend_alpha),
            float(st.user_cf_alpha),
            float(st.last_item_alpha),
            float(st.transition_alpha),
        ]
        # Gate only the non-recency channels (user_cf, last_item). A held-out
        # carved from a full-train fit leaks the recency window, so trend /
        # transition can't be judged this way — and they already have their own
        # temporal gates. user_cf / last_item are exactly the channels the
        # random-split ablation found net-negative on non-temporal data.
        active = [i for i in (1, 2) if alphas0[i] > 0.0]
        if len(active) < 1:
            return
        # Held-out = the last ~20% of each (time-ordered) history, matching the
        # leave-last-fraction deployment objective. Leave-last-*out* (one item)
        # would mis-rank recency channels: the single last item is already
        # recoverable from the base's co-occurrence, so trend/transition look
        # useless against it even though they help predict the recent tail.
        ents = [(e, h) for e, h in st.owned_by_entity.items() if h.size >= 5]
        if len(ents) < 50:
            return
        rng = np.random.default_rng(self.random_state)
        if len(ents) > 2000:
            ents = [ents[i] for i in rng.choice(len(ents), 2000, replace=False)]
        held = []
        for e, h in ents:
            k = max(1, h.size // 5)
            hist = h[:-k].tolist()
            if hist:
                held.append((int(st.entity_to_user_idx.get(e, -1)), hist, set(h[-k:].tolist())))
        if len(held) < 50:
            return

        from kindling._native_engine import build_native_engine

        native = build_native_engine(self)
        if native is None:
            return

        idcg = [0.0] + [
            sum(1.0 / math.log2(r + 2) for r in range(min(k, 10))) for k in range(1, 11)
        ]

        def ndcg(alphas: list[float]) -> float:
            # NDCG@10, not recall@10: recency channels (trend/transition) mostly
            # re-order items already in the pool, which lifts NDCG without
            # changing top-10 membership — recall would be blind to their value.
            native.set_channel_alphas(*alphas)
            total = 0.0
            for ur, hist, targets in held:
                items, _, _ = native.recommend(hist, ur, 10, 0.0)
                dcg = sum(1.0 / math.log2(r + 2) for r, it in enumerate(items) if it in targets)
                total += dcg / idcg[min(len(targets), 10)]
            return total / len(held)

        # Backward elimination: drop the channel whose removal most improves
        # held-out NDCG, repeat until no removal strictly helps.
        current = list(alphas0)
        cur_r = ndcg(current)
        dropped: list[str] = []
        while True:
            best_i, best_r = None, cur_r
            for i in active:
                if current[i] == 0.0:
                    continue
                trial = list(current)
                trial[i] = 0.0
                r = ndcg(trial)
                if r > best_r:
                    best_r, best_i = r, i
            if best_i is None:
                break
            current[best_i] = 0.0
            cur_r = best_r
            dropped.append(names[best_i])

        (
            st.trend_alpha,
            st.user_cf_alpha,
            st.last_item_alpha,
            st.transition_alpha,
        ) = current
        native.set_channel_alphas(*current)
        self._native = native
        self._native_built = True
        st.profile["channels_gated"] = dropped

    def _apply_repeat_gate(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        timestamps_col: np.ndarray | None,
    ) -> None:
        """Held-out repeat gate. Keep the reorder module only when recommending
        repeats *strictly* improves a leave-last-fraction held-out (NDCG@10, seen
        items eligible). The held-out is leak-free and faithful: the reorder
        profile is rebuilt on the held-out *history* with its real timestamps, so
        the timing multiplier behaves exactly as at serve time. That auto-declines
        fake-repeat data — e.g. steam, where REPLENISH suppresses the just-played
        games the user re-logs — which a repeat-rate threshold cannot tell from
        true repurchase. Only runs for repeat_recommend == 'auto'."""
        st = self._state
        assert st is not None
        if not st.repeat_active or self.repeat_recommend != "auto" or timestamps_col is None:
            return
        if timestamps_col.size == 0 or timestamps_col.max() <= timestamps_col.min():
            return

        # Chronological GLOBAL split, matching the benchmark protocol: hold out the
        # most-recent 15% of interactions by global time (not per-user-recent,
        # which over-represents re-logged repeats). hist = pre-cut, targets =
        # post-cut, per user.
        cut = float(np.quantile(timestamps_col, 0.85))
        is_held = timestamps_col >= cut
        order = np.argsort(user_idx, kind="stable")
        su, si, sts, sh = (
            user_idx[order], item_idx[order], timestamps_col[order], is_held[order]
        )
        bounds = np.flatnonzero(np.r_[True, su[1:] != su[:-1], True])
        rng = np.random.default_rng(self.random_state)
        held: list[tuple[int, set[int]]] = []
        hist_by: dict[int, list[int]] = {}
        hts_by: dict[int, list[float]] = {}
        for a, b in zip(bounds[:-1], bounds[1:]):
            hmask = sh[a:b]
            if hmask.all() or not hmask.any():
                continue  # need both pre-cut history and post-cut targets
            u = int(su[a])
            held.append((u, set(si[a:b][hmask].tolist())))
            hist_by[u] = si[a:b][~hmask].tolist()
            hts_by[u] = sts[a:b][~hmask].tolist()
        if len(held) < 50:
            return
        if len(held) > 2000:
            sel = set(rng.choice(len(held), 2000, replace=False).tolist())
            held = [h for i, h in enumerate(held) if i in sel]
        keep_users = {ur for ur, _ in held}
        hist_by = {u: v for u, v in hist_by.items() if u in keep_users}
        hu_list, hi_list, hts_list = [], [], []
        for u in keep_users:
            items_u = hist_by[u]
            hu_list.extend([u] * len(items_u))
            hi_list.extend(items_u)
            hts_list.extend(hts_by[u])

        from kindling._native_engine import build_native_engine

        native = self._native if self._native_built else build_native_engine(self)
        if native is None:
            return
        # Leak-free, faithful profile: rebuilt on the held-out history *with*
        # timestamps, so the timing multiplier matches serve time. now_ts = the
        # latest kept interaction (the moment we predict the held-out tail).
        hts = np.asarray(hts_list, np.float64)
        rp = kindling_core.fit_repeat_profile(
            np.asarray(hu_list, np.int64), np.asarray(hi_list, np.int64), hts, st.n_users, 2
        )
        full = (st.repeat_indptr, st.repeat_items, st.repeat_counts,
                st.repeat_last_ts, st.repeat_periods, st.repeat_quality)
        native.set_repeat_profile(
            np.asarray(rp[0], np.int64), np.asarray(rp[1], np.int64),
            np.asarray(rp[2], np.float64), np.asarray(rp[3], np.float64),
            np.asarray(rp[4], np.float64), np.asarray(rp[5], np.float64),
            cut,
        )
        idcg = [0.0] + [
            sum(1.0 / math.log2(r + 2) for r in range(min(k, 10))) for k in range(1, 11)
        ]

        def ndcg(active: bool) -> float:
            native.set_repeat_active(active)
            total = 0.0
            for ur, targets in held:
                hist = hist_by.get(ur)
                if not hist:
                    continue
                items, _, _ = native.recommend(hist, ur, 10, 0.0)
                dcg = sum(1.0 / math.log2(r + 2) for r, it in enumerate(items) if it in targets)
                total += dcg / idcg[min(len(targets), 10)]
            return total / len(held)

        on, off = ndcg(True), ndcg(False)
        keep = bool(on > off)
        native.set_repeat_profile(*full, float(st.repeat_now_ts))  # restore full profile
        native.set_repeat_active(keep)
        st.repeat_active = keep
        self._native = native
        self._native_built = True
        st.profile["repeat_gated"] = {
            "kept": keep, "ndcg_on": round(on, 4), "ndcg_off": round(off, 4),
        }

    def _calibrate_boost(
        self,
        ease_b: np.ndarray | None,
        cooc: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        boost_adj: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        timestamps_col: np.ndarray | None,
        profile: dict[str, Any],
    ) -> tuple[float, float]:
        """Held-out (z_threshold, boost_multiplier) grid sweep for the boost
        layers. Builds leave-last-out candidate pools (base + per-layer cooc
        scores + held-out mask) for a sample of users and asks
        ``calibrate_layered_py`` for the (z, boost) cell that maximizes held-out
        NDCG@10 — falling back to (2.5, 3.0) if none beats them. No-op without
        boost layers or timestamps."""
        default = (2.5, 3.0)
        do = self.calibrate_boost is True or self.calibrate_boost == "auto"
        if not do or not boost_adj or timestamps_col is None or timestamps_col.size == 0:
            return default
        layer_csrs = list(boost_adj.values())
        n_core = int(ease_b.shape[0]) if ease_b is not None else (cooc[2].size - 1 if cooc else 0)
        if n_core == 0:
            return default
        # Leave-last-out: hold out each user's latest interaction (count ≥ 2).
        order = np.lexsort((timestamps_col, user_idx))
        su, si = user_idx[order], item_idx[order]
        is_last = np.empty(su.shape, dtype=bool)
        is_last[-1] = True
        is_last[:-1] = su[:-1] != su[1:]
        counts = np.bincount(su, minlength=int(su.max()) + 1 if su.size else 1)
        last_pos = np.flatnonzero(is_last)
        multi = counts[su[last_pos]] >= 2
        drop_pos = last_pos[multi]
        if drop_pos.size < 100:
            return default
        held_user, held_item = su[drop_pos], si[drop_pos]
        keep = np.ones(su.shape, dtype=bool)
        keep[drop_pos] = False
        tu, ti = su[keep], si[keep]
        rng = np.random.default_rng(self.random_state)
        if held_user.size > 1500:
            sel = rng.choice(held_user.size, 1500, replace=False)
            held_user, held_item = held_user[sel], held_item[sel]
        held_set = set(held_user.tolist())
        hist: dict[int, list[int]] = {u: [] for u in held_set}
        for u, i in zip(tu.tolist(), ti.tolist()):
            h = hist.get(u)
            if h is not None and i < n_core:
                h.append(i)
        budget = min(self.retrieval_budget, n_core)
        users: list[tuple[np.ndarray, list[tuple[np.ndarray, str]], np.ndarray]] = []
        for u, t in zip(held_user.tolist(), held_item.tolist()):
            h = hist.get(u)
            if not h or t >= n_core:
                continue
            ho = np.asarray(h, dtype=np.int64)
            if ease_b is not None:
                base_full = ease_b[ho].sum(axis=0).astype(np.float64)
            else:
                d, ix, ip = cooc  # type: ignore[misc]
                base_full = np.asarray(
                    kindling_core.cooccurrence_signal(
                        d, ix, ip, owned_indices=ho.tolist(),
                        candidate_indices=list(range(n_core)),
                    ),
                    dtype=np.float64,
                )
            base_full[ho] = -np.inf
            cand = np.argpartition(-base_full, budget - 1)[:budget]
            cand = cand[np.isfinite(base_full[cand])]
            if t not in set(cand.tolist()):
                continue  # target not retrieved — no (z, boost) can credit it
            cl = cand.tolist()
            layers = [
                (
                    np.asarray(
                        kindling_core.cooccurrence_signal(
                            d, ix, ip, owned_indices=ho.tolist(), candidate_indices=cl
                        ),
                        dtype=np.float64,
                    ),
                    "nonzero",
                )
                for (d, ix, ip) in layer_csrs
            ]
            users.append((base_full[cand].astype(np.float64), layers, (cand == t).astype(np.float64)))
        if len(users) < 50:
            return default
        z_grid = [1.5, 2.0, 2.5, 3.0, 3.5]
        boost_grid = [1.0, 2.0, 3.0, 4.0, 5.0]
        _cells, (bz, bb, bndcg, fell_back) = kindling_core.calibrate_layered_py(
            users, z_grid, boost_grid, 10, 20, 3, 2.5, 3.0, 0.003, 0.0
        )
        profile["boost_calibrated"] = not fell_back
        profile["boost_calibrated_z"] = float(bz)
        profile["boost_calibrated_mult"] = float(bb)
        profile["boost_calibrated_users"] = len(users)
        return (float(bz), float(bb))

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
