"""v2 Engine path — base + z-gated boost layers via ``kindling_core``.

This is the parallel-build implementation of the PRD architecture
(``read-this-prd-ponder-fluffy-turing.md``). It runs alongside the
v1 ``Engine`` rather than replacing it. The v1 engine forwards to
this module when constructed with ``use_v2_core=True``.

Pipeline (per PRD §"Pipeline (the contract)"):

    fit(interactions):
        1. ingest + preprocess
        2. profile → LayerPlan
        3. build cooc base (Rust kernel + decay knob)
        4. if personas enabled:
              ALS factors → HDBSCAN → persona_index → persona_cooc
        5. build enabled boost layers
        6. calibrate (z, boost) via held-out NDCG sweep

    recommend(entity_id, n):
        1. retrieve candidate pool (cooc retriever)
        2. two-gate base routing:
              cluster == -1 → cooc base
              cluster >= 0  → fit ≥ 70% → persona_cooc; else cooc
        3. apply z-gated boost layers
        4. apply repeat multiplier (still Python until Phase 1g)
        5. return top-N

Subsystems still pending Rust port (Phase 1f/g):
- ALS / cosine / LightGCN / interaction_network as boost layers
- repeat module

Until those land, this engine has a smaller boost-layer set than the
final v2 design (path_tail, path_basket, session_cooc, temporal_cooc)
and uses scipy SVD as a stand-in for ALS user factors when ``implicit``
isn't available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from kindling._native import CORE_AVAILABLE, kindling_core
from kindling.explain import Explanation
from kindling.graph.cooc_transform import apply_cooc_transform, resolve_cooc_transform
from kindling.ingest.contract import canonicalize, validate_interactions
from kindling.ingest.sessions import infer_sessions
from kindling.path._sessions import sessions_from_interactions
from kindling.path.basket_index import BasketIndex, build_basket_index
from kindling.path.tail_index import TailIndex, build_tail_index
from kindling.preprocess import preprocess_interactions, weights_of


@dataclass(frozen=True)
class RecommendationV2:
    """Single output row: item + composite score + per-layer contributions."""

    item_id: object
    score: float
    base_kind: str  # "cooc" | "persona_cooc"
    explanation: Explanation | None = None


@dataclass
class V2FitState:
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
    # Personas (optional)
    personas_enabled: bool = False
    n_personas: int = 0
    user_to_persona: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    persona_distinctive: list[list[int]] = field(default_factory=list)
    persona_fit_threshold: float = 0.70
    # Per-persona cooc CSRs.
    persona_cooc_data: list[np.ndarray] = field(default_factory=list)
    persona_cooc_indices: list[np.ndarray] = field(default_factory=list)
    persona_cooc_indptr: list[np.ndarray] = field(default_factory=list)
    # Boost layers (per-layer adjacency / scoring state)
    # layer_name → CSR triple for the layer's cooc-shaped adjacency
    boost_layer_adjacencies: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    # Path-family signals (not cooc-shaped — separate Python objects).
    tail_index: TailIndex | None = None
    basket_index: BasketIndex | None = None
    history_by_entity: dict[object, tuple[object, ...]] = field(default_factory=dict)
    # ALS-as-boost (PRD dense layer): per-entity user factors + per-item
    # item factors. None when als_as_boost is disabled.
    als_user_factors: np.ndarray | None = None
    als_item_factors: np.ndarray | None = None
    # Graph-regularized MF state: per-entity user factors + per-item
    # item factors. None when use_graph_mf is False.
    gmf_user_factors: np.ndarray | None = None
    gmf_item_factors: np.ndarray | None = None
    gmf_role: str = "boost"  # "base" or "boost" — recorded so recommend knows what to do
    gmf_data_graph_kind: str = "none"  # "directional" | "co-ownership" | "none"
    # EASE base scorer: dense item-item weight matrix B (n_items × n_items,
    # f32, zero diagonal). None when the cooc base is active.
    ease_b: np.ndarray | None = None
    base_scorer_used: str = "cooc"      # "cooc" | "ease"
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
    uu_users_data: np.ndarray | None = None     # concatenated user ids
    uu_users_indptr: np.ndarray | None = None   # per-item offsets
    uu_user_deg: np.ndarray | None = None
    user_row_items: dict[int, np.ndarray] = field(default_factory=dict)
    user_cf_alpha: float = 0.0
    user_cf_k: int = 100
    # Rating-signal classification + resolved use_als decision.
    signal_kind: str = "unknown"        # "binary" | "counts" | "ratings" | "forced_*"
    als_ran: bool = False               # whether ALS actually ran in this fit
    persona_method_used: str = "none"   # "hdbscan_factors" | "louvain_graph" | "none"
    # Calibrated scoring config
    z_threshold: float = 2.5
    boost_multiplier: float = 3.0
    # Diagnostics
    fit_seconds: float = 0.0
    profile: dict[str, Any] = field(default_factory=dict)


class EngineV2:
    """v2 Engine. ``Engine(use_v2_core=True)`` constructs and forwards to this."""

    def __init__(
        self,
        n_personas: int = 30,
        persona_min_users: int = 1000,
        persona_fit_threshold: float = 0.70,
        retrieval_budget: int = 500,
        random_state: int = 0,
        # Ablation knobs for the ALS-vs-SVD experiment.
        # `hdbscan_factor_method`: which low-dim embedding feeds HDBSCAN.
        #   "als" — full implicit-ALS (Hu/Koren/Volinsky); accurate but slow.
        #   "svd" — randomized truncated SVD (Halko et al.); cheap, no
        #          implicit-feedback alignment but usually good enough
        #          for clustering.
        # `als_as_boost`: when True, fits item factors via ALS and adds
        # `user_u · item_c` as a dense-z boost layer (PRD spec).
        # Always fits ALS when the ablation requests SVD-for-HDBSCAN +
        # ALS-as-boost; otherwise we skip ALS entirely if HDBSCAN takes
        # SVD and boost is off.
        hdbscan_factor_method: str = "als",
        als_as_boost: bool = False,
        # `use_als`: governs whether ALS runs at all on this dataset.
        #   "auto"      — detect signal kind from the rating/weight column;
        #                 enable ALS only when the signal is "ratings" (a
        #                 bounded narrow-range distribution with multiple
        #                 distinct values). Skip on "binary" or "counts".
        #   "force_on"  — run ALS regardless of detected signal.
        #   "force_off" — never run ALS; HDBSCAN inputs fall back to SVD,
        #                 als_as_boost is silently disabled.
        # On detected "binary" / "counts" data the implicit-feedback trick
        # collapses to weighted SVD on the cooc structure — same signal
        # cooc already captures (see graph_mf ablation discussion).
        use_als: str = "auto",
        # `persona_method`: how to identify personas (i.e., user clusters).
        #   "auto"             — pick based on signal_kind:
        #                          ratings → hdbscan_factors
        #                          binary / counts → louvain_graph
        #   "hdbscan_factors"  — HDBSCAN over (ALS or SVD) user factors.
        #                        Density-based, k-dim continuous embedding.
        #   "louvain_graph"    — Louvain community detection on the
        #                        user-user projected graph. Modularity-
        #                        based, operates on raw edge structure.
        # The output of either path feeds the same downstream persona-
        # index pipeline (rates → z-filter → distinctive_items → fit_gate).
        persona_method: str = "auto",
        # Per-item user cap when projecting bipartite to user-user graph
        # for Louvain. Bounds memory on popular items.
        louvain_max_users_per_item: int = 100,
        # Communities below this size become noise (-1) — analogous to
        # HDBSCAN's noise label. Default mirrors HDBSCAN's min_cluster_size.
        louvain_min_community_size: int = 30,
        # Pre-process the user-user graph weights before Louvain.
        #   "raw" — accumulated Σ w_u·w_v counts (heavy-tailed);
        #            popular-item-sharing pairs dominate modularity
        #   "log" — ln(1 + w); compresses dynamic range so big-share
        #            pairs don't drown out medium-share ones
        louvain_weight_transform: str = "log",
        # Drop the bottom percentile of edges by weight (after transform).
        # 0.05 removes the long tail of single-shared-item pairs that
        # are mostly noise. 0.0 disables pruning.
        louvain_min_edge_percentile: float = 0.05,
        # Trim users from the top/bottom percentiles of activity (interaction
        # count) BEFORE building the user-user graph. Trimmed users have
        # no edges in the projected graph, so they end up as cluster=-1
        # (noise) and the persona-fit gate routes them to cooc base.
        # Top trim removes "everyone-connects-to-them" hubs; bottom trim
        # removes degenerate single-event users that only add noise.
        # 0.0 / 0.0 disables (default).
        louvain_user_trim_top: float = 0.0,
        louvain_user_trim_bottom: float = 0.0,
        # Resolution γ (Reichardt-Bornholdt 2006) for Louvain modularity:
        #   Q_γ = (1/2m) Σ (A_ij − γ·k_i·k_j / 2m) δ(c_i, c_j)
        # γ = 1.0 = standard modularity. γ > 1 produces more, smaller
        # communities (helps when standard Louvain merges too aggressively
        # on dense user-user graphs); γ < 1 produces fewer, larger.
        # Practical range 0.5–3.0.
        louvain_resolution: float = 1.0,
        # Degree-corrected SBM knobs (only used when persona_method='dc_sbm').
        # Warm-starts from Louvain on the same user-user graph and runs MAP
        # iterations until <1% of nodes move per pass or `max_passes` is hit.
        # `min_internal_fraction` provides per-node noise routing analogous
        # to HDBSCAN's -1 label: a node whose within-block edge weight is
        # below this fraction of its total degree gets reassigned to -1.
        dc_sbm_max_passes: int = 15,
        dc_sbm_min_internal_fraction: float = 0.0,
        # Resolution γ used for the Louvain warm-start that seeds DC-SBM.
        # Defaults higher than the standalone Louvain default (1.0) because
        # SBM needs multiple starting blocks to refine — with γ=1.0,
        # dense graphs can collapse to one block and SBM has nothing to
        # split. γ=1.5 gives a richer init.
        dc_sbm_warmstart_resolution: float = 1.5,
        # Init mode for SBM blocks:
        #   "louvain"  — warm-start from Louvain at warmstart_resolution
        #   "random_k" — random K-block init (use when Louvain under-
        #                clusters; SBM can't grow past warm-start block count)
        #   "auto"     — Louvain first; if it produces < random_k_floor
        #                blocks, fall back to random_k init
        dc_sbm_init_mode: str = "louvain",
        # Target block count for random init. Used directly in "random_k"
        # mode and as the threshold in "auto" mode.
        dc_sbm_random_k: int = 20,
        # Coherence filter — algorithm-agnostic post-hoc persona quality
        # gate. After clustering, compute per-persona coherence as the
        # mean cooc[i,j] over (i,j) pairs in distinctive_items[p]; drop
        # personas below the given percentile (their members → cluster=-1
        # → fit-gate routes them to cooc base). Replaces algorithm-
        # specific noise labels (HDBSCAN's -1, Louvain's min_community_size)
        # with a uniform measure of "is this persona's item set actually
        # clustered together in user co-occurrence?".
        #
        # 0.0 disables (no filtering). 0.5 keeps top half by coherence.
        # 0.25 keeps top 75%. Practical range 0.0 - 0.7.
        coherence_filter_percentile: float = 0.0,
        # Personas with fewer than this many members are treated as
        # noise *before* coherence ranking — small personas with rare
        # items get artificially high pairwise cooc and would otherwise
        # dominate the keep-list. Default: persona_min_users / 30
        # (so for default 1000-user threshold, requires ≥33 members).
        coherence_min_persona_users: int = 30,
        # Personas with more than this fraction of total users are also
        # treated as noise — a "persona" containing 90% of the user base
        # is just a relabel of global cooc and adds no differentiation.
        # 1.0 disables; 0.7 = drop personas spanning >70% of users.
        coherence_max_persona_fraction: float = 0.7,
        # Graph-regularized matrix factorization (GR-MF / graph_mf).
        # Off by default. When enabled, GR-MF runs alongside the existing
        # base path:
        #   role="base"  — replace cooc base for retrieval+scoring
        #   role="boost" — add as a dense z-mode boost layer
        # Two graphs feed it:
        #   - data graph: A3 profile-gated. When session structure is
        #     inferable, builds directional cooc + symmetrizes via
        #     D + Dᵀ. Else falls back to existing co-ownership cooc.
        #   - hierarchy graph: optional, from item_metadata when the
        #     loader provides it (B2). amazon-beauty supported first.
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
        # Per-fit base calibration (EXPERIMENTAL — default off). Holds
        # out the final 10% of train events and grid-searches
        # (ease_lambda, trend_alpha, transition_alpha) on internal
        # NDCG@10, then refits on full train with the winners.
        #
        # Measured 2026-06: internal choices do NOT transfer — on
        # amazon-beauty the internal ranking inverts the test ranking
        # for trend_alpha (internal prefers 0.0; test strongly prefers
        # 0.5) and degrades test NDCG 0.0310 → 0.0203; ml1m degrades
        # 0.2859 → 0.2741. Shifting every window back one slice changes
        # the popularity-drift structure that the trend channel
        # exploits, so the holdout systematically undervalues it. The
        # cross-dataset-validated fixed defaults are more robust. Keep
        # for diagnostics (profile["base_calibration"]["grid"]).
        calibrate_base: bool = False,
        use_graph_mf: bool = False,
        graph_mf_role: str = "boost",  # "base" or "boost"
        graph_mf_alpha_data: float = 0.1,
        graph_mf_alpha_hierarchy: float = 0.5,
        graph_mf_n_iters: int = 15,
        graph_mf_dim: int = 32,
    ):
        if not CORE_AVAILABLE:
            raise ImportError(
                "kindling_core extension not available; build with "
                "`maturin build` in native/kindling_core/"
            )
        self.n_personas = n_personas
        self.persona_min_users = persona_min_users
        self.persona_fit_threshold = persona_fit_threshold
        self.retrieval_budget = retrieval_budget
        self.random_state = random_state
        if hdbscan_factor_method not in ("als", "svd"):
            raise ValueError(
                f"hdbscan_factor_method must be 'als' or 'svd'; got "
                f"{hdbscan_factor_method!r}"
            )
        self.hdbscan_factor_method = hdbscan_factor_method
        self.als_as_boost = als_as_boost
        if use_als not in ("auto", "force_on", "force_off"):
            raise ValueError(
                f"use_als must be 'auto', 'force_on', or 'force_off'; got {use_als!r}"
            )
        self.use_als = use_als
        if persona_method not in (
            "auto", "hdbscan_factors", "louvain_graph", "dc_sbm"
        ):
            raise ValueError(
                f"persona_method must be 'auto', 'hdbscan_factors', "
                f"'louvain_graph', or 'dc_sbm'; got {persona_method!r}"
            )
        self.persona_method = persona_method
        self.louvain_max_users_per_item = louvain_max_users_per_item
        self.louvain_min_community_size = louvain_min_community_size
        if louvain_weight_transform not in ("raw", "log", "cosine"):
            raise ValueError(
                f"louvain_weight_transform must be 'raw' | 'log' | 'cosine'; "
                f"got {louvain_weight_transform!r}"
            )
        self.louvain_weight_transform = louvain_weight_transform
        if not (0.0 <= louvain_min_edge_percentile < 1.0):
            raise ValueError(
                f"louvain_min_edge_percentile must be in [0, 1); "
                f"got {louvain_min_edge_percentile!r}"
            )
        self.louvain_min_edge_percentile = louvain_min_edge_percentile
        if not (0.0 <= louvain_user_trim_top < 0.5):
            raise ValueError(
                f"louvain_user_trim_top must be in [0, 0.5); "
                f"got {louvain_user_trim_top!r}"
            )
        if not (0.0 <= louvain_user_trim_bottom < 0.5):
            raise ValueError(
                f"louvain_user_trim_bottom must be in [0, 0.5); "
                f"got {louvain_user_trim_bottom!r}"
            )
        self.louvain_user_trim_top = louvain_user_trim_top
        self.louvain_user_trim_bottom = louvain_user_trim_bottom
        if louvain_resolution <= 0.0:
            raise ValueError(
                f"louvain_resolution must be > 0; got {louvain_resolution!r}"
            )
        self.louvain_resolution = louvain_resolution
        if not (0.0 <= coherence_filter_percentile < 1.0):
            raise ValueError(
                f"coherence_filter_percentile must be in [0, 1); "
                f"got {coherence_filter_percentile!r}"
            )
        self.coherence_filter_percentile = coherence_filter_percentile
        if coherence_min_persona_users < 0:
            raise ValueError(
                f"coherence_min_persona_users must be >= 0; "
                f"got {coherence_min_persona_users!r}"
            )
        self.coherence_min_persona_users = coherence_min_persona_users
        if not (0.0 < coherence_max_persona_fraction <= 1.0):
            raise ValueError(
                f"coherence_max_persona_fraction must be in (0, 1]; "
                f"got {coherence_max_persona_fraction!r}"
            )
        self.coherence_max_persona_fraction = coherence_max_persona_fraction
        if base_scorer not in ("auto", "cooc", "ease"):
            raise ValueError(
                f"base_scorer must be 'auto', 'cooc', or 'ease'; got {base_scorer!r}"
            )
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
                f"trend_window_fraction must be in (0, 1]; "
                f"got {trend_window_fraction!r}"
            )
        self.trend_window_fraction = trend_window_fraction
        if transition_alpha < 0.0:
            raise ValueError(f"transition_alpha must be >= 0; got {transition_alpha!r}")
        self.transition_alpha = transition_alpha
        if transition_last_k < 1:
            raise ValueError(
                f"transition_last_k must be >= 1; got {transition_last_k!r}"
            )
        self.transition_last_k = transition_last_k
        if not (0.0 < transition_decay <= 1.0):
            raise ValueError(
                f"transition_decay must be in (0, 1]; got {transition_decay!r}"
            )
        self.transition_decay = transition_decay
        if content_alpha < 0.0:
            raise ValueError(f"content_alpha must be >= 0; got {content_alpha!r}")
        self.content_alpha = content_alpha
        if content_warmth_threshold < 1:
            raise ValueError(
                f"content_warmth_threshold must be >= 1; "
                f"got {content_warmth_threshold!r}"
            )
        self.content_warmth_threshold = content_warmth_threshold
        self.open_catalog = bool(open_catalog)
        self.open_catalog_max_extension = open_catalog_max_extension
        if cold_slots < 0:
            raise ValueError(f"cold_slots must be >= 0; got {cold_slots!r}")
        self.cold_slots = int(cold_slots)
        if cold_recency_beta < 0:
            raise ValueError(
                f"cold_recency_beta must be >= 0; got {cold_recency_beta!r}"
            )
        self.cold_recency_beta = float(cold_recency_beta)
        if cold_user_pop_prior < 0.0:
            raise ValueError(
                f"cold_user_pop_prior must be >= 0; got {cold_user_pop_prior!r}"
            )
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
            raise ValueError(
                f"user_cf_history_gate must be >= 0; got {user_cf_history_gate!r}"
            )
        self.user_cf_history_gate = user_cf_history_gate
        if ease_use_weights not in ("auto", "on", "off"):
            raise ValueError(
                f"ease_use_weights must be 'auto' | 'on' | 'off'; "
                f"got {ease_use_weights!r}"
            )
        self.ease_use_weights = ease_use_weights
        self.calibrate_base = bool(calibrate_base)
        if dc_sbm_max_passes < 1:
            raise ValueError(
                f"dc_sbm_max_passes must be >= 1; got {dc_sbm_max_passes!r}"
            )
        self.dc_sbm_max_passes = dc_sbm_max_passes
        if not (0.0 <= dc_sbm_min_internal_fraction < 1.0):
            raise ValueError(
                f"dc_sbm_min_internal_fraction must be in [0, 1); "
                f"got {dc_sbm_min_internal_fraction!r}"
            )
        self.dc_sbm_min_internal_fraction = dc_sbm_min_internal_fraction
        if dc_sbm_warmstart_resolution <= 0.0:
            raise ValueError(
                f"dc_sbm_warmstart_resolution must be > 0; "
                f"got {dc_sbm_warmstart_resolution!r}"
            )
        self.dc_sbm_warmstart_resolution = dc_sbm_warmstart_resolution
        if dc_sbm_init_mode not in ("louvain", "random_k", "auto"):
            raise ValueError(
                f"dc_sbm_init_mode must be 'louvain', 'random_k', or 'auto'; "
                f"got {dc_sbm_init_mode!r}"
            )
        self.dc_sbm_init_mode = dc_sbm_init_mode
        if dc_sbm_random_k < 2:
            raise ValueError(
                f"dc_sbm_random_k must be >= 2; got {dc_sbm_random_k!r}"
            )
        self.dc_sbm_random_k = dc_sbm_random_k
        if graph_mf_role not in ("base", "boost"):
            raise ValueError(
                f"graph_mf_role must be 'base' or 'boost'; got {graph_mf_role!r}"
            )
        self.use_graph_mf = use_graph_mf
        self.graph_mf_role = graph_mf_role
        self.graph_mf_alpha_data = graph_mf_alpha_data
        self.graph_mf_alpha_hierarchy = graph_mf_alpha_hierarchy
        self.graph_mf_n_iters = graph_mf_n_iters
        self.graph_mf_dim = graph_mf_dim
        self._state: V2FitState | None = None

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

        The implicit-feedback ALS confidence trick only earns its name
        when the input is 'ratings' (data-driven c_ui spread). On
        'binary' or 'counts' inputs, ALS collapses to a low-rank
        rewrite of the cooc structure — same signal cooc already has.
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

    def _resolve_use_als(self, weights: np.ndarray) -> tuple[bool, str]:
        """Resolve self.use_als into (effective_bool, signal_kind).

        Returns:
          (run_als, signal_kind) where signal_kind is one of
          'binary' | 'counts' | 'ratings' | 'forced_on' | 'forced_off'.
        """
        if self.use_als == "force_on":
            return True, "forced_on"
        if self.use_als == "force_off":
            return False, "forced_off"
        kind = self.detect_rating_signal(weights)
        return (kind == "ratings"), kind

    def _calibrate_base_params(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        timestamps_col: np.ndarray | None,
        n_users: int,
        n_items: int,
        transition_gate_open: bool,
        profile: dict[str, Any],
    ) -> tuple[float, float, float]:
        """Pick (ease_lambda, trend_alpha, transition_alpha) on an
        internal chronological holdout.

        Splits the train rows by time (last 10% = calibration slice),
        fits EASE per λ candidate on the remaining 90%, and evaluates
        every (λ, trend_α, trans_α) triple by NDCG@10 against each
        calibration user's held-out items. The α grids cost no refits —
        the channels are independent of B — so the search is |λ grid|
        inversions plus cheap vector math.
        """
        n_obs = len(user_idx)
        if timestamps_col is not None and len(timestamps_col) == n_obs:
            order = np.argsort(timestamps_col, kind="mergesort")
        else:
            order = np.random.RandomState(self.random_state).permutation(n_obs)
        cut = int(n_obs * 0.9)
        fit_rows, cal_rows = order[:cut], order[cut:]
        fu, fi = user_idx[fit_rows], item_idx[fit_rows]
        ft = timestamps_col[fit_rows] if timestamps_col is not None else None
        cu, ci = user_idx[cal_rows], item_idx[cal_rows]

        # Per-user fit-slice history (time-ordered: fit_rows is sorted)
        # + calibration relevant sets.
        owned_fit: dict[int, list[int]] = {}
        for u, i in zip(fu.tolist(), fi.tolist()):
            owned_fit.setdefault(u, []).append(i)
        cal_rel: dict[int, set[int]] = {}
        for u, i in zip(cu.tolist(), ci.tolist()):
            cal_rel.setdefault(u, set()).add(i)
        eligible = sorted(u for u in cal_rel if u in owned_fit)
        if len(eligible) > 400:
            step = len(eligible) // 400
            eligible = eligible[::step][:400]
        if not eligible:
            auto_lam = 20.0 * n_obs / max(n_items, 1)
            return auto_lam, self.trend_alpha, self.transition_alpha

        # Channels from the fit slice.
        trend_z = None
        if ft is not None and len(ft):
            t_hi, t_lo = float(np.max(ft)), float(np.min(ft))
            if t_hi > t_lo:
                cut_t = t_hi - (t_hi - t_lo) * self.trend_window_fraction
                counts = np.bincount(
                    fi[ft >= cut_t], minlength=n_items
                ).astype(np.float64)
                if counts.std() > 0:
                    trend_z = (counts - counts.mean()) / counts.std()
        trans_csr = None
        if transition_gate_open and ft is not None:
            td, ti_, tp = kindling_core.build_directional_cooc(
                fu, fi, np.ones(len(fu), dtype=np.float32),
                n_sessions=n_users, n_items=n_items, timestamps=ft,
            )
            trans_csr = (
                np.asarray(td, dtype=np.float32),
                np.asarray(ti_, dtype=np.int32),
                np.asarray(tp, dtype=np.int32),
            )

        def _user_trans_z(owned_list: list[int]) -> np.ndarray | None:
            if trans_csr is None:
                return None
            td, ti_, tp = trans_csr
            v = np.zeros(n_items, dtype=np.float64)
            for j, item in enumerate(owned_list[::-1][: self.transition_last_k]):
                s_, e_ = int(tp[item]), int(tp[item + 1])
                if e_ > s_:
                    v[ti_[s_:e_]] += (self.transition_decay ** j) * td[s_:e_]
            std = v.std()
            return (v - v.mean()) / std if std > 0 else None

        def _ndcg10(top: np.ndarray, rel: set[int]) -> float:
            dcg = sum(
                1.0 / np.log2(r + 2) for r, it in enumerate(top.tolist()) if it in rel
            )
            ideal = sum(1.0 / np.log2(r + 2) for r in range(min(10, len(rel))))
            return dcg / ideal if ideal > 0 else 0.0

        auto_lam = 20.0 * len(fu) / max(n_items, 1)
        lam_grid = (
            [self.ease_lambda]
            if self.ease_lambda is not None
            else [auto_lam * m for m in (0.5, 1.0, 2.0, 4.0)]
        )
        trend_grid = [0.0, 0.25, 0.5, 1.0] if trend_z is not None else [0.0]
        trans_grid = [0.0, 0.25, 0.5] if trans_csr is not None else [0.0]

        best = (lam_grid[0], self.trend_alpha if trend_z is not None else 0.0, 0.0)
        best_score = -1.0
        results: list[dict[str, float]] = []
        for lam in lam_grid:
            b = np.asarray(
                kindling_core.fit_ease_py(
                    fu, fi, n_users=n_users, n_items=n_items, lambda_=lam
                ),
                dtype=np.float32,
            )
            # Precompute per-user z(ease) + trans_z once per λ.
            per_user: list[tuple[np.ndarray, np.ndarray | None, np.ndarray, set[int]]] = []
            for u in eligible:
                owned_list = owned_fit[u]
                owned_arr = np.asarray(owned_list, dtype=np.int64)
                ease = b[owned_arr].sum(axis=0, dtype=np.float64)
                std = ease.std()
                if std > 0:
                    ease = (ease - ease.mean()) / std
                per_user.append((ease, _user_trans_z(owned_list), owned_arr, cal_rel[u]))
            del b
            for ta in trend_grid:
                for xa in trans_grid:
                    total = 0.0
                    for ez, tz, owned_arr, rel in per_user:
                        s = ez.copy()
                        if ta > 0 and trend_z is not None:
                            s += ta * trend_z
                        if xa > 0 and tz is not None:
                            s += xa * tz
                        s[owned_arr] = -np.inf
                        top = np.argpartition(-s, 10)[:10]
                        top = top[np.argsort(-s[top], kind="stable")]
                        total += _ndcg10(top, rel)
                    mean_ndcg = total / len(per_user)
                    results.append(
                        {"lambda": lam, "trend_alpha": ta,
                         "transition_alpha": xa, "ndcg10": mean_ndcg}
                    )
                    if mean_ndcg > best_score:
                        best_score = mean_ndcg
                        best = (lam, ta, xa)
        profile["base_calibration"] = {
            "n_cal_users": len(eligible),
            "chosen": {"lambda": best[0], "trend_alpha": best[1],
                       "transition_alpha": best[2]},
            "internal_ndcg10": best_score,
            "grid": results,
        }
        return best

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

    def _open_catalog_extension_cap(self, n_obs: int, n_train_items: int) -> int:
        """Max metadata-only extension items that fit under the RAM budget.

        Reserves the estimated interaction-fit peak, then spends the
        remaining headroom (under 80% of physical RAM) on the extension.
        Returns 0 when the interaction fit alone already exceeds budget —
        the engine then runs catalog-only rather than risking an OOM.
        """
        if self.open_catalog_max_extension is not None:
            return max(0, int(self.open_catalog_max_extension))
        try:
            import psutil
            total = psutil.virtual_memory().total
        except Exception:
            total = 8 * 1024**3  # conservative fallback when psutil absent
        # Budget against 80% of PHYSICAL RAM, not currently-available:
        # macOS swap absorbed the 17.4 GB catalog-only book fit even with
        # ~11 GB free, so the jetsam wall sits near physical, not avail.
        # 0.80×24 GB ≈ 20.6 GB lands between the 17.4 GB that survived and
        # the ~23 GB (357k+200k ext) that died. The interaction fit is
        # still ahead of this call → reserve its estimated peak first.
        ceiling = 0.80 * total
        interaction_peak = (
            self._PEAK_BYTES_PER_OBS * n_obs
            + self._PEAK_BYTES_PER_TRAIN_ITEM * n_train_items
        )
        headroom = ceiling - interaction_peak
        if headroom <= 0:
            return 0
        return int(headroom / self._EXTENSION_BYTES_PER_ITEM)

    def fit(
        self,
        interactions: pd.DataFrame,
        item_metadata: pd.DataFrame | None = None,
    ) -> "EngineV2":
        t0 = time.perf_counter()
        # Same contract as v1: validate → canonicalize → preprocess.
        schema = validate_interactions(interactions)
        canonical = canonicalize(interactions, schema)
        canonical, _ctx = preprocess_interactions(canonical, use_ratings=None)
        interactions = canonical
        weights = weights_of(interactions).astype(np.float32)
        # Resolve use_als BEFORE building the persona pipeline so we
        # know whether to instantiate ALS at all.
        run_als, signal_kind = self._resolve_use_als(weights)
        if not run_als:
            # When ALS won't run, force HDBSCAN to use SVD inputs and
            # silently disable ALS-as-boost. Caller can still see the
            # decision in fit_summary().
            self.hdbscan_factor_method = "svd"
            self.als_as_boost = False
        # Build catalogs.
        item_ids = pd.Index(interactions["item_id"].unique())
        item_to_idx = {item: i for i, item in enumerate(item_ids)}
        n_items = len(item_ids)
        entity_ids = pd.Index(interactions["entity_id"].unique())
        entity_to_user_idx = {e: i for i, e in enumerate(entity_ids)}
        n_users = len(entity_ids)

        item_idx = interactions["item_id"].map(item_to_idx).to_numpy(dtype=np.int64)
        user_idx = (
            interactions["entity_id"].map(entity_to_user_idx).to_numpy(dtype=np.int64)
        )
        # ── Open catalog: metadata-only items get catalog indices ≥
        # n_items. All interaction-derived structures (cooc, EASE,
        # transitions, personas, user-CF) stay in the train subspace
        # [0, n_items); only catalog-length vectors (trend, content,
        # blend output) span the extension.
        n_items_ext = n_items
        extension_capped: dict | None = None
        if (
            self.open_catalog
            and item_metadata is not None
            and "item_id" in item_metadata.columns
        ):
            extra = pd.Index(
                item_metadata["item_id"].dropna().unique()
            ).difference(item_ids)
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
                cap = self._open_catalog_extension_cap(
                    n_obs=len(user_idx), n_train_items=n_items
                )
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
        # Keep a reference to the RAW co-counts: the cooc base transform below
        # rebinds `cooc_data`, but embedding imputation (cold slots) needs the
        # untransformed counts for its PPMI-SVD cooc embedding.
        cooc_raw_data = cooc_data

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
        # Effective hyper-params: calibrated per fit when enabled, else
        # the configured/auto values.
        eff_trend_alpha = self.trend_alpha
        eff_trans_alpha = self.transition_alpha if transition_gate_open else 0.0
        if ease_eligible:
            if self.calibrate_base:
                t_cal = time.perf_counter()
                eff_lambda, eff_trend_alpha, eff_trans_alpha = (
                    self._calibrate_base_params(
                        user_idx, item_idx, timestamps_col,
                        n_users, n_items, transition_gate_open, profile,
                    )
                )
                profile["base_calibration_seconds"] = time.perf_counter() - t_cal
            else:
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
                    user_idx, item_idx,
                    n_users=n_users, n_items=n_items,
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
                counts = np.bincount(
                    item_idx[recent_mask], minlength=n_items_ext
                ).astype(np.float64)
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
                user_idx, item_idx, weights,
                n_sessions=n_users, n_items=n_items,
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
        if (self.content_alpha > 0.0 or self.cold_slots > 0) and item_metadata is not None:
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

        # ── Cold-slot release recency (schema-inferred). Find the first
        # metadata column whose values are majority-parseable datetimes;
        # recency = exp(−days_before_train_end / 180), 0 when unknown.
        cold_recency: np.ndarray | None = None
        if (
            self.cold_slots > 0 or self.cold_recency_beta > 0
        ) and item_metadata is not None:
            # `errors="coerce"` + scalar→Series so an out-of-range epoch
            # (bad/0/huge timestamp) yields NaT rather than raising.
            ref_end = None
            if timestamps_col is not None and len(timestamps_col):
                _re = pd.to_datetime(
                    pd.Series([float(np.max(timestamps_col))]),
                    unit="s", errors="coerce",
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
        median_history = float(
            np.median(user_counts_for_gate[user_counts_for_gate > 0])
        ) if (user_counts_for_gate > 0).any() else 0.0
        profile["median_items_per_user"] = median_history
        user_cf_open = (
            self.user_cf_alpha > 0.0
            and median_history <= self.user_cf_history_gate
        )
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
                u: sorted_i[bounds[u]:bounds[u + 1]]
                for u in range(n_users)
                if bounds[u + 1] > bounds[u]
            }
        else:
            user_row_items = {}
        profile["user_cf_channel_active"] = bool(user_cf_open)

        # ── Personas (if enabled).
        personas_enabled = bool(plan["personas_enabled"]) and n_users >= self.persona_min_users
        n_personas_actual = 0
        user_to_persona = np.array([], dtype=np.int64)
        persona_distinctive: list[list[int]] = []
        persona_cooc_data: list[np.ndarray] = []
        persona_cooc_indices: list[np.ndarray] = []
        persona_cooc_indptr: list[np.ndarray] = []
        # Resolve persona_method (auto picks based on signal_kind).
        persona_method = self.persona_method
        if persona_method == "auto":
            persona_method = (
                "hdbscan_factors" if signal_kind == "ratings" or signal_kind == "forced_on"
                else "louvain_graph"
            )
        # ALS-as-boost requires item factors even if personas aren't on.
        # Factor fitting only needed for hdbscan_factors path or als_as_boost.
        item_factors_for_boost: np.ndarray | None = None
        user_factors: np.ndarray | None = None
        need_factors = (
            (personas_enabled and persona_method == "hdbscan_factors")
            or self.als_as_boost
        )
        if need_factors:
            user_factors, item_factors = self._fit_factors(
                user_idx, item_idx, weights, n_users, n_items
            )
            if self.als_as_boost:
                item_factors_for_boost = item_factors
            # L2-row-normalize the user factors before passing to HDBSCAN.
            # Petal-clustering's eps=0.5 default expects a unit-bounded
            # embedding (UMAP's natural output range). Raw ALS/SVD factors
            # span 1-4 in pairwise distance, so most users look isolated
            # to HDBSCAN unless we normalize. After normalization,
            # Euclidean distance is angular; clustering aligns with taste
            # similarity rather than activity-level magnitude.
            user_norms = np.linalg.norm(user_factors, axis=1, keepdims=True)
            user_factors_normalized = user_factors / np.maximum(user_norms, 1e-9)
        else:
            user_factors_normalized = None
        noise_frac: float = 0.0
        if personas_enabled and persona_method == "hdbscan_factors":
            assert user_factors_normalized is not None
            # On the unit sphere, eps=0.5 is sane and 30 is enough to
            # form meaningful clusters without the 0.5% threshold dominating.
            assignments, _probs, n_personas_actual, noise_frac = kindling_core.fit_hdbscan_py(
                user_factors_normalized,
                min_cluster_size=max(30, int(0.001 * n_users)),
                min_samples=10,
            )
            assignments = np.asarray(assignments, dtype=np.int64)
            user_to_persona = assignments
        elif personas_enabled and persona_method == "louvain_graph":
            # Optional user trim: identify the top/bottom percentile users
            # by raw interaction count and exclude their (user, item) rows
            # from the graph build. Trimmed users will have no edges in
            # the projected graph → cluster=-1 → fit-gate routes them
            # to cooc base.
            lo = self.louvain_user_trim_bottom
            hi = self.louvain_user_trim_top
            if lo > 0.0 or hi > 0.0:
                user_counts = np.bincount(user_idx, minlength=n_users)
                # Threshold by percentile across users that touched ≥1 item.
                active = user_counts[user_counts > 0]
                if active.size > 0:
                    lo_thr = np.percentile(active, lo * 100.0) if lo > 0 else 0.0
                    hi_thr = (
                        np.percentile(active, (1.0 - hi) * 100.0)
                        if hi > 0 else float("inf")
                    )
                    keep_user = (user_counts > lo_thr) & (user_counts <= hi_thr)
                    keep_row = keep_user[user_idx]
                    trim_user_idx = user_idx[keep_row]
                    trim_item_idx = item_idx[keep_row]
                    trim_weights = weights[keep_row]
                else:
                    trim_user_idx, trim_item_idx, trim_weights = (
                        user_idx, item_idx, weights
                    )
            else:
                trim_user_idx, trim_item_idx, trim_weights = (
                    user_idx, item_idx, weights
                )
            # Build user-user projected graph + run Louvain.
            uu_data, uu_indices, uu_indptr = kindling_core.build_user_user_graph(
                trim_user_idx, trim_item_idx, trim_weights,
                n_users=n_users, n_items=n_items,
                max_users_per_item=self.louvain_max_users_per_item,
                seed=self.random_state,
                weight_transform=self.louvain_weight_transform,
                min_edge_percentile=self.louvain_min_edge_percentile,
            )
            uu_data = np.asarray(uu_data, dtype=np.float32)
            uu_indices = np.asarray(uu_indices, dtype=np.int32)
            uu_indptr = np.asarray(uu_indptr, dtype=np.int32)
            assignments, n_personas_actual, modularity, _passes, noise_frac = (
                kindling_core.fit_louvain_py(
                    uu_data, uu_indices, uu_indptr,
                    min_community_size=self.louvain_min_community_size,
                    max_passes=30,
                    modularity_tol=1e-6,
                    resolution=self.louvain_resolution,
                )
            )
            assignments = np.asarray(assignments, dtype=np.int64)
            user_to_persona = assignments
            profile["louvain_modularity"] = float(modularity)
        elif personas_enabled and persona_method == "dc_sbm":
            # Degree-corrected stochastic block model — hand-rolled MAP
            # estimator (Rust, `kindling_core::cluster::dc_sbm`) with
            # Louvain warm-start on the same user-user graph. The
            # Louvain weight-transform / edge-prune / user-trim knobs
            # all apply to the underlying graph build.
            lo = self.louvain_user_trim_bottom
            hi = self.louvain_user_trim_top
            if lo > 0.0 or hi > 0.0:
                user_counts = np.bincount(user_idx, minlength=n_users)
                active = user_counts[user_counts > 0]
                if active.size > 0:
                    lo_thr = np.percentile(active, lo * 100.0) if lo > 0 else 0.0
                    hi_thr = (
                        np.percentile(active, (1.0 - hi) * 100.0)
                        if hi > 0 else float("inf")
                    )
                    keep_user = (user_counts > lo_thr) & (user_counts <= hi_thr)
                    keep_row = keep_user[user_idx]
                    trim_user_idx = user_idx[keep_row]
                    trim_item_idx = item_idx[keep_row]
                    trim_weights = weights[keep_row]
                else:
                    trim_user_idx, trim_item_idx, trim_weights = (
                        user_idx, item_idx, weights
                    )
            else:
                trim_user_idx, trim_item_idx, trim_weights = (
                    user_idx, item_idx, weights
                )
            uu_data, uu_indices, uu_indptr = kindling_core.build_user_user_graph(
                trim_user_idx, trim_item_idx, trim_weights,
                n_users=n_users, n_items=n_items,
                max_users_per_item=self.louvain_max_users_per_item,
                seed=self.random_state,
                weight_transform=self.louvain_weight_transform,
                min_edge_percentile=self.louvain_min_edge_percentile,
            )
            uu_data = np.asarray(uu_data, dtype=np.float32)
            uu_indices = np.asarray(uu_indices, dtype=np.int32)
            uu_indptr = np.asarray(uu_indptr, dtype=np.int32)
            # SBM init: pick between Louvain warm-start and random K-block
            # init. SBM can't grow past the starting block count — it only
            # reassigns nodes between existing blocks — so on sparse
            # graphs where Louvain under-clusters, random_k init gives
            # SBM more headroom.
            init_mode = self.dc_sbm_init_mode
            louv_init: np.ndarray | None = None
            if init_mode in ("louvain", "auto"):
                louv_assign, _louv_n, _modularity, _passes, _noise = (
                    kindling_core.fit_louvain_py(
                        uu_data, uu_indices, uu_indptr,
                        min_community_size=self.louvain_min_community_size,
                        max_passes=10,
                        modularity_tol=1e-6,
                        resolution=self.dc_sbm_warmstart_resolution,
                    )
                )
                louv_init = np.asarray(louv_assign, dtype=np.int64)
                positives = louv_init[louv_init >= 0]
                n_louv_blocks = int(np.unique(positives).size) if positives.size > 0 else 0
                profile["dc_sbm_louvain_blocks"] = n_louv_blocks
                # auto: fall through to random_k if Louvain under-clusters
                if init_mode == "auto" and n_louv_blocks < self.dc_sbm_random_k:
                    init_mode = "random_k"
            if init_mode == "random_k":
                rng = np.random.RandomState(self.random_state)
                louv_init = rng.randint(0, self.dc_sbm_random_k, size=n_users).astype(np.int64)
                profile["dc_sbm_init_mode_used"] = "random_k"
            else:
                profile["dc_sbm_init_mode_used"] = "louvain"
            assert louv_init is not None
            assignments, n_blocks, sbm_passes, sbm_noise_frac = (
                kindling_core.fit_dcsbm_py(
                    uu_data, uu_indices, uu_indptr,
                    init_assignments=louv_init,
                    max_passes=self.dc_sbm_max_passes,
                    min_internal_fraction=self.dc_sbm_min_internal_fraction,
                    move_threshold_pct=0.01,
                )
            )
            assignments = np.asarray(assignments, dtype=np.int64)
            n_personas_actual = n_blocks
            noise_frac = float(sbm_noise_frac)
            user_to_persona = assignments
            profile["dc_sbm_passes"] = int(sbm_passes)
            profile["dc_sbm_n_blocks"] = int(n_blocks)

        # Shared post-clustering: build persona index + per-persona cooc
        # if any method produced ≥1 cluster.
        if personas_enabled and n_personas_actual > 0:
            _sizes, _rates_csr, _tfidf_csr, _idf, distinctive = (
                kindling_core.build_persona_index_py(
                    assignments.tolist(),
                    user_idx.tolist(),
                    item_idx.tolist(),
                    n_personas=n_personas_actual,
                    n_items=n_items,
                    z_filter=1.5,
                )
            )
            persona_distinctive = [list(d) for d in distinctive]

            # ── Coherence filter (algorithm-agnostic). For each persona
            # compute the mean cooc[i,j] over its distinctive-item set;
            # drop personas below the configured percentile so their
            # members route to cooc base. Operates on the same cooc CSR
            # the base scorer uses, so coherence is in the same units as
            # base scores.
            coherence_scores = np.asarray(
                kindling_core.compute_persona_coherence_py(
                    cooc_data, cooc_indices, cooc_indptr,
                    distinctive_items=persona_distinctive,
                ),
                dtype=np.float64,
            )
            profile["persona_coherence"] = {
                "mean": float(coherence_scores.mean()) if coherence_scores.size else 0.0,
                "median": float(np.median(coherence_scores)) if coherence_scores.size else 0.0,
                "p25": float(np.percentile(coherence_scores, 25)) if coherence_scores.size else 0.0,
                "p75": float(np.percentile(coherence_scores, 75)) if coherence_scores.size else 0.0,
                "min": float(coherence_scores.min()) if coherence_scores.size else 0.0,
                "max": float(coherence_scores.max()) if coherence_scores.size else 0.0,
            }
            n_personas_kept = n_personas_actual
            persona_sizes_arr = np.asarray(_sizes, dtype=np.int64)
            # Pre-filter: drop personas that are too small (artificially
            # high coherence on rare items) OR too large (≈ global cooc,
            # adds no differentiation). Members of either get -1.
            max_size = int(self.coherence_max_persona_fraction * n_users)
            size_ok = (persona_sizes_arr >= self.coherence_min_persona_users) & (
                persona_sizes_arr <= max_size
            )
            if self.coherence_filter_percentile > 0.0 and coherence_scores.size > 0:
                # Threshold on personas that pass size + have >0 coherence.
                valid_mask = size_ok & (coherence_scores > 0.0)
                if valid_mask.any():
                    valid_coh = coherence_scores[valid_mask]
                    threshold = float(np.percentile(
                        valid_coh, self.coherence_filter_percentile * 100.0
                    ))
                    keep_persona = size_ok & (coherence_scores >= threshold)
                else:
                    # No persona passes size + coherence → keep none.
                    keep_persona = np.zeros(n_personas_actual, dtype=bool)
                # Reassign users in dropped personas to -1 (noise).
                pos_mask = assignments >= 0
                drop_user = pos_mask & ~keep_persona[assignments.clip(min=0)]
                if drop_user.any():
                    assignments = assignments.copy()
                    assignments[drop_user] = -1
                    user_to_persona = assignments
                n_personas_kept = int(keep_persona.sum())
                profile["persona_coherence"]["n_personas_kept"] = n_personas_kept
                profile["persona_coherence"]["filter_threshold"] = (
                    threshold if valid_mask.any() else 0.0
                )
                profile["persona_coherence"]["filter_percentile"] = (
                    self.coherence_filter_percentile
                )

            pc_data, pc_indices, pc_indptr, _pc_sizes = (
                kindling_core.build_persona_cooccurrence(
                    user_idx,
                    item_idx,
                    weights,
                    user_to_persona=assignments.tolist(),
                    n_users=n_users,
                    n_items=n_items,
                    n_personas=n_personas_actual,
                    kernel=plan["kernel"],
                    alpha=plan["alpha"],
                    half_life_days=plan["half_life_days"],
                    timestamps=timestamps_col,
                    min_persona_users=5,
                )
            )
            persona_cooc_data = [np.asarray(d, dtype=np.float32) for d in pc_data]
            persona_cooc_indices = [np.asarray(i, dtype=np.int32) for i in pc_indices]
            persona_cooc_indptr = [np.asarray(p, dtype=np.int32) for p in pc_indptr]
        if personas_enabled:
            profile["noise_fraction"] = float(noise_frac)
            profile["n_personas"] = int(n_personas_actual)
            profile["persona_method"] = persona_method

        # ── Boost layers. Each gets its own cooc-shaped adjacency —
        # i.e., a full DUPLICATE of the item-item CSR. On very large
        # catalogs that duplication is multi-GB for layers that have
        # never shown lift beyond the fused channels; gate them off.
        # (24GB machine + 360k-item amazon-book: base cooc + temporal
        # cooc + session structures together exceed physical RAM.)
        boost_layers_size_ok = n_items <= 100_000
        if not boost_layers_size_ok:
            profile["boost_layers_skipped"] = (
                f"catalog too large ({n_items} items > 100k): "
                "duplicate adjacency builds gated off"
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
                user_idx, item_idx, weights,
                n_users=n_users, n_items=n_items,
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
            session_idx = (
                interactions["session_id"].map(session_to_idx).to_numpy(dtype=np.int64)
            )
            sd, si, spt = kindling_core.build_session_cooccurrence(
                session_idx, item_idx, weights,
                n_sessions=len(session_ids), n_items=n_items,
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

        # ── Graph-regularized MF (GR-MF / graph_mf). Optional.
        # Builds the data-driven graph (A3 profile-gated: directional
        # cooc-derived-symmetric when sessions are inferable, else
        # co-ownership cooc fallback). Hierarchy graph plumbed but
        # currently always None (B2 extractor lands as a follow-on).
        gmf_user_factors: np.ndarray | None = None
        gmf_item_factors: np.ndarray | None = None
        gmf_data_graph_kind = "none"
        if self.use_graph_mf:
            data_graph, gmf_data_graph_kind = self._build_data_graph(
                interactions, item_to_idx, item_idx, n_items, cooc_data,
                cooc_indices, cooc_indptr,
            )
            hier_graph = self._build_hierarchy_graph(
                item_metadata, item_to_idx, n_items,
            )
            gmf_u, gmf_i, gmf_iters, gmf_deltas = kindling_core.fit_graph_mf_py(
                user_idx, item_idx, weights,
                n_users=n_users, n_items=n_items,
                dim=self.graph_mf_dim,
                n_iters=self.graph_mf_n_iters,
                alpha_data=self.graph_mf_alpha_data,
                alpha_hierarchy=self.graph_mf_alpha_hierarchy,
                regularization=0.01,
                als_alpha=40.0,
                seed=self.random_state,
                min_users=10,
                min_items=10,
                data_graph_data=data_graph[0] if data_graph else None,
                data_graph_indices=data_graph[1] if data_graph else None,
                data_graph_indptr=data_graph[2] if data_graph else None,
                hierarchy_graph_data=hier_graph[0] if hier_graph else None,
                hierarchy_graph_indices=hier_graph[1] if hier_graph else None,
                hierarchy_graph_indptr=hier_graph[2] if hier_graph else None,
            )
            gmf_user_factors = np.asarray(gmf_u, dtype=np.float64)
            gmf_item_factors = np.asarray(gmf_i, dtype=np.float64)

        # ── path_tail + path_basket: infer sessions, build indices.
        # Build is plan-aware: skip basket on rating-burst datasets, etc.
        # The Rust score_many kernels run at recommend time; here we just
        # feed the Python orchestrators that own session walking.
        tail_index: TailIndex | None = None
        basket_index: BasketIndex | None = None
        try:
            sess_inf = infer_sessions(interactions)
            sessions = list(
                sessions_from_interactions(interactions, sess_inf.session_ids)
            )
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

        self._state = V2FitState(
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
            personas_enabled=personas_enabled and n_personas_actual > 0,
            n_personas=n_personas_actual,
            user_to_persona=user_to_persona,
            persona_distinctive=persona_distinctive,
            persona_fit_threshold=self.persona_fit_threshold,
            persona_cooc_data=persona_cooc_data,
            persona_cooc_indices=persona_cooc_indices,
            persona_cooc_indptr=persona_cooc_indptr,
            als_user_factors=user_factors if self.als_as_boost else None,
            als_item_factors=item_factors_for_boost,
            gmf_user_factors=gmf_user_factors,
            gmf_item_factors=gmf_item_factors,
            gmf_role=self.graph_mf_role,
            gmf_data_graph_kind=gmf_data_graph_kind,
            signal_kind=signal_kind,
            als_ran=run_als,
            persona_method_used=persona_method if personas_enabled else "none",
            boost_layer_adjacencies=boost_adj,
            z_threshold=2.5,
            boost_multiplier=3.0,
            fit_seconds=time.perf_counter() - t0,
            profile=profile,
        )
        return self

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    def recommend(self, entity_id: object, n: int = 10) -> list[RecommendationV2]:
        if self._state is None:
            raise RuntimeError("EngineV2 not fitted. Call .fit(interactions) first.")
        owned = self._state.owned_by_entity.get(entity_id)
        if owned is None or owned.size == 0:
            return []
        return self._recommend_core(owned, entity_id, n)

    def recommend_for_items(
        self, seed_item_ids: object, n: int = 10
    ) -> list[RecommendationV2]:
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
            raise RuntimeError("EngineV2 not fitted. Call .fit(interactions) first.")
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

    def _cold_recommend(self, n: int) -> list[RecommendationV2]:
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
            RecommendationV2(
                item_id=st.item_ids[int(c)], score=float(scores[c]),
                base_kind="cold_popularity",
            )
            for c in top
        ]

    def _recommend_core(
        self, owned: np.ndarray, entity_id: object, n: int, pop_prior: float = 0.0
    ) -> list[RecommendationV2]:
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
                base_vec = np.concatenate(
                    [base_vec, np.zeros(st.n_items - base_vec.size)]
                )
            ease_scores_full = self._blend_channels(
                st, owned, base_vec,
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
                st, owned, cooc_full,
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
                st.cooc_data, st.cooc_indices, st.cooc_indptr,
                owned_indices=owned.tolist(),
                budget=self.retrieval_budget,
                include_owned=False,
            )
            cand_ids = list(cand_ids)
            base_kind = ""
        if not cand_ids:
            return []

        # ── 2. Base scores. Fused paths (ease / cooc_fused) bypass the
        # cooc/persona routing; the legacy path applies two-gate persona
        # routing.
        if ease_scores_full is not None:
            base = ease_scores_full[np.asarray(cand_ids, dtype=np.int64)]
        else:
            base_kind, base = self._compute_base(entity_id, owned, cand_ids)

        # ── 3. Layered scoring.
        layer_specs = self._build_layer_specs(entity_id, owned, cand_ids)
        composite = kindling_core.layered_score_py(
            base, layer_specs,
            z_threshold=st.z_threshold,
            boost_multiplier=st.boost_multiplier,
        )
        composite = np.asarray(composite)

        # ── 4. Top-N (skip repeat module — not yet ported).
        # The positivity filter encodes cooc semantics (score 0 = no
        # co-occurrence evidence). EASE weights are signed, so a small
        # negative composite can still be the best available candidate.
        order = np.argsort(-composite)[:n]
        out: list[RecommendationV2] = []
        for rank, idx in enumerate(order):
            if composite[idx] <= 0.0 and base_kind not in ("ease", "cooc_fused"):
                continue
            cid = cand_ids[idx]
            out.append(RecommendationV2(
                item_id=st.item_ids[cid],
                score=float(composite[idx]),
                base_kind=base_kind,
            ))

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
                    out.append(RecommendationV2(
                        item_id=st.item_ids[i],
                        score=float(cs[i]),
                        base_kind="cold_content",
                    ))
        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _blend_channels(
        self,
        st: V2FitState,
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
        uu_on = (
            st.uu_users_data is not None
            and st.user_cf_alpha > 0.0
            and owned.size > 0
        )
        if not (trend_on or trans_on or content_on or last_on or uu_on):
            return scores_full
        std = scores_full.std()
        if std > 0:
            scores_full = (scores_full - scores_full.mean()) / std
        if trend_on:
            scores_full = scores_full + st.trend_alpha * st.trend_z
        if uu_on:
            # Otsuka-Ochiai k-NN: overlap counts via the inverted index,
            # normalized by sqrt(deg_u · deg_v); top-k neighbors vote
            # for their items, weighted by similarity.
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
                sims = counts[nz] / (
                    np.sqrt(st.uu_user_deg[nz]) * np.sqrt(max(owned.size, 1))
                )
                if nz.size > st.user_cf_k:
                    keep = np.argpartition(-sims, st.user_cf_k)[: st.user_cf_k]
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
            last_row = st.ease_b[int(owned[-1])].astype(np.float64)
            if last_row.size < st.n_items:
                last_row = np.concatenate(
                    [last_row, np.zeros(st.n_items - last_row.size)]
                )
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
                coldness = (
                    st.content_coldness
                    if st.content_coldness is not None
                    else 1.0
                )
                scores_full = scores_full + st.content_alpha * coldness * cz
        if trans_on:
            trans = np.zeros(st.n_items, dtype=np.float64)
            recent = owned[::-1][: st.transition_last_k]
            for j, item in enumerate(recent):
                s_ = int(st.trans_indptr[item])
                e_ = int(st.trans_indptr[item + 1])
                if e_ > s_:
                    trans[st.trans_indices[s_:e_]] += (
                        st.transition_decay ** j
                    ) * st.trans_data[s_:e_]
            t_std = trans.std()
            if t_std > 0:
                scores_full = scores_full + st.transition_alpha * (
                    (trans - trans.mean()) / t_std
                )
        return scores_full

    def _compute_base(
        self, entity_id: object, owned: np.ndarray, cand_ids: list[int]
    ) -> tuple[str, np.ndarray]:
        """Two-gate routing. Returns (base_kind, base_scores).

        When `graph_mf_role == "base"` and GR-MF factors are fitted,
        replace the cooc base with `user_factor · item_factor` over the
        candidate pool. Otherwise the standard cooc/persona_cooc routing.
        """
        st = self._state
        assert st is not None
        # GR-MF as base: skip the cooc/persona routing entirely.
        if (
            st.gmf_role == "base"
            and st.gmf_user_factors is not None
            and st.gmf_item_factors is not None
        ):
            uidx = st.entity_to_user_idx.get(entity_id, -1)
            if 0 <= uidx < st.gmf_user_factors.shape[0]:
                u_vec = st.gmf_user_factors[uidx]
                cand_arr = np.asarray(cand_ids, dtype=np.int64)
                item_vecs = st.gmf_item_factors[cand_arr]
                base = (item_vecs @ u_vec).astype(np.float64)
                return "graph_mf", base
        cluster_id = -1
        if st.personas_enabled and len(st.user_to_persona):
            user_idx = st.entity_to_user_idx.get(entity_id, -1)
            if 0 <= user_idx < len(st.user_to_persona):
                cluster_id = int(st.user_to_persona[user_idx])
        # Persona-fit gate.
        use_persona = False
        if cluster_id >= 0 and cluster_id < len(st.persona_distinctive):
            distinctive = st.persona_distinctive[cluster_id]
            use_persona = kindling_core.should_use_persona_base(
                cluster_id, owned.tolist(), distinctive,
                threshold=st.persona_fit_threshold,
            )
        if use_persona and cluster_id < len(st.persona_cooc_data):
            base = kindling_core.cooccurrence_signal(
                st.persona_cooc_data[cluster_id],
                st.persona_cooc_indices[cluster_id],
                st.persona_cooc_indptr[cluster_id],
                owned_indices=owned.tolist(),
                candidate_indices=cand_ids,
            )
            return "persona_cooc", np.asarray(base)
        base = kindling_core.cooccurrence_signal(
            st.cooc_data, st.cooc_indices, st.cooc_indptr,
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
        clean. ALS-as-boost is opt-in (`als_as_boost=True`) and surfaces
        below as a dense layer; it's empirically degenerate on the
        datasets we've measured.
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
                data, indices, indptr,
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
                basket_scores = st.basket_index.score_many(
                    cand_external, query_basket=query_basket
                )
                out.append((np.asarray(basket_scores, dtype=np.float64), "nonzero"))

        # GR-MF-as-boost (dense, candidate-pool z-mode). Mirrors the
        # ALS-as-boost path but uses GR-MF factors which are graph-
        # regularized — same scoring shape (user_factor · item_factor).
        if (
            st.gmf_role == "boost"
            and st.gmf_user_factors is not None
            and st.gmf_item_factors is not None
        ):
            uidx = st.entity_to_user_idx.get(entity_id, -1)
            if 0 <= uidx < st.gmf_user_factors.shape[0]:
                u_vec = st.gmf_user_factors[uidx]
                cand_arr = np.asarray(cand_ids, dtype=np.int64)
                item_vecs = st.gmf_item_factors[cand_arr]
                gmf_scores = (item_vecs @ u_vec).astype(np.float64)
                out.append((gmf_scores, "pool"))

        # ALS-as-boost (dense, candidate-pool z-mode per the v2 PRD).
        # score(c) = user_factor[entity] · item_factor[c]. Fires when
        # the user-item dot product stands out z-significantly across
        # the retrieved pool.
        if (
            st.als_user_factors is not None
            and st.als_item_factors is not None
        ):
            user_idx_int = st.entity_to_user_idx.get(entity_id, -1)
            if 0 <= user_idx_int < st.als_user_factors.shape[0]:
                u_vec = st.als_user_factors[user_idx_int]
                cand_array = np.asarray(cand_ids, dtype=np.int64)
                item_vecs = st.als_item_factors[cand_array]
                als_scores = (item_vecs @ u_vec).astype(np.float64)
                out.append((als_scores, "pool"))

        return out

    def _build_hierarchy_graph(
        self,
        item_metadata: pd.DataFrame | None,
        item_to_idx: dict[object, int],
        n_items: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """B2: build a flat-partition hierarchy graph from item metadata.

        Strategy: cluster items by `store` (brand) when present. Items in
        the same brand → undirected edge with weight 1.0.

        Caveat: this is *brand-as-hierarchy*, NOT a tree-structured
        hierarchy. The 2023 Amazon Reviews dataset (only available HF
        mirror) ships `categories: []` for All_Beauty — the original
        2018 hierarchical category tree was dropped in the rewrite. So
        brand is the strongest flat-partition signal we can extract on
        amazon-beauty today; a real hierarchy with ancestor/descendant
        edges would need either a different metadata source or hand-
        curated taxonomies.

        Returns CSR triple or None when no usable metadata.
        """
        if item_metadata is None or len(item_metadata) == 0:
            return None
        if "store" not in item_metadata.columns:
            return None

        from collections import defaultdict

        # Bucket items in our catalog by their brand.
        by_brand: dict[object, list[int]] = defaultdict(list)
        for _, row in item_metadata.iterrows():
            item_id = row["item_id"]
            store = row.get("store")
            if not store or pd.isna(store):
                continue
            idx = item_to_idx.get(item_id, -1)
            if idx < 0:
                continue
            by_brand[store].append(idx)

        # Drop singleton brands (no edges to add). Cap brand size to
        # avoid quadratic blowup on dominant brands.
        max_brand_size = 100
        rows_acc: dict[int, list[tuple[int, float]]] = {}
        for brand_items in by_brand.values():
            if len(brand_items) < 2:
                continue
            members = brand_items[:max_brand_size]
            # Symmetric clique edges with unit weight.
            for a in members:
                for b in members:
                    if a == b:
                        continue
                    rows_acc.setdefault(a, []).append((b, 1.0))
        if not rows_acc:
            return None
        # Pack as CSR.
        out_data: list[float] = []
        out_indices: list[int] = []
        out_indptr: list[int] = [0]
        for i in range(n_items):
            row = rows_acc.get(i, [])
            row.sort(key=lambda t: t[0])
            # Dedup (a brand member could appear multiple times if duplicated).
            seen: set[int] = set()
            for j, w in row:
                if j in seen:
                    continue
                seen.add(j)
                out_indices.append(j)
                out_data.append(w)
            out_indptr.append(len(out_indices))
        return (
            np.asarray(out_data, dtype=np.float32),
            np.asarray(out_indices, dtype=np.int32),
            np.asarray(out_indptr, dtype=np.int32),
        )

    def _build_data_graph(
        self,
        interactions: pd.DataFrame,
        item_to_idx: dict[object, int],
        item_idx: np.ndarray,
        n_items: int,
        cooc_data: np.ndarray,
        cooc_indices: np.ndarray,
        cooc_indptr: np.ndarray,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        str,
    ]:
        """A3: profile-gated data graph for GR-MF.

        Preference order:
          1. Explicit session_id column → directional cooc + symmetrize.
          2. Timestamps present → infer sessions via GMM gap detection,
             then directional cooc + symmetrize.
          3. Fallback to co-ownership cooc CSR (always available).

        Returns ((data, indices, indptr), kind_label).
        """
        sidx: np.ndarray | None = None
        n_sessions: int = 0
        kind: str = "none"

        # Path 1: explicit session_id column.
        if "session_id" in interactions.columns:
            session_ids = pd.Index(interactions["session_id"].unique())
            session_to_idx = {s: i for i, s in enumerate(session_ids)}
            sidx = (
                interactions["session_id"]
                .map(session_to_idx)
                .to_numpy(dtype=np.int64)
            )
            n_sessions = len(session_ids)
            kind = "directional_explicit"

        # Path 2: timestamp-inferred sessions (GMM gap detection).
        elif "timestamp" in interactions.columns:
            try:
                inferred = infer_sessions(interactions)
                sidx_arr = np.asarray(inferred.session_ids, dtype=np.int64)
                if sidx_arr.size > 0 and sidx_arr.max() >= 0:
                    sidx = sidx_arr
                    n_sessions = int(sidx_arr.max()) + 1
                    kind = "directional_inferred"
            except Exception as exc:  # pragma: no cover — defensive
                import warnings
                warnings.warn(
                    f"session inference failed ({exc!r}); "
                    "falling back to co-ownership graph for graph_mf",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if sidx is not None and n_sessions > 0:
            ts = (
                interactions["timestamp"].to_numpy(dtype=np.float64)
                if "timestamp" in interactions.columns
                else None
            )
            ws = np.ones(len(interactions), dtype=np.float32)
            d_data, d_indices, d_indptr = kindling_core.build_directional_cooc(
                sidx, item_idx, ws,
                n_sessions=n_sessions,
                n_items=n_items,
                timestamps=ts,
            )
            sym_data, sym_indices, sym_indptr = kindling_core.symmetrize_via_transpose(
                np.asarray(d_data, dtype=np.float32),
                np.asarray(d_indices, dtype=np.int32),
                np.asarray(d_indptr, dtype=np.int32),
            )
            return (
                (
                    np.asarray(sym_data, dtype=np.float32),
                    np.asarray(sym_indices, dtype=np.int32),
                    np.asarray(sym_indptr, dtype=np.int32),
                ),
                kind,
            )

        # Path 3: existing co-ownership cooc.
        return ((cooc_data, cooc_indices, cooc_indptr), "co-ownership")

    def _fit_factors(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        weights: np.ndarray,
        n_users: int,
        n_items: int,
        n_factors: int = 32,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Compute user factors (and optionally item factors).

        Returns ``(user_factors, item_factors)`` where ``item_factors`` is
        None when the active method doesn't produce them. Routing:

        - ``hdbscan_factor_method == "als"`` always produces both
          (user, item) factors — full implicit ALS.
        - ``hdbscan_factor_method == "svd"`` produces only user factors.
          When ``als_as_boost`` is True, an additional ALS run is needed
          to get item factors for the boost layer; that branch returns
          (svd_users, als_items).
        """
        k = min(n_factors, max(2, min(n_users, n_items) - 1))
        if self.hdbscan_factor_method == "als":
            user_factors, item_factors, _losses = kindling_core.fit_als_py(
                user_idx, item_idx, weights,
                n_users=n_users, n_items=n_items,
                n_factors=k, n_iters=5,
                alpha=40.0, regularization=0.01,
                seed=self.random_state,
            )
            return (
                np.asarray(user_factors, dtype=np.float64),
                np.asarray(item_factors, dtype=np.float64),
            )
        # SVD path.
        user_factors = kindling_core.truncated_svd_py(
            user_idx, item_idx, weights,
            n_users=n_users, n_items=n_items,
            n_factors=k, n_oversample=10, n_power_iters=1,
            seed=self.random_state,
        )
        user_factors = np.asarray(user_factors, dtype=np.float64)
        item_factors: np.ndarray | None = None
        if self.als_as_boost:
            # Pay for ALS only if we actually need item factors for boost.
            _u_als, item_factors_als, _losses = kindling_core.fit_als_py(
                user_idx, item_idx, weights,
                n_users=n_users, n_items=n_items,
                n_factors=k, n_iters=5,
                alpha=40.0, regularization=0.01,
                seed=self.random_state,
            )
            item_factors = np.asarray(item_factors_als, dtype=np.float64)
        return user_factors, item_factors

    def _profile(
        self,
        interactions: pd.DataFrame,
        weights: np.ndarray,
        n_users: int,
        n_items: int,
    ) -> dict[str, Any]:
        density = (
            float(len(interactions)) / max(n_users * n_items, 1)
            if n_users and n_items
            else 0.0
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
        kernel = "pure_count" if profile["rating_burst_detected"] else (
            "hybrid_temporal" if profile["has_timestamps"] else "pure_count"
        )
        # Personas enabled with adequate user count.
        personas_enabled = profile["n_users"] >= self.persona_min_users
        # Boost layer enable/disable per profile.
        enabled = []
        if profile["has_timestamps"] and not profile["rating_burst_detected"]:
            enabled.append("temporal_cooccurrence")
        if profile["has_sessions"] and profile["deep_session_fraction"] >= 0.30:
            enabled.append("session_cooccurrence")
        # path_tail / path_basket / interaction_network / ALS / cosine /
        # lightgcn deferred until their builders / scorers are wired here.
        return {
            "kernel": kernel,
            "alpha": 1.0,
            "half_life_days": half_life_days,
            "personas_enabled": personas_enabled,
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
            "personas_enabled": st.personas_enabled,
            "n_personas": st.n_personas,
            "kernel": st.kernel,
            "half_life_days": st.half_life_days,
            "enabled_boost_layers": st.enabled_boost_layers,
            "z_threshold": st.z_threshold,
            "boost_multiplier": st.boost_multiplier,
            "profile": st.profile,
            "graph_mf_active": st.gmf_user_factors is not None,
            "graph_mf_role": st.gmf_role if st.gmf_user_factors is not None else None,
            "graph_mf_data_graph_kind": st.gmf_data_graph_kind,
            "signal_kind": st.signal_kind,
            "als_ran": st.als_ran,
            "use_als_setting": self.use_als,
            "persona_method_used": st.persona_method_used,
            "persona_method_setting": self.persona_method,
            "base_scorer_used": st.base_scorer_used,
            "base_scorer_setting": self.base_scorer,
        }
