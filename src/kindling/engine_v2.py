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
    # Rating-signal classification + resolved use_als decision.
    signal_kind: str = "unknown"        # "binary" | "counts" | "ratings" | "forced_*"
    als_ran: bool = False               # whether ALS actually ran in this fit
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
        timestamps_col = (
            interactions["timestamp"].to_numpy(dtype=np.float64)
            if "timestamp" in interactions.columns
            else None
        )

        # owned_by_entity + history (timestamp-ordered) per entity.
        owned_by_entity: dict[object, np.ndarray] = {}
        history_by_entity: dict[object, tuple[object, ...]] = {}
        sort_col = "timestamp" if "timestamp" in interactions.columns else None
        for entity, group in interactions.groupby("entity_id", sort=False):
            if sort_col is not None:
                group = group.sort_values(sort_col, kind="mergesort")
            owned_by_entity[entity] = (
                group["item_id"].map(item_to_idx).dropna().astype(np.int64).to_numpy()
            )
            history_by_entity[entity] = tuple(group["item_id"].tolist())

        # ── Profile + Plan decisions.
        profile = self._profile(interactions, weights, n_users, n_items)
        plan = self._plan(profile)

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

        # ── Personas (if enabled).
        personas_enabled = bool(plan["personas_enabled"]) and n_users >= self.persona_min_users
        n_personas_actual = 0
        user_to_persona = np.array([], dtype=np.int64)
        persona_distinctive: list[list[int]] = []
        persona_cooc_data: list[np.ndarray] = []
        persona_cooc_indices: list[np.ndarray] = []
        persona_cooc_indptr: list[np.ndarray] = []
        # ALS-as-boost requires item factors even if personas aren't on.
        # SVD-for-HDBSCAN cleanly skips ALS when boost is also off.
        item_factors_for_boost: np.ndarray | None = None
        user_factors: np.ndarray | None = None
        need_factors = personas_enabled or self.als_as_boost
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
        if personas_enabled:
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
            if n_personas_actual > 0:
                # Build persona index (rates → z-filter → distinctive_items → TF-IDF → L2).
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
                # Build per-persona cooc.
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
            profile["noise_fraction"] = float(noise_frac)
            profile["n_personas"] = int(n_personas_actual)

        # ── Boost layers. Each gets its own cooc-shaped adjacency.
        boost_adj: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        # temporal_cooccurrence is just cooc with hybrid_temporal kernel — only
        # built when timestamps present and not rating-burst.
        if "temporal_cooccurrence" in plan["enabled_boost_layers"] and timestamps_col is not None:
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
        if "session_cooccurrence" in plan["enabled_boost_layers"] and "session_id" in interactions.columns:
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
            n_items=n_items,
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
        st = self._state
        owned = st.owned_by_entity.get(entity_id)
        if owned is None or owned.size == 0:
            return []

        # ── 1. Retrieve candidate pool via cooc.
        # Boost layers refine ranking within this pool but cannot promote
        # items outside it; some path-favored items will be missed when
        # they fall below the cooc top-K cut. We bump retrieval_budget
        # (default 500) to cut that miss rate without expanding to a
        # multi-retriever fusion stage.
        cand_ids, _scores = kindling_core.cooccurrence_retrieve(
            st.cooc_data, st.cooc_indices, st.cooc_indptr,
            owned_indices=owned.tolist(),
            budget=self.retrieval_budget,
            include_owned=False,
        )
        if not cand_ids:
            return []
        cand_ids = list(cand_ids)

        # ── 2. Two-gate base routing.
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
        order = np.argsort(-composite)[:n]
        out: list[RecommendationV2] = []
        for rank, idx in enumerate(order):
            if composite[idx] <= 0.0:
                continue
            cid = cand_ids[idx]
            out.append(RecommendationV2(
                item_id=st.item_ids[cid],
                score=float(composite[idx]),
                base_kind=base_kind,
            ))
        return out

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

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
        }
