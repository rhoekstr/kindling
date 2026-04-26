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
    # Base layer: global cooc CSR
    cooc_data: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    cooc_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    cooc_indptr: np.ndarray = field(default_factory=lambda: np.array([0], dtype=np.int32))
    # Personas (optional)
    personas_enabled: bool = False
    n_personas: int = 0
    user_to_persona: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    persona_distinctive: list[list[int]] = field(default_factory=list)
    persona_fit_threshold: float = 0.70
    # Per-persona cooc CSRs
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
        retrieval_budget: int = 200,
        random_state: int = 0,
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
        self._state: V2FitState | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, interactions: pd.DataFrame) -> "EngineV2":
        t0 = time.perf_counter()
        # Same contract as v1: validate → canonicalize → preprocess.
        schema = validate_interactions(interactions)
        canonical = canonicalize(interactions, schema)
        canonical, _ctx = preprocess_interactions(canonical, use_ratings=None)
        interactions = canonical
        weights = weights_of(interactions).astype(np.float32)
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

        # ── Base cooc.
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
        if personas_enabled:
            # Compute user factors. For now: TruncatedSVD as a stand-in
            # for ALS implicit (Phase 1f port). Cheap and ABI-safe.
            user_factors = self._fit_user_factors(
                user_idx, item_idx, weights, n_users, n_items
            )
            assignments, _probs, n_personas_actual, noise_frac = kindling_core.fit_hdbscan_py(
                user_factors,
                min_cluster_size=max(15, int(0.005 * n_users)),
                min_samples=15,
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

        # ── item_cosine: derive from cooc CSR + per-item user counts.
        # Always builds when we have the cooc base; cheap (one matmul-shaped
        # transform). Always-on per the v2 boost-layer table.
        item_counts = np.bincount(
            item_idx, minlength=n_items
        ).astype(np.int64)
        ic_data, ic_indices, ic_indptr = kindling_core.build_item_cosine(
            cooc_data, cooc_indices, cooc_indptr,
            item_counts,
            top_k=200,
            min_cosine=0.01,
        )
        boost_adj["item_cosine"] = (
            np.asarray(ic_data, dtype=np.float32),
            np.asarray(ic_indices, dtype=np.int32),
            np.asarray(ic_indptr, dtype=np.int32),
        )

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
        # NOTE: in the v2 boost-layer architecture, layers contribute
        # only when items are *already in the cooc-retrieved pool*. Boost
        # magnitudes are calibrated to ~3 rank positions within the
        # base top-K, so they can't elevate out-of-pool items into top-K.
        # path_tail / interaction_network etc. are useful only for
        # candidates that BOTH cooc retrieves AND the layer ranks high.
        # On no-session datasets where the two distributions don't
        # intersect (e.g., amazon-beauty), those layers don't fire.
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
        """Two-gate routing. Returns (base_kind, base_scores)."""
        st = self._state
        assert st is not None
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
        """Build (layer_scores, z_mode) tuples for the layered scorer."""
        st = self._state
        assert st is not None
        out: list[tuple[np.ndarray, str]] = []
        # Cooc-shaped layers (cosine, temporal_cooc, session_cooc) all use
        # the same signal kernel against an item-item adjacency CSR.
        for layer_name in [
            *st.enabled_boost_layers,
            *(["item_cosine"] if "item_cosine" in st.boost_layer_adjacencies
              and "item_cosine" not in st.enabled_boost_layers else []),
        ]:
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

        # path_tail: sparse, queries the user's most-recent item.
        if st.tail_index is not None and st.tail_index.counts:
            history = st.history_by_entity.get(entity_id, ())
            last_item_internal = None
            if history:
                last_internal = st.item_to_idx.get(history[-1], -1)
                last_item_external = history[-1]
                if last_internal >= 0:
                    last_item_internal = last_item_external
            if last_item_internal is not None:
                # tail_index.score_many takes external item_ids.
                cand_external = [st.item_ids[ci] for ci in cand_ids]
                tail_scores = st.tail_index.score_many(
                    cand_external, last_item=last_item_internal
                )
                out.append((np.asarray(tail_scores, dtype=np.float64), "nonzero"))

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

        return out

    def _fit_user_factors(
        self,
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        weights: np.ndarray,
        n_users: int,
        n_items: int,
        n_factors: int = 32,
    ) -> np.ndarray:
        """Compute user factors via Rust implicit ALS (Hu, Koren, Volinsky 2008).

        Returns float64 (n_users, n_factors) suitable as HDBSCAN input.
        """
        # Cap factors to feasible value for tiny datasets.
        k = min(n_factors, max(2, min(n_users, n_items) - 1))
        # 5 iters is sufficient for HDBSCAN inputs — factors converge
        # to taste-coherent clusters quickly even before full convergence.
        user_factors, _item_factors, _losses = kindling_core.fit_als_py(
            user_idx, item_idx, weights,
            n_users=n_users, n_items=n_items,
            n_factors=k,
            n_iters=5,
            alpha=40.0,
            regularization=0.01,
            seed=self.random_state,
        )
        return np.asarray(user_factors, dtype=np.float64)

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
        }
