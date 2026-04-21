"""Public Engine - the user-facing class wiring the three stages.

Phase 2 upgrade from Phase 1:
  - Sessions inferred via GMM or explicit column (ingest.sessions).
  - Path structures (PathTree, TailIndex, BasketIndex) built per-session.
  - Path-endpoint retriever added alongside co-occurrence.
  - Heuristic blend over (full, tail, basket, cooccurrence) with path-family
    Gram-Schmidt decorrelation fit on the chronological tail 10% of train.
  - Per-candidate signal scores feed the ``Explanation.debug()`` surface.

The public ``recommend`` signature is unchanged. The heuristic weights are a
placeholder for Phase 3's Bayesian posterior; the decorrelation basis
persists with the engine and applies verbatim at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from kindling.blend.bayesian import BayesianBlend
from kindling.blend.decorrelate import DecorrelationBasis, fit_decorrelation
from kindling.blend.diagnostics import DiagnosticsReport, run_diagnostics
from kindling.blend.heuristic import PATH_FAMILY, HeuristicBlend, SignalFeatures
from kindling.blend.likelihoods import (
    LikelihoodProtocol,
    ListwiseCalibration,
)
from kindling.blend.outcome_builder import OutcomeBuildConfig, build_outcomes
from kindling.blend.priors import DataFeatures, construct_prior
from kindling.explain import Explanation
from kindling.graph.item_graph import ItemGraph, build_item_graph
from kindling.ingest.contract import (
    InteractionSchema,
    canonicalize,
    validate_interactions,
)
from kindling.ingest.sessions import SessionInference, infer_sessions
from kindling.lifecycle.decay import DecayProtocol, ExponentialDecay
from kindling.path._sessions import sessions_from_interactions
from kindling.path.basket_index import BasketIndex, BasketSimilarity, build_basket_index
from kindling.path.path_tree import PathTree, build_path_tree
from kindling.path.tail_index import TailIndex, build_tail_index
from kindling.rerank.constraints import ConstraintPredicate, apply_constraints
from kindling.retrieve.cooccurrence import CoOccurrenceRetriever
from kindling.retrieve.path_endpoint import PathEndpointRetriever
from kindling.retrieve.protocol import Candidate

DEFAULT_RETRIEVAL_BUDGET = 500
# Fixed path-family-first signal order for decorrelation (PRD §6.2).
SIGNAL_ORDER: tuple[str, ...] = ("path_full", "path_tail", "path_basket", "cooccurrence")
# Cap the query basket at recommend time. Full-history users (ML-1M power
# raters with 500+ ratings) otherwise trigger posting-list unions over
# essentially every training observation, pushing basket scoring to minutes.
# Using the most recent 50 items matches the "recently-relevant composition"
# intuition of the basket mechanism without the quadratic blowup.
MAX_QUERY_BASKET_SIZE = 50


@dataclass(frozen=True)
class Recommendation:
    """A single recommendation.

    Phase 3 surfaces a Bayesian credible interval on the score. The
    interval is derived from the Dirichlet posterior over blend weights
    and is labelled ``credible_interval`` - not ``confidence_interval`` -
    because it is a Bayesian credible interval, not a frequentist
    coverage guarantee. Conformal prediction (v1.x) will add the latter.
    """

    item_id: object
    score: float
    explanation: Explanation
    credible_interval: tuple[float, float] | None = None
    credible_coverage: float | None = None


class EngineNotFittedError(RuntimeError):
    pass


class Engine:
    """The primary kindling entry point."""

    def __init__(
        self,
        retrieval_budget: int = DEFAULT_RETRIEVAL_BUDGET,
        decay: DecayProtocol | None = None,
        max_path_prefix: int = 3,
        max_history_for_recommend: int = 5,
        basket_similarity: BasketSimilarity = BasketSimilarity.COVERAGE,
        use_bayesian_blend: bool = True,
        likelihood: LikelihoodProtocol | None = None,
        credible_coverage: float = 0.9,
        seed: int = 0,
        vi_max_iter: int = 300,
    ) -> None:
        self.retrieval_budget = retrieval_budget
        self.decay: DecayProtocol = (
            decay
            if decay is not None
            else cast(DecayProtocol, ExponentialDecay(half_life_days=180.0))
        )
        self.max_path_prefix = max_path_prefix
        self.max_history_for_recommend = max_history_for_recommend
        self.basket_similarity = basket_similarity

        # Bayesian blend configuration (Phase 3).
        self.use_bayesian_blend = use_bayesian_blend
        self.likelihood: LikelihoodProtocol = (
            likelihood
            if likelihood is not None
            else cast(LikelihoodProtocol, ListwiseCalibration())
        )
        self.credible_coverage = credible_coverage
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.vi_max_iter = vi_max_iter

        self._schema: InteractionSchema | None = None
        self._interactions: pd.DataFrame | None = None
        self._reference_timestamp: float | None = None
        self._session_inference: SessionInference | None = None

        self._item_graph: ItemGraph | None = None
        self._tail_index: TailIndex | None = None
        self._path_tree: PathTree | None = None
        self._basket_index: BasketIndex | None = None

        self._cooc_retriever: CoOccurrenceRetriever | None = None
        self._path_retriever: PathEndpointRetriever | None = None

        self._heuristic_blend = HeuristicBlend()
        self._bayesian_blend: BayesianBlend | None = None
        self._diagnostics: DiagnosticsReport | None = None
        self._owned_by_entity: dict[object, np.ndarray] = {}
        self._history_by_entity: dict[object, tuple[object, ...]] = {}

    # ---- fitting ----------------------------------------------------------

    def fit(self, interactions: pd.DataFrame) -> Engine:
        """Validate, canonicalize, and build derived structures."""
        schema = validate_interactions(interactions)
        self._schema = schema
        self._interactions = canonicalize(interactions, schema)
        self._reference_timestamp = _reference_timestamp_from(self._interactions)

        self._session_inference = infer_sessions(self._interactions)
        sessions = list(
            sessions_from_interactions(
                self._interactions,
                self._session_inference.session_ids,
            )
        )

        self._item_graph = build_item_graph(self._interactions)
        self._tail_index = build_tail_index(
            sessions, decay=self.decay, reference_timestamp=self._reference_timestamp
        )
        self._path_tree = build_path_tree(
            sessions,
            max_prefix=self.max_path_prefix,
            decay=self.decay,
            reference_timestamp=self._reference_timestamp,
        )
        self._basket_index = build_basket_index(
            sessions, decay=self.decay, reference_timestamp=self._reference_timestamp
        )

        self._cooc_retriever = CoOccurrenceRetriever(self._item_graph)
        self._path_retriever = PathEndpointRetriever(self._path_tree, self._tail_index)

        self._owned_by_entity = {
            entity: group["item_id"].to_numpy()
            for entity, group in self._interactions.groupby("entity_id", sort=False)
        }
        self._history_by_entity = _build_histories(self._interactions, schema)

        path_basis = _fit_path_family_decorrelation(
            interactions=self._interactions,
            item_graph=self._item_graph,
            tail_index=self._tail_index,
            path_tree=self._path_tree,
            basket_index=self._basket_index,
            basket_similarity=self.basket_similarity,
            history_by_entity=self._history_by_entity,
            owned_by_entity=self._owned_by_entity,
        )
        self._heuristic_blend.path_basis = path_basis

        # Bayesian blend: prior from data features + VI fit on the
        # chronological tail.
        if self.use_bayesian_blend:
            self._fit_bayesian_blend(path_basis, sessions_count=len(sessions))

        return self

    def _fit_bayesian_blend(
        self,
        path_basis: DecorrelationBasis | None,
        sessions_count: int,
    ) -> None:
        """Construct the data-adaptive prior and run VI on the tail outcomes."""
        assert self._interactions is not None
        assert self._item_graph is not None
        features = self._compute_data_features(sessions_count)
        alpha = construct_prior(signal_names=SIGNAL_ORDER, features=features)
        self._bayesian_blend = BayesianBlend.from_prior(
            signal_names=SIGNAL_ORDER,
            prior_alpha=alpha,
            path_basis=path_basis,
        )

        outcomes = build_outcomes(
            interactions=self._interactions,
            compute_signals=self._build_features_for_outcome,
            config=OutcomeBuildConfig(),
            rng=self._rng,
        )
        if outcomes.n_outcomes == 0:
            return

        self._bayesian_blend.fit_posterior(
            batch=outcomes,
            likelihood=self.likelihood,
            rng=np.random.default_rng(self.seed),
            max_iter=self.vi_max_iter,
        )
        self._diagnostics = run_diagnostics(
            blend=self._bayesian_blend,
            batch=outcomes,
            likelihood=self.likelihood,
            rng=np.random.default_rng(self.seed + 1),
        )

    def _compute_data_features(self, sessions_count: int) -> DataFeatures:
        assert self._interactions is not None
        assert self._item_graph is not None
        n_items = self._item_graph.n_items
        n_entities = int(self._interactions["entity_id"].nunique())
        n_interactions = len(self._interactions)
        max_edges = max(n_items * (n_items - 1), 1)
        density = self._item_graph.n_edges / max_edges
        # Clustering coefficient estimate: use the fraction of edges that
        # share a neighbor. Cheap-enough approximation for prior construction;
        # exact clustering is only needed if we ever learn the coefficient.
        clustering = _approx_clustering_coefficient(self._item_graph)
        session_density = n_interactions / max(sessions_count, 1)
        return DataFeatures(
            graph_density=float(density),
            clustering_coefficient=float(clustering),
            session_density=float(session_density),
            catalog_to_entity_ratio=float(n_items) / max(n_entities, 1),
            n_interactions=n_interactions,
        )

    def _build_features_for_outcome(
        self,
        entity: object,
        items: list[object],
        owned_arr: np.ndarray,
    ) -> SignalFeatures:
        """Used by outcome_builder to materialize signals for a (positive +
        negatives) training list."""
        assert self._item_graph is not None
        assert self._tail_index is not None
        assert self._path_tree is not None
        assert self._basket_index is not None
        history = tuple(owned_arr.tolist())
        # Pseudo-candidates: outcome_builder just needs one feature row per
        # item; the score/source fields don't matter here.
        fake_cands = [Candidate(item_id=i, score=0.0, source="outcome") for i in items]
        return _compute_signal_features(
            candidates=fake_cands,
            owned_items=owned_arr,
            query_basket=frozenset(history[-MAX_QUERY_BASKET_SIZE:]),
            history=history[-self.max_history_for_recommend :],
            item_graph=self._item_graph,
            tail_index=self._tail_index,
            path_tree=self._path_tree,
            basket_index=self._basket_index,
            basket_similarity=self.basket_similarity,
        )

    # ---- introspection (PRD §10.2 power-user surface) ---------------------

    @property
    def item_graph(self) -> ItemGraph:
        self._require_fitted()
        assert self._item_graph is not None
        return self._item_graph

    @property
    def tail_index(self) -> TailIndex:
        self._require_fitted()
        assert self._tail_index is not None
        return self._tail_index

    @property
    def path_tree(self) -> PathTree:
        self._require_fitted()
        assert self._path_tree is not None
        return self._path_tree

    @property
    def basket_index(self) -> BasketIndex:
        self._require_fitted()
        assert self._basket_index is not None
        return self._basket_index

    @property
    def session_inference(self) -> SessionInference:
        self._require_fitted()
        assert self._session_inference is not None
        return self._session_inference

    @property
    def schema(self) -> InteractionSchema:
        self._require_fitted()
        assert self._schema is not None
        return self._schema

    def data_density(self) -> dict[str, float | int]:
        self._require_fitted()
        assert self._interactions is not None
        assert self._item_graph is not None
        n_items = self._item_graph.n_items
        n_entities = self._interactions["entity_id"].nunique()
        n_interactions = len(self._interactions)
        max_edges = max(n_items * (n_items - 1), 1)
        return {
            "n_items": n_items,
            "n_entities": n_entities,
            "n_interactions": n_interactions,
            "graph_density": self._item_graph.n_edges / max_edges,
        }

    # ---- recommending -----------------------------------------------------

    def recommend(
        self,
        entity_id: object,
        n: int = 10,
        constraints: list[ConstraintPredicate] | None = None,
    ) -> list[Recommendation]:
        """Return up to ``n`` recommendations for the given entity."""
        self._require_fitted()
        assert self._cooc_retriever is not None
        assert self._path_retriever is not None
        assert self._tail_index is not None
        assert self._path_tree is not None
        assert self._basket_index is not None

        owned_items = self._owned_by_entity.get(entity_id, np.array([]))
        owned_set: set[object] = set(owned_items.tolist()) if owned_items.size else set()
        history = self._history_by_entity.get(entity_id, ())
        query_basket: frozenset[object] = frozenset(history[-MAX_QUERY_BASKET_SIZE:])

        # Stage 1: retrieve from both sources, union + dedup.
        raw_candidates = list(self._cooc_retriever.retrieve(owned_items, self.retrieval_budget))
        raw_candidates.extend(
            self._path_retriever.retrieve(
                recent_history=history, budget=self.retrieval_budget, exclude=owned_set
            )
        )
        candidates = _dedup_max_score(raw_candidates, self.retrieval_budget)

        if constraints:
            candidates = apply_constraints(candidates, constraints)

        if not candidates:
            return []

        # Stage 2: score each candidate on each signal.
        assert self._item_graph is not None
        features = _compute_signal_features(
            candidates=candidates,
            owned_items=owned_items,
            query_basket=query_basket,
            history=history[-self.max_history_for_recommend :],
            item_graph=self._item_graph,
            tail_index=self._tail_index,
            path_tree=self._path_tree,
            basket_index=self._basket_index,
            basket_similarity=self.basket_similarity,
        )
        # Stage 2: score via Bayesian posterior mean when available,
        # heuristic blend otherwise. Both operate on the same SignalFeatures.
        ci_lower: np.ndarray | None
        ci_upper: np.ndarray | None
        if self._bayesian_blend is not None and self.use_bayesian_blend:
            mean, lower, upper = self._bayesian_blend.score_with_uncertainty(
                features, coverage=self.credible_coverage
            )
            weights_for_explanation = {
                name: float(w)
                for name, w in zip(
                    self._bayesian_blend.signal_names,
                    self._bayesian_blend.posterior_mean,
                    strict=True,
                )
            }
            scores = mean
            ci_lower, ci_upper = lower, upper
        else:
            scores = self._heuristic_blend.score(features)
            ci_lower, ci_upper = None, None
            weights_for_explanation = dict(self._heuristic_blend.weights)

        order = np.argsort(-scores)
        top = order[:n]

        return [
            Recommendation(
                item_id=candidates[i].item_id,
                score=float(scores[i]),
                explanation=_build_explanation(
                    candidate=candidates[i],
                    blended_score=float(scores[i]),
                    signal_names=features.signal_names,
                    signal_row=features.matrix[i],
                    weights=weights_for_explanation,
                ),
                credible_interval=(
                    (float(ci_lower[i]), float(ci_upper[i]))
                    if ci_lower is not None and ci_upper is not None
                    else None
                ),
                credible_coverage=(self.credible_coverage if ci_lower is not None else None),
            )
            for i in top
        ]

    # ---- Phase 3 introspection --------------------------------------------

    def posterior_summary(self) -> dict[str, object]:
        """Posterior statistics and diagnostics (PRD §6.7).

        Returns per-signal posterior mean, credible interval, prior alpha,
        and the convergence diagnostic report. Use this in production
        monitoring dashboards to catch VI failures early.
        """
        self._require_fitted()
        if self._bayesian_blend is None:
            return {"bayesian_blend_active": False}
        blend = self._bayesian_blend
        ci = blend.credible_interval(coverage=self.credible_coverage)
        summary = {
            "bayesian_blend_active": True,
            "signal_names": list(blend.signal_names),
            "posterior_mean": blend.posterior_mean.tolist(),
            "posterior_variance": blend.posterior_variance.tolist(),
            "credible_interval": ci.tolist(),
            "credible_coverage": self.credible_coverage,
            "prior_alpha": blend.prior_alpha.tolist(),
            "elbo_trace_length": len(blend.elbo_trace),
            "likelihood": self.likelihood.name,
        }
        if self._diagnostics is not None:
            summary["diagnostics"] = {
                "elbo_monotonic": self._diagnostics.elbo_monotonic,
                "elbo_final": self._diagnostics.elbo_final,
                "elbo_peak": self._diagnostics.elbo_peak,
                "ppc_deviation": self._diagnostics.ppc_deviation,
                "ppc_passes": self._diagnostics.ppc_passes,
                "ess_ratio": self._diagnostics.ess_ratio,
                "ess_passes": self._diagnostics.ess_passes,
                "all_pass": self._diagnostics.all_pass,
                "warnings": self._diagnostics.warnings(),
            }
        return summary

    # ---- internals --------------------------------------------------------

    def _require_fitted(self) -> None:
        if self._interactions is None:
            raise EngineNotFittedError("Engine.fit must be called before use")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _approx_clustering_coefficient(item_graph: ItemGraph) -> float:
    """Cheap clustering-coefficient proxy for prior construction.

    The classical local clustering coefficient requires enumerating
    triangles per node - O(sum(deg^2)) which is too expensive at fit time
    on moderate graphs. For prior construction we only need a scalar in
    [0, 1] that correlates with "does this graph have meaningful
    communities." We use the normalized density of the second-hop
    adjacency - the sparsity of ``A @ A`` vs. the full matrix.
    """
    adj = item_graph.adjacency
    n = item_graph.n_items
    if n < 3:
        return 0.0
    # Sample-based estimate: pick up to 200 items, count triangles per node.
    sample_size = min(200, n)
    rng = np.random.default_rng(seed=0)
    idxs = rng.choice(n, size=sample_size, replace=False)
    ratios: list[float] = []
    for i in idxs:
        row = adj.getrow(i).toarray().ravel()
        neighbors = np.where(row > 0)[0]
        deg = len(neighbors)
        if deg < 2:
            continue
        # Triangles at node i = number of neighbor pairs that are themselves
        # connected. Upper bound deg * (deg - 1) / 2.
        sub = adj[neighbors][:, neighbors]
        triangles = float(sub.sum()) / 2.0
        possible = deg * (deg - 1) / 2.0
        ratios.append(triangles / possible if possible > 0 else 0.0)
    return float(np.mean(ratios)) if ratios else 0.0


def _reference_timestamp_from(interactions: pd.DataFrame) -> float | None:
    if "timestamp" not in interactions.columns or len(interactions) == 0:
        return None
    return float(interactions["timestamp"].max().timestamp())


def _build_histories(
    interactions: pd.DataFrame, schema: InteractionSchema
) -> dict[object, tuple[object, ...]]:
    """Per-entity ordered item history (oldest -> newest)."""
    if schema.has_timestamp:
        sorted_df = interactions.sort_values(["entity_id", "timestamp"], kind="mergesort")
    else:
        sorted_df = interactions
    out: dict[object, tuple[object, ...]] = {}
    for entity, group in sorted_df.groupby("entity_id", sort=False):
        out[entity] = tuple(group["item_id"].tolist())
    return out


def _dedup_max_score(candidates: list[Candidate], budget: int) -> list[Candidate]:
    """Merge candidates across retrievers, keeping the max score per item.

    Preserves the source of the winning candidate for provenance in
    explanations.
    """
    if not candidates:
        return []
    best: dict[object, Candidate] = {}
    for c in candidates:
        existing = best.get(c.item_id)
        if existing is None or c.score > existing.score:
            best[c.item_id] = c
    deduped = sorted(best.values(), key=lambda c: -c.score)
    return deduped[:budget]


def _compute_signal_features(
    candidates: list[Candidate],
    owned_items: np.ndarray,
    query_basket: frozenset[object],
    history: tuple[object, ...],
    item_graph: ItemGraph,
    tail_index: TailIndex,
    path_tree: PathTree,
    basket_index: BasketIndex,
    basket_similarity: BasketSimilarity,
) -> SignalFeatures:
    """Compute the (N_candidates, K_signals) feature matrix in SIGNAL_ORDER."""
    n = len(candidates)
    matrix = np.zeros((n, len(SIGNAL_ORDER)), dtype=np.float64)
    cand_ids = [c.item_id for c in candidates]

    last_item = history[-1] if history else None
    # path_full, path_tail, path_basket
    matrix[:, 0] = path_tree.score_many(cand_ids, history)
    matrix[:, 1] = tail_index.score_many(cand_ids, last_item)
    matrix[:, 2] = basket_index.score_many(
        cand_ids, query_basket=query_basket, similarity=basket_similarity
    )
    # cooccurrence - recompute from the graph against the entity's owned set.
    # Using the retriever's max-score would let the path_endpoint retriever's
    # score leak into this feature for candidates it won via dedup.
    matrix[:, 3] = _cooccurrence_signal(cand_ids, owned_items, item_graph)

    return SignalFeatures(matrix=matrix, signal_names=SIGNAL_ORDER)


def _cooccurrence_signal(
    cand_ids: list[object],
    owned_items: np.ndarray,
    item_graph: ItemGraph,
) -> np.ndarray:
    """Sum of item-graph edges between each candidate and the owned set."""
    if item_graph.n_items == 0 or owned_items.size == 0:
        return np.zeros(len(cand_ids), dtype=np.float64)
    owned_indices = [item_graph.item_index[i] for i in owned_items if i in item_graph.item_index]
    if not owned_indices:
        return np.zeros(len(cand_ids), dtype=np.float64)
    summed = np.asarray(item_graph.adjacency[owned_indices].sum(axis=0)).ravel()
    out = np.zeros(len(cand_ids), dtype=np.float64)
    for i, cid in enumerate(cand_ids):
        idx = item_graph.item_index.get(cid)
        if idx is not None:
            out[i] = float(summed[idx])
    return out


def _build_explanation(
    candidate: Candidate,
    blended_score: float,
    signal_names: tuple[str, ...],
    signal_row: np.ndarray,
    weights: dict[str, float],
) -> Explanation:
    """Build an Explanation from the dominant signal and the debug payload."""
    # Contribution = weight * (raw signal score), for the *user-facing*
    # narrative. This is an approximation because the blend actually operates
    # on rescaled decorrelated signals, but it's accurate enough for the
    # primary-sentence template and matches what practitioners expect to see.
    contributions: dict[str, float] = {}
    for i, name in enumerate(signal_names):
        contributions[name] = float(signal_row[i]) * weights.get(name, 0.0)

    dominant = (
        max(contributions, key=lambda k: contributions[k]) if contributions else candidate.source
    )
    primary = _PRIMARY_TEMPLATES.get(dominant, "Recommended based on your history.")
    debug: dict[str, Any] = {
        "signals": {
            name: {"raw": float(signal_row[i]), "weight": weights.get(name, 0.0)}
            for i, name in enumerate(signal_names)
        },
        "blended_score": blended_score,
        "dominant_signal": dominant,
    }
    return Explanation(primary=primary, debug_payload=debug)


_PRIMARY_TEMPLATES: dict[str, str] = {
    "cooccurrence": "Often seen with items you've already interacted with.",
    "path_tail": "Frequently follows what you just interacted with.",
    "path_full": "Matches a longer pattern of items you recently interacted with.",
    "path_basket": "Commonly added next by others with a similar collection.",
}


def _fit_path_family_decorrelation(
    interactions: pd.DataFrame,
    item_graph: ItemGraph,
    tail_index: TailIndex,
    path_tree: PathTree,
    basket_index: BasketIndex,
    basket_similarity: BasketSimilarity,
    history_by_entity: dict[object, tuple[object, ...]],
    owned_by_entity: dict[object, np.ndarray],
) -> DecorrelationBasis | None:
    """Fit the Gram-Schmidt basis over the path family on the chronological
    tail 10% of training.

    Simulates recommendation on the tail: for each held-out event, treat the
    entity's prior interactions as history and score a small candidate pool
    under each path signal. The path-family signal matrix feeds
    ``fit_decorrelation``. Cooccurrence is NOT part of this basis (PRD §6.2
    puts it in a separate block).
    """
    if "timestamp" in interactions.columns:
        sorted_df = interactions.sort_values("timestamp", kind="mergesort")
    else:
        sorted_df = interactions
    n = len(sorted_df)
    if n < 50:
        return None
    cutoff = int(n * 0.9)
    held_out = sorted_df.iloc[cutoff:]

    # Sample up to 200 held-out events to keep fit time bounded.
    sample = held_out.iloc[:: max(1, len(held_out) // 200)].head(200)

    candidate_pool: list[object] = list(item_graph.item_ids[:500])
    if not candidate_pool:
        return None

    rows: list[np.ndarray] = []
    for _, event in sample.iterrows():
        entity = event["entity_id"]
        owned_arr = owned_by_entity.get(entity, np.array([]))
        if owned_arr.size == 0:
            continue
        history = history_by_entity.get(entity, ())
        owned_set = set(owned_arr.tolist())
        cand_ids = [c for c in candidate_pool if c not in owned_set][:100]
        if not cand_ids:
            continue
        fake_candidates = [
            Candidate(item_id=c, score=0.0, source="decorrelation_fit") for c in cand_ids
        ]
        feats = _compute_signal_features(
            candidates=fake_candidates,
            owned_items=owned_arr,
            query_basket=frozenset(history[-MAX_QUERY_BASKET_SIZE:]),
            history=history,
            item_graph=item_graph,
            tail_index=tail_index,
            path_tree=path_tree,
            basket_index=basket_index,
            basket_similarity=basket_similarity,
        )
        # Extract just the path family columns for the basis fit.
        path_indices = [feats.signal_names.index(n) for n in PATH_FAMILY]
        rows.append(feats.matrix[:, path_indices])

    if not rows:
        return None
    signal_matrix = np.vstack(rows)
    return fit_decorrelation(signal_matrix, signal_names=list(PATH_FAMILY))
