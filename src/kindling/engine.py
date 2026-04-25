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

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from kindling._native import NATIVE_AVAILABLE, kindling_native
from kindling.blend.bayesian import BayesianBlend
from kindling.blend.normalize import NormalizeMode, normalize_columns
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
from kindling.graph.als_factors import ALSFactors, build_als_factors
from kindling.graph.temporal_interaction import (
    TemporalInteractionGraph,
    build_temporal_interaction_graph,
)
from kindling.gate import GatingConfig, GatingNetwork, fit_gating_network
from kindling.gate.features import compute_context_features as _compute_gate_context
from kindling.graph.lightgcn import LightGCNConfig, LightGCNModel, build_lightgcn
from kindling.graph.cost_graph import CostGraph, build_cost_graph
from kindling.graph.item_cosine import ItemCosineMatrix, build_item_cosine_matrix
from kindling.graph.item_graph import ItemGraph, build_item_graph
from kindling.graph.session_cooccurrence import (
    SessionCooccurrenceGraph,
    build_session_cooccurrence_graph,
)
from kindling.blend.layered import LayeredConfig, layered_score
from kindling.ingest.contract import (
    InteractionSchema,
    canonicalize,
    validate_interactions,
)
from kindling.ingest.sessions import SessionInference, infer_sessions
from kindling.lifecycle.decay import DecayProtocol, ExponentialDecay
from kindling.lifecycle.drift import DriftReport, DriftTracker
from kindling.lifecycle.pruning import PreservedAggregate, PruningConfig
from kindling.outcomes.log import OutcomeLog
from kindling.outcomes.replay import replay_to_batch
from kindling.path._sessions import sessions_from_interactions
from kindling.path.basket_index import BasketIndex, BasketSimilarity, build_basket_index
from kindling.path.path_tree import PathTree, build_path_tree
from kindling.path.tail_index import TailIndex, build_tail_index
from kindling.rerank.calibration import (
    CategoryIndex,
    apply_calibration,
    build_category_index,
)
from kindling.rerank.constraints import ConstraintPredicate, apply_constraints
from kindling.rerank.dpp import (
    CooccurrenceCosineKernel,
    DPPGreedy,
    SimilarityKernel,
)
from kindling.rerank.lift import (
    PopulationBaselines,
    apply_lift,
    compute_population_baselines,
)
from kindling.rerank.temperature import (
    TemperatureInput,
    TemperatureObjective,
    resolve_temperature,
)
from kindling.rerank.temperature import solve as solve_temperature
from kindling.personas.config import PersonaConfig
from kindling.personas.index import PersonaIndex
from kindling.preprocess import InteractionContext, preprocess_interactions
from kindling.rank.lightgbm_ranker import LightGBMRanker
from kindling.repeat import (
    RepeatConfig,
    RepeatProfileTable,
    fit_repeat_profiles,
    multiplier as repeat_multiplier,
)
from kindling.retrieve.cooccurrence import CoOccurrenceRetriever
from kindling.retrieve.path_endpoint import PathEndpointRetriever
from kindling.retrieve.policy import RetrieverEntry, build_retriever_stack
from kindling.retrieve.protocol import Candidate

DEFAULT_RETRIEVAL_BUDGET = 500
# Fixed path-family-first signal order for decorrelation (PRD §6.2, §6.1).
# Signals after ``cooccurrence`` are the "other" block; the Phase 5 cost
# signals (PRD §3.6) are three negative-oriented entries. The feature
# matrix stores -effective_cost so that positive Dirichlet weights
# translate into penalties.
SIGNAL_ORDER: tuple[str, ...] = (
    "path_full",
    "path_tail",
    "path_basket",
    "cooccurrence",
    "cost_population",
    "cost_entity",
    "cost_context",
    "item_item_cosine",
    "als_factor",
    "persona",
    "lightgcn",
    "temporal_cooccurrence",
    "session_cooccurrence",
)
# Deactivated signals: the column remains in SIGNAL_ORDER for back-compat
# but _compute_signal_features leaves it at zero and the Bayesian posterior
# learns weight -> 0. Avoids a cascading rename through priors.toml,
# persistence, benchmarks, and tests.
#
#   path_full: consistently the weakest signal across datasets
#   (grocery 0.047, ml1m 0.025, yelp 0.000, amazon-beauty 0.001).
#   The matrix harness still computes it on demand when a user
#   explicitly requests per-signal NDCG; the live blend skips it.
DEACTIVATED_SIGNALS: frozenset[str] = frozenset({"path_full"})
NEGATIVE_SIGNAL_MODES = frozenset({"positive_only", "explicit", "implicit_from_impressions"})
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
        basket_scan_cap: int | None = 10_000,
        # Default 0 = never skip. Set to 0.05 to trade ~7% NDCG for a ~100x
        # latency win on ratings-style data (measured on ML-1M full):
        # p95 152ms -> 1.6ms, NDCG 0.213 -> 0.198, MRR 0.364 -> 0.331.
        # Signals with posterior weight below the threshold contribute
        # negligibly to the blend score; skipping their computation
        # entirely saves the dominant basket_index.score_many call.
        skip_signal_weight_threshold: float = 0.0,
        use_bayesian_blend: bool = True,
        likelihood: LikelihoodProtocol | None = None,
        credible_coverage: float = 0.9,
        seed: int = 0,
        vi_max_iter: int = 300,
        # Phase 4 re-rank configuration.
        item_metadata: pd.DataFrame | None = None,
        category_column: str = "category",
        diversity_kernel: SimilarityKernel | None = None,
        # Phase 5 negative-signal configuration.
        negative_signal_mode: str | None = None,
        alpha_pop: float = 0.3,
        outcome_log_path: str | None = None,
        # Phase 6 lifecycle configuration.
        pruning_config: PruningConfig | None = None,
        # Warm-regime learned ranker (LightGBM LambdaRank). Off by default
        # because the current training distribution (random-sampled
        # negatives) doesn't match the inference distribution (retrieved
        # candidates) - enabling produces worse NDCG than the Bayesian
        # blend. See ADR-lightgbm-warm-regime.md. Opt in once the training
        # generator is upgraded to retrieved-candidate negatives.
        use_ranker: bool = False,
        ranker_negatives_per_positive: int = 99,
        ranker_min_train_pairs: int = 500,
        # Persona signal (PRD supplement). Off by default - opt in via
        # Engine(persona_config=PersonaConfig(enabled=True, ...)).
        persona_config: PersonaConfig | None = None,
        # Repeat-consumption module. Off by default - opt in when your
        # dataset has legitimate repeat interactions (grocery, media,
        # replenishment). ML-1M-style no-repeat datasets don't need this.
        repeat_config: RepeatConfig | None = None,
        # Rating handling. None = auto-detect from the rating column;
        # True = force rating-weighted positive signals; False = force
        # binary implicit feedback (ignore rating column).
        use_ratings: bool | None = None,
        # LightGCN signal (pure-numpy, two-stage: BPR-train base
        # embeddings + inference-time graph propagation). Off by default.
        lightgcn_config: LightGCNConfig | None = None,
        # Signal-column normalization applied between feature computation
        # and blend scoring. Default "none" preserves the historical
        # architecture where cooc's raw-magnitude dominance compensated
        # for its under-weighted prior. "zscore" puts every column on
        # the same scale - correct for the gating network and RRF-style
        # aggregation, but produces regressions when paired directly
        # with the Bayesian blend's current priors (the priors were
        # calibrated against raw magnitudes). See ADR-score-normalization.
        signal_normalization: NormalizeMode = "none",
        # Gating network (pure-numpy MLP) producing per-entity softmax
        # weights over the K signals. When enabled, its output replaces
        # the Bayesian blend's posterior mean for scoring, and the
        # engine forces signal_normalization="zscore" internally so
        # the gate trains + infers on the same scale.
        gating_config: GatingConfig | None = None,
        # Cooc + adaptive boosting (the "layered" scoring architecture).
        # When enabled, replaces the Bayesian blend at recommend time
        # with: cooc primary + cumulative one-tailed z-gated boost
        # layers (path_basket / session_cooccurrence /
        # temporal_cooccurrence). LayeredAutoCalibrator runs at fit
        # time to pick (z_threshold, boost_multiplier) per dataset.
        # Disable Bayesian blend independently if you don't need
        # credible intervals - layered doesn't produce them.
        layered_scoring: bool = False,
        layered_config: "LayeredConfig | None" = None,
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
        self.basket_scan_cap = basket_scan_cap
        self.skip_signal_weight_threshold = skip_signal_weight_threshold

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
        self._session_cooc_graph: SessionCooccurrenceGraph | None = None
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

        # Phase 4 re-rank state.
        self.item_metadata = item_metadata
        self.category_column = category_column
        self._diversity_kernel_override = diversity_kernel
        self._diversity_kernel: SimilarityKernel | None = None
        self._population_baselines: PopulationBaselines | None = None
        # Popularity ranking cached at fit time for O(N) cold-start fallback.
        self._popular_items_ranked: list[object] = []
        self._category_index: CategoryIndex | None = None
        # Item-item cosine matrix (8th signal).
        self._item_cosine: "ItemCosineMatrix | None" = None
        # ALS latent factors (9th signal, optional - requires `implicit`).
        self._als_factors: "ALSFactors | None" = None
        # Warm-regime LambdaRank scorer over the 9-signal feature matrix.
        self.use_ranker = use_ranker
        self.ranker_negatives_per_positive = ranker_negatives_per_positive
        self.ranker_min_train_pairs = ranker_min_train_pairs
        self._ranker: "LightGBMRanker | None" = None
        self._ranker_retrieval_hit_rate: float = 0.0

        # Persona signal state (10th column in SignalFeatures).
        self.persona_config = persona_config
        self._persona_index: PersonaIndex | None = None

        # Repeat-consumption module state.
        self.repeat_config = repeat_config
        self._repeat_table: RepeatProfileTable | None = None

        # Rating handling + preprocess-time context.
        self.use_ratings = use_ratings
        self._interaction_context: InteractionContext | None = None

        # LightGCN signal state (11th column in SignalFeatures).
        self.lightgcn_config = lightgcn_config
        self._lightgcn: LightGCNModel | None = None

        # Temporal interaction graph (substrate for temporal_cooccurrence
        # 12th signal; auto-falls-back to pure-count when session-presence
        # guard detects rating-burst timestamps).
        self._temporal_graph: TemporalInteractionGraph | None = None

        # Per-query signal-column normalization mode.
        self.signal_normalization: NormalizeMode = signal_normalization

        # Gating network state.
        self.gating_config = gating_config
        self._gate: GatingNetwork | None = None
        self._gate_context_cache: dict[object, np.ndarray] = {}

        # Layered (cooc + adaptive boosting) scoring state.
        self.layered_scoring = layered_scoring
        self.layered_config: LayeredConfig | None = layered_config
        # Populated at fit time when layered_scoring=True. The
        # auto-calibrator picks (z_threshold, boost_multiplier) per
        # dataset shape. Can also be passed explicitly to skip
        # auto-calibration.
        self._layered_calibration: object | None = None

        # Per-subsystem fit timings (seconds). Populated during fit()
        # for benchmarks + diagnostics. Keys: item_graph, item_cosine,
        # als, path_tree, tail_index, basket_index, persona, lightgcn,
        # bayesian_blend, ranker, gate.
        self._fit_timings: dict[str, float] = {}
        if gating_config is not None and gating_config.enabled:
            # Gate requires same-scale inputs at train + infer time.
            self.signal_normalization = "zscore"
        # Per-(entity, item) most-recent interaction timestamp. Used to
        # compute time_since_last at recommend time. Populated during
        # fit when the repeat module is enabled.
        self._last_interaction_ts: dict[tuple[object, object], float] = {}

        # Data-adaptive retriever stack (ADR-retriever-union). Built at
        # fit time from engine fitted state + DataFeatures. Populated on
        # first recommend() after load via _ensure_retriever_stack().
        self._retriever_stack: list[RetrieverEntry] = []
        # Cached for post-load stack rebuild (the only DataFeatures
        # field the stack gates on).
        self._has_explicit_sessions: bool = False

        # Phase 5 negative-signal + outcome-log state.
        if negative_signal_mode is not None and negative_signal_mode not in NEGATIVE_SIGNAL_MODES:
            raise ValueError(
                f"negative_signal_mode must be one of {sorted(NEGATIVE_SIGNAL_MODES)} "
                f"or None (auto-detect), got {negative_signal_mode!r}"
            )
        self._user_negative_mode = negative_signal_mode
        self.negative_signal_mode: str = "positive_only"  # resolved at fit time
        self.alpha_pop = alpha_pop
        self._cost_graph: CostGraph | None = None
        self.outcome_log = OutcomeLog(path=outcome_log_path or ":memory:")

        # Phase 6 lifecycle state.
        self.pruning_config: PruningConfig = (
            pruning_config if pruning_config is not None else PruningConfig()
        )
        self._preserved_aggregates: list[PreservedAggregate] = []
        self._drift_tracker = DriftTracker()

    # ---- fitting ----------------------------------------------------------

    def fit(self, interactions: pd.DataFrame) -> Engine:
        """Validate, canonicalize, and build derived structures."""
        schema = validate_interactions(interactions)
        self._schema = schema
        canonical = canonicalize(interactions, schema)
        # Preprocess: attach the _interaction_weight column every positive
        # signal reads. Auto-detects ratings; respects self.use_ratings
        # override. Must happen before any signal is built.
        canonical, context = preprocess_interactions(
            canonical, use_ratings=self.use_ratings
        )
        self._interactions = canonical
        self._interaction_context = context
        self._reference_timestamp = _reference_timestamp_from(self._interactions)

        self._session_inference = infer_sessions(self._interactions)
        sessions = list(
            sessions_from_interactions(
                self._interactions,
                self._session_inference.session_ids,
            )
        )

        import time as _time

        _t0 = _time.perf_counter()
        self._item_graph = build_item_graph(self._interactions)
        self._fit_timings["item_graph"] = _time.perf_counter() - _t0

        # Session-cooccurrence: items co-occurring in the same session,
        # row dimension = session_id (vs item_graph's entity_id rows).
        # Auto-skips datasets without deep enough sessions; the
        # _compute_signal_features helper sees None and zero-fills the
        # column.
        _t0 = _time.perf_counter()
        self._session_cooc_graph = build_session_cooccurrence_graph(
            interactions=self._interactions,
            item_index=self._item_graph.item_index,
            session_ids=self._session_inference.session_ids,
            session_strategy=self._session_inference.strategy,
            session_gap_seconds=self._session_inference.gap_threshold_seconds,
        )
        self._fit_timings["session_cooc_graph"] = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        self._tail_index = build_tail_index(
            sessions, decay=self.decay, reference_timestamp=self._reference_timestamp
        )
        self._fit_timings["tail_index"] = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        self._path_tree = build_path_tree(
            sessions,
            max_prefix=self.max_path_prefix,
            decay=self.decay,
            reference_timestamp=self._reference_timestamp,
        )
        self._fit_timings["path_tree"] = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        self._basket_index = build_basket_index(
            sessions, decay=self.decay, reference_timestamp=self._reference_timestamp
        )
        self._fit_timings["basket_index"] = _time.perf_counter() - _t0

        # Kept for backward compatibility with the ranker-training
        # code path and the outcome-builder helper. The Engine's
        # primary retrieval path uses self._retriever_stack (built below)
        # instead - see ADR-retriever-union.md.
        self._cooc_retriever = CoOccurrenceRetriever(self._item_graph)
        self._path_retriever = PathEndpointRetriever(self._path_tree, self._tail_index)

        self._owned_by_entity = {
            entity: group["item_id"].to_numpy()
            for entity, group in self._interactions.groupby("entity_id", sort=False)
        }
        self._history_by_entity = _build_histories(self._interactions, schema)

        # Item-item cosine matrix (8th signal). Built once at fit; scored
        # by summing cos over owned items. Needs per-item user counts, which
        # are the diagonal of U^T U that ItemGraph does not store - derive
        # from the population-baseline fraction * n_entities.
        n_entities_val = int(self._interactions["entity_id"].nunique())
        pop_for_counts = compute_population_baselines(self._interactions)
        ordered_counts = np.array(
            [
                pop_for_counts.item_to_baseline.get(item, 0.0) * n_entities_val
                for item in self._item_graph.item_ids
            ],
            dtype=np.float64,
        )
        _t0 = _time.perf_counter()
        self._item_cosine = build_item_cosine_matrix(
            cooccurrence=self._item_graph.adjacency,
            item_counts=ordered_counts,
            top_k=200,
        )
        self._fit_timings["item_cosine"] = _time.perf_counter() - _t0

        # ALS latent factors (9th signal). Optional: graceful no-op if
        # `implicit` isn't installed. Small factor count (32) keeps fit
        # time sub-second on ML-1M scale.
        _t0 = _time.perf_counter()
        self._als_factors = build_als_factors(
            interactions=self._interactions,
            item_graph_item_index=self._item_graph.item_index,
            factors=32,
            iterations=8,
            random_state=self.seed,
        )
        self._fit_timings["als"] = _time.perf_counter() - _t0

        # LightGCN: pure-numpy BPR-trained base embeddings + K-layer graph
        # propagation at inference. Off by default (lightgcn_config=None);
        # opt in via Engine(lightgcn_config=LightGCNConfig(...)).
        if self.lightgcn_config is not None:
            _t0 = _time.perf_counter()
            self._lightgcn = build_lightgcn(
                interactions=self._interactions,
                item_graph_item_index=self._item_graph.item_index,
                config=self.lightgcn_config,
            )
            self._fit_timings["lightgcn"] = _time.perf_counter() - _t0

        # Temporal interaction graph: substrate for the temporal_cooccurrence
        # signal. Auto-calibrates the logistic-decay kernel via GMM on
        # inter-event log-deltas; falls back to pure-count when timestamps
        # are missing or session-presence guard detects rating-burst data.
        _t0 = _time.perf_counter()
        self._temporal_graph = build_temporal_interaction_graph(
            interactions=self._interactions,
            item_index=self._item_graph.item_index,
        )
        self._fit_timings["temporal_graph"] = _time.perf_counter() - _t0

        # Phase 4 re-rank state.
        self._population_baselines = compute_population_baselines(self._interactions)
        # Cold-start fallback ranking (plan ADR-growth-curves.md §1):
        # items sorted by popularity descending. Used when retrieval returns
        # empty candidates for an entity (unseen or new).
        if self._population_baselines is not None and self._population_baselines.item_to_baseline:
            self._popular_items_ranked = sorted(
                self._population_baselines.item_to_baseline,
                key=lambda i: self._population_baselines.item_to_baseline[i],  # type: ignore[union-attr]
                reverse=True,
            )
        else:
            self._popular_items_ranked = []
        if self._diversity_kernel_override is not None:
            self._diversity_kernel = self._diversity_kernel_override
        else:
            self._diversity_kernel = CooccurrenceCosineKernel(item_graph=self._item_graph)
        if self.item_metadata is not None:
            self._category_index = build_category_index(
                interactions=self._interactions,
                item_metadata=self.item_metadata,
                category_column=self.category_column,
            )

        # Phase 5 negative-signal state.
        if self._user_negative_mode is None:
            self.negative_signal_mode = "explicit" if schema.has_action_type else "positive_only"
        else:
            self.negative_signal_mode = self._user_negative_mode
        if self.negative_signal_mode == "positive_only":
            self._cost_graph = CostGraph(alpha_pop=self.alpha_pop)
        else:
            self._cost_graph = build_cost_graph(
                interactions=self._interactions, alpha_pop=self.alpha_pop
            )

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

        # Phase 6: prune at retrain-time per PRD §3.5. Ordering per the
        # plan: decay is already baked into the stored weights, so prune
        # happens before the posterior refit so the Bayesian blend sees
        # the pruned structures.
        if self.pruning_config.enabled and self.pruning_config.schedule == "retrain":
            self.prune()

        # Persona signal (PRD supplement). Must precede blend + ranker
        # fits because it produces the 10th column of SignalFeatures
        # that both downstream consumers read.
        if self.persona_config is not None and self.persona_config.enabled:
            _t0 = _time.perf_counter()
            self._fit_persona_index()
            self._fit_timings["persona"] = _time.perf_counter() - _t0

        # Repeat-consumption module. Independent of blend/ranker fits;
        # lives as a post-scoring multiplier, so ordering doesn't matter.
        if self.repeat_config is not None and self.repeat_config.enabled:
            _t0 = _time.perf_counter()
            self._fit_repeat_module()
            self._fit_timings["repeat"] = _time.perf_counter() - _t0

        # Bayesian blend: prior from data features + VI fit on the
        # chronological tail.
        if self.use_bayesian_blend:
            _t0 = _time.perf_counter()
            self._fit_bayesian_blend(path_basis, sessions_count=len(sessions))
            self._fit_timings["bayesian_blend"] = _time.perf_counter() - _t0

        # Build the data-adaptive retriever stack now that all fitted
        # state is available (item_graph, path indexes, als_factors,
        # item_cosine, persona_index). DataFeatures drives the gating
        # (session-specific retrievers only when sessions are explicit).
        features = self._compute_data_features(len(sessions))
        self._has_explicit_sessions = features.has_explicit_sessions
        self._retriever_stack = build_retriever_stack(
            engine=self, features=features, retrieval_budget=self.retrieval_budget
        )

        # Warm-regime LambdaRank over the 9-signal feature matrix. Trained
        # via last-item holdout + negative sampling. Silent no-op when
        # `lightgbm` isn't installed or training data is too small.
        if self.use_ranker:
            _t0 = _time.perf_counter()
            self._fit_ranker()
            self._fit_timings["ranker"] = _time.perf_counter() - _t0

        # Gating network. Trained BPR-style on the fitted signal stack;
        # replaces Bayesian-blend posterior mean for scoring when fitted.
        if self.gating_config is not None and self.gating_config.enabled:
            _t0 = _time.perf_counter()
            self._gate = fit_gating_network(self, self.gating_config)
            if self._gate is not None:
                self._gate_context_cache = _compute_gate_context(self)
            self._fit_timings["gate"] = _time.perf_counter() - _t0

        # Layered (cooc + adaptive boosting) auto-calibration. Sweeps
        # (z_threshold, boost_multiplier) on a leave-out slice of
        # training data and picks the cell with highest held-out
        # NDCG@10. When the user explicitly passed a layered_config,
        # use it verbatim and skip the sweep.
        if self.layered_scoring:
            _t0 = _time.perf_counter()
            if self.layered_config is None:
                from kindling.blend.layered_calibrator import calibrate

                cal = calibrate(self)
                self.layered_config = cal.best_config
                self._layered_calibration = cal
            self._fit_timings["layered_calibration"] = _time.perf_counter() - _t0

        return self

    def _fit_persona_index(self) -> None:
        """Cluster users into personas and build the persona index.

        Skipped when user count is below ``min_activation_users``, when
        clustering produces zero personas, or when the persona config
        is disabled. The index becomes the 10th column in
        SignalFeatures via ``_compute_signal_features``.
        """
        assert self._interactions is not None
        assert self._item_graph is not None
        assert self.persona_config is not None

        from kindling.personas.build import build_persona_index, build_user_vectors

        cfg = self.persona_config
        entity_order = list(self._owned_by_entity.keys())
        n_users = len(entity_order)
        if n_users < cfg.min_activation_users:
            self._persona_index = None
            return

        item_ids = np.asarray(self._item_graph.item_ids, dtype=object)
        user_mat = build_user_vectors(
            self._interactions, item_ids=item_ids, entity_order=entity_order
        )

        clustering = cfg.resolved_clustering()
        # Dense input for clustering. Most real catalogs require
        # dimensionality reduction before clustering; HDBSCANClustering
        # handles it internally. K-means expects something already-low-
        # dim; if the caller passed an ALS reduction upstream, they
        # should've selected it here.
        try:
            cluster_result = clustering.fit(user_mat.toarray())
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Persona clustering failed ({exc!r}); signal disabled for this fit.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._persona_index = None
            return

        if cluster_result.n_personas == 0:
            self._persona_index = None
            return

        noise_fraction = float((cluster_result.assignments < 0).sum() / max(n_users, 1))
        if noise_fraction > 0.5:
            import warnings

            warnings.warn(
                f"Persona clustering produced {noise_fraction:.1%} noise points; "
                "the signal will contribute little. Consider tuning "
                "PersonaConfig.clustering or adjusting min_cluster_size_pct.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._persona_index = build_persona_index(
            interactions=self._interactions,
            cluster_result=cluster_result,
            item_ids=item_ids,
            entity_order=entity_order,
            z_threshold=cfg.z_threshold,
        )

        # Cold-start: compute per-(item, persona) overperformance. Items
        # that under-weight in the main persona_vectors (because the
        # z-score filter dropped them or they appear in few personas)
        # still get a signal when their early-interaction users
        # concentrate in a specific persona.
        if cfg.cold_start_weight > 0.0:
            from kindling.personas.cold_start import compute_cold_start_weights

            self._persona_index.cold_start_weights = compute_cold_start_weights(
                interactions=self._interactions,
                index=self._persona_index,
                overperformance_threshold=cfg.cold_start_overperformance_threshold,
            )
            self._persona_index.cold_start_weight = cfg.cold_start_weight

    def _fit_repeat_module(self) -> None:
        """Build the repeat-profile table + per-(entity, item) last-seen
        timestamps. Silent no-op when the input lacks a timestamp column.
        """
        assert self._interactions is not None
        assert self.repeat_config is not None
        if "timestamp" not in self._interactions.columns:
            self._repeat_table = RepeatProfileTable()
            return

        self._repeat_table = fit_repeat_profiles(
            interactions=self._interactions,
            item_graph=self._item_graph,
            config=self.repeat_config,
        )

        # Build last-seen timestamp per (entity, item). Used at recommend
        # time to compute time_since_last.
        df = self._interactions[["entity_id", "item_id", "timestamp"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
        df["ts_s"] = df["timestamp"].astype("int64") // 1_000_000_000
        # groupby last - since df is sorted by timestamp after fit ingestion
        # (sort happens in fit_repeat_profiles via stable mergesort), we
        # just take the max timestamp per pair.
        last_ts = df.groupby(["entity_id", "item_id"])["ts_s"].max()
        self._last_interaction_ts = {
            (ent, item): float(ts) for (ent, item), ts in last_ts.items()
        }

    def _fit_ranker(self) -> None:
        """Train a LightGBMRanker over the 9-signal feature matrix using
        last-item holdout + **retrieved-candidate negatives**.

        Per entity with >=2 interactions:
        - positive = the most recent item (held out).
        - negatives = the top-K candidates the retriever produces from the
          entity's prior history (same distribution as inference).

        Training distribution == inference distribution. This fixes the
        earlier random-negative sampler, which let LambdaRank learn the
        trivially-discriminating "cooc > 0" rule useless at inference time.
        See bench/reports/ADR-lightgbm-warm-regime.md for the measurement
        that motivated this change.

        Leakage caveat: signals are fit on the FULL interaction set
        including the positive, so the positive's features are slightly
        inflated. A leakage-free variant would refit signals per held-out
        entity, which is prohibitive. Accepted v1 limitation.
        """
        try:
            from kindling.rank.lightgbm_ranker import _require_lightgbm

            _require_lightgbm()
        except ImportError:
            # lightgbm not installed - silently fall through to blend.
            self._ranker = None
            return

        assert self._interactions is not None
        assert self._item_graph is not None
        assert self._cooc_retriever is not None
        assert self._path_retriever is not None

        n_neg = max(1, self.ranker_negatives_per_positive)
        group_size = n_neg + 1  # 1 positive + n_neg retrieved negatives
        rng = np.random.default_rng(self.seed + 7)

        features_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        groups: list[int] = []

        # Iterate entities with enough history. Cap at 2000 entities for a
        # bounded training-time budget; plenty for LambdaRank to generalize.
        eligible_entities = [
            e for e, hist in self._history_by_entity.items() if len(hist) >= 2
        ]
        if len(eligible_entities) > 2000:
            eligible_entities = list(rng.choice(eligible_entities, size=2000, replace=False))

        # Stats: how often does the retriever surface the held-out positive?
        # Diagnostic for the ranker-has-something-to-learn question.
        n_positive_in_retrieved = 0

        for entity in eligible_entities:
            history = self._history_by_entity.get(entity, ())
            if len(history) < 2:
                continue
            positive = history[-1]
            prior_history = history[:-1]
            prior_owned_arr = np.asarray(list(prior_history), dtype=object)
            prior_owned_set = set(prior_history)

            # Run the retriever exactly as recommend() does, but on the
            # entity's history with the last item hidden.
            raw_candidates = list(
                self._cooc_retriever.retrieve(prior_owned_arr, self.retrieval_budget)
            )
            raw_candidates.extend(
                self._path_retriever.retrieve(
                    recent_history=prior_history,
                    budget=self.retrieval_budget,
                    exclude=prior_owned_set,
                )
            )
            retrieved = _dedup_max_score(raw_candidates, self.retrieval_budget)
            if not retrieved:
                continue

            # Only train on entities where the retriever surfaces the
            # positive in its top-K. LambdaRank can only reorder what the
            # retriever produces at inference time, so training on
            # distribution-mismatched groups teaches the wrong function.
            positive_in_topk = any(c.item_id == positive for c in retrieved[:n_neg])
            if not positive_in_topk:
                continue
            n_positive_in_retrieved += 1

            # Put the positive first, then the top-(n_neg) retrieved
            # candidates excluding the positive, trimmed to group_size.
            negatives = [c for c in retrieved[:n_neg] if c.item_id != positive]
            negatives = negatives[: group_size - 1]
            positive_candidate = Candidate(
                item_id=positive, score=0.0, source="ranker_positive"
            )
            candidates = [positive_candidate, *negatives]
            if len(candidates) < 2:
                continue

            features = _compute_signal_features(
                candidates=candidates,
                owned_items=prior_owned_arr,
                query_basket=frozenset(prior_history[-MAX_QUERY_BASKET_SIZE:]),
                history=prior_history[-self.max_history_for_recommend :],
                item_graph=self._item_graph,
                tail_index=self._tail_index,
                path_tree=self._path_tree,
                basket_index=self._basket_index,
                basket_similarity=self.basket_similarity,
                cost_graph=self._cost_graph,
                entity_id=entity,
                basket_scan_cap=self.basket_scan_cap,
                rng=rng,
                item_cosine=self._item_cosine,
                als_factors=self._als_factors,
                persona_index=self._persona_index,
                lightgcn=self._lightgcn,
                temporal_graph=self._temporal_graph,
                session_cooc_graph=self._session_cooc_graph,
            )
            # Stack the Bayesian blend's posterior-mean score as a 10th
            # feature. Worst case, LambdaRank learns "just use the blend"
            # and matches it; better cases use the raw signals to
            # reorder locally. Without this the model has no baseline.
            feat_matrix = features.matrix
            if self._bayesian_blend is not None:
                blend_col = self._bayesian_blend.score(features).reshape(-1, 1)
                feat_matrix = np.hstack([feat_matrix, blend_col])
            features_list.append(feat_matrix)
            labels = np.zeros(len(candidates), dtype=np.int32)
            labels[0] = 1
            labels_list.append(labels)
            groups.append(len(candidates))

        if not features_list or sum(groups) < self.ranker_min_train_pairs:
            self._ranker = None
            return

        # Report the retrieval-hit rate: how often did the retriever
        # surface the held-out positive in its top-K? Low hit rates mean
        # LambdaRank is mostly learning "the positive isn't among the
        # retrieved alternatives" which isn't useful; high hit rates mean
        # it's learning to reorder retrieved candidates, which is exactly
        # what we want it to do at inference time.
        self._ranker_retrieval_hit_rate = (
            n_positive_in_retrieved / max(len(groups), 1)
        )

        X = np.vstack(features_list)
        y = np.concatenate(labels_list)
        group_arr = np.asarray(groups, dtype=np.int32)

        # Regularize harder than the previous version - we have smaller
        # groups (100 items each) and fewer groups (up to 2000), so a
        # shallower tree with more samples per leaf fights overfitting.
        ranker = LightGBMRanker(
            num_leaves=15,
            learning_rate=0.05,
            n_estimators=150,
            random_state=int(self.seed),
        )
        ranker.fit(features=X, labels=y, groups=group_arr)
        self._ranker = ranker

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
        # Flag explicit-session input so the prior builder can shrink path
        # priors when sessions were GMM-inferred from timestamps on ratings-
        # style data. See blend/priors.toml [session_stiffness].
        has_explicit_sessions = bool("session_id" in self._interactions.columns)
        return DataFeatures(
            graph_density=float(density),
            clustering_coefficient=float(clustering),
            session_density=float(session_density),
            catalog_to_entity_ratio=float(n_items) / max(n_entities, 1),
            n_interactions=n_interactions,
            has_explicit_sessions=has_explicit_sessions,
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
            cost_graph=self._cost_graph,
            entity_id=entity,
            basket_scan_cap=self.basket_scan_cap,
            rng=self._rng,
            item_cosine=self._item_cosine,
            als_factors=self._als_factors,
            persona_index=self._persona_index,
            lightgcn=self._lightgcn,
            temporal_graph=self._temporal_graph,
            session_cooc_graph=self._session_cooc_graph,
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
        # Phase 4 re-rank parameters.
        diversity: float = 0.0,
        temperature: TemperatureInput = 0.0,
        temperature_solver: str = "beam",
        temperature_beam_width: int = 10,
        calibration_weight: float = 0.0,
        emphasis: str | None = None,
        lift_weight: float = 1.0,
    ) -> list[Recommendation]:
        """Return up to ``n`` recommendations for the given entity.

        Phase 4 adds re-rank controls layered on top of the Bayesian score:

        * ``diversity`` in ``[0, 1]``: DPP diversity weight.
        * ``temperature``: per-position novelty control (scalar / list /
          named profile / dict per PRD §7.3).
        * ``temperature_solver``: ``"beam"`` (default), ``"greedy"``, or
          ``"dpp"``.
        * ``calibration_weight`` in ``[0, 1]``: Steck 2018 category
          calibration. Requires ``item_metadata`` at construction.
        * ``emphasis``: ``"distinctive"`` activates lift emphasis using
          population baselines cached at fit time.
        * ``lift_weight`` in ``[0, 1]``: strength of the lift boost when
          ``emphasis="distinctive"``.
        """
        self._require_fitted()
        assert self._tail_index is not None
        assert self._path_tree is not None
        assert self._basket_index is not None

        owned_items = self._owned_by_entity.get(entity_id, np.array([]))
        owned_set: set[object] = set(owned_items.tolist()) if owned_items.size else set()
        history = self._history_by_entity.get(entity_id, ())
        query_basket: frozenset[object] = frozenset(history[-MAX_QUERY_BASKET_SIZE:])

        # When repeat-consumption is enabled, owned items are NOT excluded
        # from retrieval - the multiplier after scoring decides what to
        # suppress (pattern-4 / too-recent replenishment) vs. re-surface
        # (pattern-1 / past-refractory items).
        repeat_active = (
            self.repeat_config is not None
            and self.repeat_config.enabled
            and self._repeat_table is not None
        )
        retrieval_exclude: set[object] = set() if repeat_active else owned_set

        # Stage 1: run the data-adaptive retriever stack + RRF fuse.
        # Each retriever produces its own ranked candidate list; RRF
        # merges them by rank (score-scale-independent). Replaces the
        # max-score merge that was dominated by cooc's raw-magnitude
        # scores (ADR-retriever-union).
        candidates = self._retrieve_rrf(
            entity_id=entity_id,
            owned_items=owned_items,
            owned_set=retrieval_exclude,
            history=history,
            query_basket=query_basket,
        )

        if constraints:
            candidates = apply_constraints(candidates, constraints)

        if not candidates:
            # Cold-start / empty-retrieval fallback: rank by global popularity,
            # drop items the entity already owns, apply constraints. Returns
            # minimal-signal recommendations (no credible interval, trivial
            # explanation) rather than failing with []. Single largest
            # accuracy win on unseen-entity traffic per ADR-growth-curves.
            return self._cold_start_fallback(
                owned_set=owned_set, constraints=constraints, n=n
            )

        # Stage 2: score each candidate on each signal. Skip signals whose
        # posterior weight is below the threshold - they contribute
        # negligibly to the blend but can dominate compute (basket_index
        # in particular on ratings data). The outcome-training path
        # always computes all signals (passes skip_weight_threshold=0).
        posterior_for_skip = (
            self._bayesian_blend.posterior_mean
            if self._bayesian_blend is not None and self.use_bayesian_blend
            else None
        )
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
            cost_graph=self._cost_graph,
            entity_id=entity_id,
            basket_scan_cap=self.basket_scan_cap,
            rng=self._rng,
            item_cosine=self._item_cosine,
            als_factors=self._als_factors,
            persona_index=self._persona_index,
            lightgcn=self._lightgcn,
            temporal_graph=self._temporal_graph,
            session_cooc_graph=self._session_cooc_graph,
            posterior_mean=posterior_for_skip,
            skip_weight_threshold=self.skip_signal_weight_threshold,
        )
        # Per-query normalization so no signal's raw magnitude dominates
        # the linear blend. Applied IN PLACE to features.matrix - blend
        # scoring and any downstream use see the normalized values.
        # Fixes the "dead in the blend" pattern ADR-signal-audit documented.
        if self.signal_normalization != "none":
            features = SignalFeatures(
                matrix=normalize_columns(features.matrix, mode=self.signal_normalization),
                signal_names=features.signal_names,
            )
        # Stage 2: score via LambdaRank (warm regime), Bayesian blend mean
        # (cold regime), or heuristic blend (no Bayesian).  All three paths
        # operate on the same SignalFeatures. LambdaRank replaces the raw
        # score; credible intervals still come from the Bayesian blend for
        # consumers that want uncertainty quantification.
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

        # Gating network: per-entity learned softmax over the K signals.
        # When fitted, its weights replace whatever the blend computed
        # for ordering. Credible intervals above remain (from Bayesian
        # path) for uncertainty-surfacing consumers - the gate doesn't
        # provide its own CIs in v1.
        if self._gate is not None:
            ctx = self._gate_context_cache.get(entity_id)
            if ctx is not None:
                gate_weights = self._gate.forward(ctx)
                scores = np.asarray(features.matrix @ gate_weights, dtype=np.float64)
                weights_for_explanation = {
                    name: float(w)
                    for name, w in zip(features.signal_names, gate_weights, strict=True)
                }

        # Layered scoring: cooc primary + cumulative one-tailed z-gated
        # boost layers. Replaces blend/gate scoring when enabled.
        # Auto-calibrated (z, boost) was picked at fit time. Pulls
        # individual layer columns directly from the SignalFeatures
        # matrix.
        if self.layered_scoring and self.layered_config is not None:
            primary_idx = SIGNAL_ORDER.index("cooccurrence")
            primary = features.matrix[:, primary_idx]
            layer_indices = []
            for layer_name in ("path_basket", "session_cooccurrence", "temporal_cooccurrence"):
                if layer_name in SIGNAL_ORDER:
                    layer_indices.append(SIGNAL_ORDER.index(layer_name))
            layers = [features.matrix[:, idx] for idx in layer_indices]
            scores = layered_score(primary, layers, config=self.layered_config)
            weights_for_explanation = {
                "cooccurrence": 1.0,
                "layered_z": float(self.layered_config.z_threshold),
                "layered_boost_multiplier": float(self.layered_config.boost_multiplier),
            }

        # Warm regime: LightGBM LambdaRank over the feature matrix +
        # blend-mean-as-feature (10th column). The blend score +
        # credible intervals above remain for explanation and uncertainty
        # surfaces; the ranker replaces the score used for ordering.
        # This is PRD §6.3's "warm regime handoff" in v1.
        if self._ranker is not None and self._ranker.is_fitted:
            feat_matrix = features.matrix
            if self._bayesian_blend is not None:
                blend_col = np.asarray(scores).reshape(-1, 1)
                feat_matrix = np.hstack([feat_matrix, blend_col])
            scores = self._ranker.score_features(feat_matrix)

        # Repeat-consumption multiplier (between stage 2 and stage 3).
        # For each candidate, look up the entity's last interaction with
        # that item; compute a per-item multiplier from the profile;
        # multiply into the score. Non-owned candidates get multiplier
        # 1.0 (no history, no adjustment). Rerank then sees the
        # time-adjusted scores and won't diversify over suppressed items.
        if repeat_active and self._repeat_table is not None:
            one_shot_eps = float(self.repeat_config.one_shot_epsilon)  # type: ignore[union-attr]
            ref_ts = (
                float(self._reference_timestamp)
                if self._reference_timestamp is not None
                else 0.0
            )
            multipliers = np.ones(len(candidates), dtype=np.float64)
            for i, c in enumerate(candidates):
                last_ts = self._last_interaction_ts.get((entity_id, c.item_id))
                time_since = None if last_ts is None else max(ref_ts - last_ts, 0.0)
                profile = self._repeat_table.get(c.item_id)
                multipliers[i] = repeat_multiplier(
                    profile=profile,
                    time_since_last_seconds=time_since,
                    one_shot_epsilon=one_shot_eps,
                )
            # Convert multiplier to additive log-penalty. Preserves
            # multiplier=1.0 -> zero change, gives strong suppression
            # for pattern-4 (log(1e-3) ~= -6.9), and composes correctly
            # with negative z-scored blend scores.
            scores = scores + np.log(np.maximum(multipliers, 1e-20))

        # Stage 3 re-rank pipeline: lift -> diversity (DPP) -> calibration
        # -> temperature -> top-N. Each step transforms the score or the
        # candidate ordering. Constraints already applied before ranking
        # (plan departure from PRD §7.6). Order matches PRD §7.1 except
        # lift moves first so it influences diversity and calibration.
        item_ids_all = [c.item_id for c in candidates]

        # 1. Lift emphasis.
        if emphasis == "distinctive" and self._population_baselines is not None:
            scores = apply_lift(
                scores=scores,
                item_ids=item_ids_all,
                baselines=self._population_baselines,
                weight=lift_weight,
            )

        # 2. Starting order from the current (possibly lift-adjusted) scores.
        ordered_indices: list[int] = list(np.argsort(-scores))

        # 3. Diversity re-rank (DPP).
        if diversity > 0.0 and self._diversity_kernel is not None:
            dpp = DPPGreedy(kernel=self._diversity_kernel, diversity_weight=diversity)
            dpp_k = min(max(n * 3, n), len(candidates))
            dpp_order = dpp.rerank(
                item_ids=item_ids_all,
                qualities=np.maximum(scores, 0.0),
                k=dpp_k,
            )
            if dpp_order:
                ordered_indices = dpp_order + [i for i in ordered_indices if i not in dpp_order]

        # 4. Calibration (Steck).
        if calibration_weight > 0.0 and self._category_index is not None:
            ordered_indices = apply_calibration(
                ordered_indices=ordered_indices,
                item_ids=item_ids_all,
                scores=scores,
                entity_id=entity_id,
                index=self._category_index,
                weight=calibration_weight,
                k=max(len(ordered_indices), n),
            )

        # 5. Temperature optimization (per-position novelty control).
        temps = resolve_temperature(temperature, n=n)
        if float(np.max(temps)) > 0.0:
            # Novelty: inverse of population baseline (rare = novel).
            if self._population_baselines is not None:
                baseline_vec = self._population_baselines.lookup_many(item_ids_all)
                novelty = 1.0 / np.maximum(baseline_vec, 1e-9)
            else:
                novelty = np.ones_like(scores)
            # Restrict to the pre-temperature candidate pool to keep
            # computations bounded.
            pool = ordered_indices[: min(len(ordered_indices), max(50, n * 5))]
            pool_ids = [item_ids_all[i] for i in pool]
            pool_scores = np.asarray([max(scores[i], 1e-9) for i in pool], dtype=np.float64)
            pool_novelty = np.asarray([novelty[i] for i in pool], dtype=np.float64)
            objective = TemperatureObjective(scores=pool_scores, novelty=pool_novelty)
            chosen_local = solve_temperature(
                objective=objective,
                temperatures=temps,
                n_positions=n,
                solver=temperature_solver,
                beam_width=temperature_beam_width,
                item_ids=pool_ids,
                kernel_dpp=(
                    DPPGreedy(kernel=self._diversity_kernel, diversity_weight=diversity)
                    if temperature_solver == "dpp" and self._diversity_kernel is not None
                    else None
                ),
            )
            ordered_indices = [pool[j] for j in chosen_local]

        top = ordered_indices[:n]

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

    def _ensure_retriever_stack(self) -> None:
        """Rebuild the retriever stack if missing (post-load state)."""
        if self._retriever_stack or self._item_graph is None:
            return
        # DataFeatures is only used for the has_explicit_sessions gate,
        # which we cached at fit time. Build a minimal stand-in here.
        from kindling.blend.priors import DataFeatures

        features = DataFeatures(
            graph_density=0.0,
            clustering_coefficient=0.0,
            session_density=0.0,
            catalog_to_entity_ratio=0.0,
            n_interactions=0,
            has_explicit_sessions=self._has_explicit_sessions,
        )
        self._retriever_stack = build_retriever_stack(
            engine=self, features=features, retrieval_budget=self.retrieval_budget
        )

    def _retrieve_rrf(
        self,
        entity_id: object,
        owned_items: np.ndarray,
        owned_set: set[object],
        history: tuple[object, ...],
        query_basket: frozenset[object],
        rrf_k: float = 60.0,
    ) -> list[Candidate]:
        """Reciprocal Rank Fusion over the data-adaptive retriever stack.

        For each retriever, get its ranked candidate list, then accumulate
        ``rrf_weight / (rrf_k + rank)`` per item across retrievers. Items
        appearing high across multiple retrievers win. Score-scale-
        independent, so cooc's raw-magnitude scores don't dominate.
        """
        self._ensure_retriever_stack()
        stack = self._retriever_stack
        if not stack:
            return []

        # When repeat-consumption is active, owned_set is empty (caller
        # passed set()) and we also want the cooc retriever to include
        # owned items in its candidate list so the multiplier can
        # choose suppression per-pattern.
        include_owned = (
            self.repeat_config is not None
            and self.repeat_config.enabled
            and self._repeat_table is not None
        )

        rrf_scores: dict[object, float] = {}
        first_source: dict[object, str] = {}
        for entry in stack:
            name = entry.name
            r = entry.retriever
            per_budget = entry.budget
            weight = entry.rrf_weight
            if name == "cooccurrence":
                cands = r.retrieve(  # type: ignore[attr-defined]
                    owned_items, per_budget, include_owned=include_owned
                )
            elif name == "item_item_cosine":
                cands = r.retrieve(  # type: ignore[attr-defined]
                    owned_items=owned_items, budget=per_budget, exclude=owned_set
                )
            elif name == "als_factor":
                cands = r.retrieve(  # type: ignore[attr-defined]
                    entity_id=entity_id, budget=per_budget, exclude=owned_set
                )
            elif name == "path_basket":
                cands = r.retrieve(  # type: ignore[attr-defined]
                    query_basket=query_basket, budget=per_budget, exclude=owned_set
                )
            elif name == "persona":
                cands = r.retrieve(  # type: ignore[attr-defined]
                    entity_id=entity_id,
                    owned_items=owned_items,
                    history=history,
                    budget=per_budget,
                    exclude=owned_set,
                )
            else:
                continue
            for rank_zero, c in enumerate(cands):
                if c.item_id in owned_set:
                    continue
                delta = weight / (rrf_k + (rank_zero + 1))
                rrf_scores[c.item_id] = rrf_scores.get(c.item_id, 0.0) + delta
                first_source.setdefault(c.item_id, name)

        ordered = sorted(rrf_scores.items(), key=lambda kv: -kv[1])[: self.retrieval_budget]
        return [
            Candidate(item_id=item, score=score, source=first_source.get(item, "rrf"))
            for item, score in ordered
        ]

    def _cold_start_fallback(
        self,
        owned_set: set[object],
        constraints: list[ConstraintPredicate] | None,
        n: int,
    ) -> list[Recommendation]:
        """Popularity-ranked fallback for entities that produced no
        candidates in retrieval. Applied when an entity is unseen (no
        history in training) or when retrievers returned nothing.

        The explanation surfaces the fallback transparently so consumers
        can distinguish cold-start recommendations from signal-driven
        ones.
        """
        if not self._popular_items_ranked:
            return []
        picks: list[object] = []
        for item in self._popular_items_ranked:
            if item in owned_set:
                continue
            if constraints and not all(p(item) for p in constraints):
                continue
            picks.append(item)
            if len(picks) >= n:
                break
        if not picks:
            return []
        baseline = self._population_baselines
        return [
            Recommendation(
                item_id=item,
                score=float(baseline.item_to_baseline.get(item, 0.0)) if baseline else 0.0,
                explanation=Explanation(
                    primary="cold-start: ranked by population popularity.",
                    debug_payload={"fallback": "popularity"},
                ),
                credible_interval=None,
                credible_coverage=None,
            )
            for item in picks
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
        # Phase 5: surface simple-reporter calibration degradation and
        # negative-signal-mode context so production monitoring catches
        # silent quality drift.
        summary["negative_signal_mode"] = self.negative_signal_mode
        summary["outcome_log_size"] = len(self.outcome_log)
        diag = summary.get("diagnostics") or {}
        warnings_list = (
            list(diag["warnings"]) if isinstance(diag, dict) and "warnings" in diag else []
        )
        if self.outcome_log.has_simple_mode_records():
            warnings_list.append(
                "Simple-mode outcome reports in the log. Position-bias "
                "correction is disabled for those rows; the resulting "
                "posterior calibration is approximate. Use "
                "engine.report_outcome(...) with impression tracking "
                "for full calibration."
            )
        if warnings_list:
            diag = summary.setdefault("diagnostics", {})
            if isinstance(diag, dict):
                diag["warnings"] = warnings_list
        return summary

    # ---- Phase 5 outcome reporting + refit ---------------------------------

    def report_outcome(
        self,
        *,
        entity_id: object,
        recommendation_id: str,
        shown_items: list[object],
        selected_items: list[object] | set[object] | None = None,
        rejected_items: list[object] | set[object] | None = None,
        positions: list[int] | None = None,
        timestamp: datetime | None = None,
    ) -> int:
        """Precise-mode outcome report (PRD §6.6).

        Returns the number of new rows inserted (duplicates are silently
        deduped via the primary key).
        """
        self._require_fitted()
        return self.outcome_log.report_precise(
            entity_id=entity_id,
            recommendation_id=recommendation_id,
            shown_items=shown_items,
            selected_items=selected_items,
            rejected_items=rejected_items,
            positions=positions,
            timestamp=timestamp,
        )

    def report_interaction(
        self,
        *,
        entity_id: object,
        item_id: object,
        action: str,
        rating: float | None = None,
        timestamp: datetime | None = None,
    ) -> int:
        """Simple-mode report (PRD §6.6). Position-bias correction is
        disabled for these rows; posterior_summary() surfaces a warning
        when any simple rows exist."""
        self._require_fitted()
        return self.outcome_log.report_simple(
            entity_id=entity_id,
            item_id=item_id,
            action=action,
            rating=rating,
            timestamp=timestamp,
        )

    def report_outcome_correction(
        self,
        *,
        entity_id: object,
        recommendation_id: str,
        item_id: object,
        shown: bool,
        selected: bool,
        rejected: bool = False,
        rating: float | None = None,
        position: int = 1,
        timestamp: datetime | None = None,
    ) -> None:
        """Supersede a prior row. See OutcomeLog.report_correction."""
        self._require_fitted()
        self.outcome_log.report_correction(
            entity_id=entity_id,
            recommendation_id=recommendation_id,
            item_id=item_id,
            shown=shown,
            selected=selected,
            rejected=rejected,
            rating=rating,
            position=position,
            timestamp=timestamp,
        )

    def refit_posterior(self, max_iter: int | None = None) -> DiagnosticsReport | None:
        """Re-fit the Bayesian posterior from the current outcome log.

        Returns the updated DiagnosticsReport (or None when there is no
        Bayesian blend). Use this after a batch of ``report_outcome``
        calls to incorporate fresh feedback into the posterior. Phase 6
        will add scheduled / continuous refit options; Phase 5 is
        on-demand.
        """
        self._require_fitted()
        if self._bayesian_blend is None:
            return None
        if len(self.outcome_log) == 0:
            return self._diagnostics

        batch = replay_to_batch(self.outcome_log, self._build_signal_row_for_outcome)
        if batch.n_outcomes == 0:
            return self._diagnostics
        self._bayesian_blend.fit_posterior(
            batch=batch,
            likelihood=self.likelihood,
            rng=np.random.default_rng(self.seed),
            max_iter=max_iter if max_iter is not None else self.vi_max_iter,
        )
        self._diagnostics = run_diagnostics(
            blend=self._bayesian_blend,
            batch=batch,
            likelihood=self.likelihood,
            rng=np.random.default_rng(self.seed + 1),
        )
        return self._diagnostics

    def _build_signal_row_for_outcome(
        self,
        entity: object,
        item: object,
    ) -> np.ndarray | None:
        """Replay hook: signal vector for (entity, item) under current
        fitted state. Returns None when either is unknown.

        The outcome log stores ids as strings for a uniform primary key;
        this method resolves back to the original id types used by the
        item graph and owned-set dict.
        """
        resolved_entity = _resolve_id(entity, self._owned_by_entity)
        if resolved_entity is None:
            return None
        owned_arr = self._owned_by_entity[resolved_entity]
        assert self._item_graph is not None
        resolved_item = _resolve_id(item, self._item_graph.item_index)
        if resolved_item is None:
            return None
        features = self._build_features_for_outcome(
            entity=resolved_entity,
            items=[resolved_item],
            owned_arr=owned_arr,
        )
        return np.asarray(features.matrix[0].copy(), dtype=np.float64)

    # ---- Phase 10 persistence --------------------------------------------

    def save(self, path: "str | Path") -> None:
        """Write the fitted engine state to a gzipped file (PRD §10.4)."""
        from kindling.persist import save_engine

        save_engine(self, path)

    @classmethod
    def load(
        cls,
        path: "str | Path",
        registry: "dict[str, Any] | None" = None,
    ) -> Engine:
        """Reconstruct a fitted engine from a saved file.

        ``registry`` maps qualified plugin names to factory callables,
        used when the saved manifest references a user-supplied
        pluggable component (retriever, DPP kernel, etc.). Built-in
        kindling plugins are resolved automatically.
        """
        from kindling.persist import load_engine

        return load_engine(path, registry=registry)

    def export_arrow(self, path: "str | Path") -> None:
        """Export item graph + posterior params as Apache Arrow IPC
        files for cross-language consumption (PRD §10.4). Requires
        the optional ``pyarrow`` dependency."""
        from kindling.persist import export_arrow

        export_arrow(self, path)

    # ---- Phase 6 lifecycle surface ----------------------------------------

    def prune(self) -> list[PreservedAggregate]:
        """Apply the configured pruning policy to all derived structures.

        Returns a list of ``PreservedAggregate`` records describing what
        was dropped, one per structure. Safe to call repeatedly; the
        second call is a near-no-op (pruning is idempotent).
        """
        self._require_fitted()
        if not self.pruning_config.enabled:
            return []
        threshold = self.pruning_config.support_threshold
        aggregates: list[PreservedAggregate] = []

        for name, structure in (
            ("tail_index", self._tail_index),
            ("path_tree", self._path_tree),
            ("basket_index", self._basket_index),
            ("item_graph", self._item_graph),
            ("cost_graph", self._cost_graph),
        ):
            if structure is None:
                continue
            n, weight = structure.prune_below(threshold)
            if n > 0:
                aggregates.append(
                    PreservedAggregate(
                        structure_name=name,
                        n_pruned_entries=n,
                        total_pruned_weight=weight,
                        config=self.pruning_config,
                    )
                )
        self._preserved_aggregates.extend(aggregates)
        return aggregates

    def drift_report(self) -> dict[str, object]:
        """Compute drift metrics against the training interactions (PRD
        §3.5). Updates the drift tracker's baseline on the first call
        and compares subsequent calls against it."""
        self._require_fitted()
        assert self._interactions is not None
        report = self._drift_tracker.compute(self._interactions)
        return report.to_dict()

    @property
    def last_drift_report(self) -> DriftReport | None:
        """Most recent drift report, or ``None`` if not computed yet."""
        return self._drift_tracker.last_report

    @property
    def preserved_aggregates(self) -> list[PreservedAggregate]:
        """All pruning aggregates recorded since fit. Used by the
        Bayesian blend to account for data the posterior should know
        exists but has had its detail dropped."""
        return list(self._preserved_aggregates)

    # ---- internals --------------------------------------------------------

    def _require_fitted(self) -> None:
        # A fitted engine has at minimum an item graph + owned-set
        # caches. We don't require ``self._interactions`` because
        # persistence drops the raw DataFrame to keep save files small.
        if self._item_graph is None:
            raise EngineNotFittedError("Engine.fit must be called before use")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _resolve_id(raw: object, mapping: Mapping[object, object]) -> object | None:
    """Outcome-log entries store ids as strings; the item graph and owned-
    sets use the original types. Try the raw value, then int, then float
    conversion before giving up."""
    if raw in mapping:
        return raw
    if isinstance(raw, str):
        try:
            as_int = int(raw)
            if as_int in mapping:
                return as_int
        except ValueError:
            pass
        try:
            as_float = float(raw)
            if as_float in mapping:
                return as_float
        except ValueError:
            pass
    return None


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
    cost_graph: CostGraph | None = None,
    entity_id: object = None,
    basket_scan_cap: int | None = None,
    rng: np.random.Generator | None = None,
    item_cosine: ItemCosineMatrix | None = None,
    als_factors: ALSFactors | None = None,
    persona_index: "PersonaIndex | None" = None,
    lightgcn: "LightGCNModel | None" = None,
    temporal_graph: "TemporalInteractionGraph | None" = None,
    session_cooc_graph: "SessionCooccurrenceGraph | None" = None,
    posterior_mean: np.ndarray | None = None,
    skip_weight_threshold: float = 0.0,
) -> SignalFeatures:
    """Compute the (N_candidates, K_signals) feature matrix in SIGNAL_ORDER.

    When ``posterior_mean`` is provided and a signal's posterior weight is
    below ``skip_weight_threshold``, its column stays zero rather than
    being computed. The blend score is a weighted sum, so a near-zero-
    weight signal contributes near-zero to the final score either way.
    Saves the dominant basket_index.score_many call on ratings-style
    data where path_basket weight shrinks to <0.05 post-stiffness.
    """
    n = len(candidates)
    matrix = np.zeros((n, len(SIGNAL_ORDER)), dtype=np.float64)
    cand_ids = [c.item_id for c in candidates]

    # Which signals are "hot enough" to compute. When the threshold is 0
    # or no posterior is available, compute everything (preserves debug
    # payload completeness).
    def _hot(i: int) -> bool:
        if skip_weight_threshold <= 0.0 or posterior_mean is None:
            return True
        return bool(posterior_mean[i] >= skip_weight_threshold)

    last_item = history[-1] if history else None
    # path_full, path_tail, path_basket
    # NOTE: path_full is in DEACTIVATED_SIGNALS; column left at zero so
    # the Bayesian posterior drives its weight to zero naturally.
    # Uncomment to re-enable: path_full was consistently the weakest
    # signal across all evaluated datasets (NDCG < 0.05).
    # if _hot(0):
    #     matrix[:, 0] = path_tree.score_many(cand_ids, history)
    if _hot(1):
        matrix[:, 1] = tail_index.score_many(cand_ids, last_item)
    if _hot(2):
        matrix[:, 2] = basket_index.score_many(
            cand_ids,
            query_basket=query_basket,
            similarity=basket_similarity,
            scan_cap=basket_scan_cap,
            rng=rng,
        )
    # cooccurrence - recompute from the graph against the entity's owned set.
    # Using the retriever's max-score would let the path_endpoint retriever's
    # score leak into this feature for candidates it won via dedup.
    if _hot(3):
        matrix[:, 3] = _cooccurrence_signal(cand_ids, owned_items, item_graph)

    # Cost signals (PRD §3.6). Store as negative values so positive
    # Dirichlet weights translate into penalties. Zero when no cost graph
    # is active (positive_only mode).
    if cost_graph is not None:
        owned_set = frozenset(owned_items.tolist()) if owned_items.size else frozenset()
        if _hot(4):
            matrix[:, 4] = -cost_graph.population_costs_many(cand_ids)
        if _hot(5):
            matrix[:, 5] = -cost_graph.entity_costs_many(cand_ids, entity_id)
        if _hot(6):
            matrix[:, 6] = -cost_graph.context_costs_many(cand_ids, owned_set)

    # item_item_cosine (8th signal). Cosine kNN directly scored against
    # the entity's owned items, normalized to [0, 1].
    if item_cosine is not None and owned_items.size > 0 and _hot(7):
        cand_indices = np.fromiter(
            (item_graph.item_index.get(c, -1) for c in cand_ids),
            dtype=np.int64,
            count=len(cand_ids),
        )
        owned_idx_list = [
            item_graph.item_index[o]
            for o in owned_items.tolist()
            if o in item_graph.item_index
        ]
        owned_indices = np.asarray(owned_idx_list, dtype=np.int64)
        valid = cand_indices >= 0
        if valid.any() and owned_indices.size > 0:
            scores = item_cosine.score_many(cand_indices[valid], owned_indices)
            full = np.zeros(len(cand_ids), dtype=np.float64)
            full[valid] = scores
            matrix[:, 7] = full

    # als_factor (9th signal). Latent-factor score = U_entity @ V_item.
    # Complements the neighborhood signals by generalizing across items
    # with no direct cooccurrence. Silent no-op when `implicit` wasn't
    # installed at fit time.
    if als_factors is not None and _hot(8):
        cand_indices_als = np.fromiter(
            (item_graph.item_index.get(c, -1) for c in cand_ids),
            dtype=np.int64,
            count=len(cand_ids),
        )
        valid_als = cand_indices_als >= 0
        if valid_als.any():
            als_scores = als_factors.score_many(entity_id, cand_indices_als[valid_als])
            full_als = np.zeros(len(cand_ids), dtype=np.float64)
            full_als[valid_als] = als_scores
            matrix[:, 8] = full_als

    # persona (10th signal). Group-level taste score = sum_P match(P) *
    # persona_weight(c, P). Zero for entities the clustering didn't place
    # (noise points) and for candidates not in the persona vocabulary.
    if persona_index is not None and _hot(9) and persona_index.n_personas > 0:
        from kindling.personas.matching import (
            build_user_query_vector,
            match_user,
            score_candidates,
        )

        user_vec = build_user_query_vector(
            owned_items=owned_items, history_items=history, index=persona_index
        )
        matches = match_user(user_vec, persona_index)
        if matches.any():
            matrix[:, 9] = score_candidates(matches, persona_index, cand_ids)

    # lightgcn (11th signal). Graph-propagated latent-factor dot product.
    # Complements als_factor by using the full bipartite graph with
    # symmetric-degree normalization rather than the ALS objective.
    if lightgcn is not None and _hot(10):
        cand_indices_gcn = np.fromiter(
            (item_graph.item_index.get(c, -1) for c in cand_ids),
            dtype=np.int64,
            count=len(cand_ids),
        )
        valid_gcn = cand_indices_gcn >= 0
        if valid_gcn.any():
            gcn_scores = lightgcn.score_many(entity_id, cand_indices_gcn[valid_gcn])
            full_gcn = np.zeros(len(cand_ids), dtype=np.float64)
            full_gcn[valid_gcn] = gcn_scores
            matrix[:, 10] = full_gcn

    # temporal_cooccurrence (12th signal). Direct kernel-weighted pair
    # lookup on the temporal interaction graph - same scoring mechanism
    # as the count-based cooccurrence (matrix[:, 3]) but with
    # time-decay-weighted edges. When the substrate's session-presence
    # guard detects rating-burst timestamps (ml1m-like), the kernel
    # auto-falls-back to pure-count and this signal becomes a near-
    # duplicate of cooccurrence (Bayesian posterior decorrelates it).
    # On real-session data the kernel earns meaningful lift
    # (grocery: +37% NDCG standalone vs pure-count).
    if temporal_graph is not None and temporal_graph.n_edges > 0 and _hot(11):
        owned_idx = np.fromiter(
            (temporal_graph.item_index.get(o, -1) for o in owned_items.tolist()),
            dtype=np.int64, count=owned_items.size,
        ) if owned_items.size else np.empty(0, dtype=np.int64)
        owned_idx = owned_idx[owned_idx >= 0]
        if owned_idx.size > 0:
            excl_set = {int(i) for i in owned_idx.tolist()}
            tc_full = temporal_graph.score_against_owned(owned_idx, exclude_indices=excl_set)
            cand_indices_tc = np.fromiter(
                (temporal_graph.item_index.get(c, -1) for c in cand_ids),
                dtype=np.int64, count=len(cand_ids),
            )
            valid_tc = cand_indices_tc >= 0
            if valid_tc.any():
                out_tc = np.zeros(len(cand_ids), dtype=np.float64)
                out_tc[valid_tc] = tc_full[cand_indices_tc[valid_tc]]
                matrix[:, 11] = out_tc

    # session_cooccurrence (13th signal). Same scoring shape as cooc,
    # but the underlying graph counts session-co-membership (S.T @ S
    # with S indexed by session_id) instead of entity-co-occurrence.
    # The graph is None when sessions aren't deep enough (<30% of
    # sessions have 2+ items by default), in which case the column
    # stays zero and the Bayesian posterior learns weight 0.
    if session_cooc_graph is not None and session_cooc_graph.n_edges > 0 and _hot(12):
        owned_idx_sc = np.fromiter(
            (session_cooc_graph.item_index.get(o, -1) for o in owned_items.tolist()),
            dtype=np.int64, count=owned_items.size,
        ) if owned_items.size else np.empty(0, dtype=np.int64)
        owned_idx_sc = owned_idx_sc[owned_idx_sc >= 0]
        if owned_idx_sc.size > 0:
            excl_set_sc = {int(i) for i in owned_idx_sc.tolist()}
            sc_full = session_cooc_graph.score_against_owned(
                owned_idx_sc, exclude_indices=excl_set_sc,
            )
            cand_indices_sc = np.fromiter(
                (session_cooc_graph.item_index.get(c, -1) for c in cand_ids),
                dtype=np.int64, count=len(cand_ids),
            )
            valid_sc = cand_indices_sc >= 0
            if valid_sc.any():
                out_sc = np.zeros(len(cand_ids), dtype=np.float64)
                out_sc[valid_sc] = sc_full[cand_indices_sc[valid_sc]]
                matrix[:, 12] = out_sc

    return SignalFeatures(matrix=matrix, signal_names=SIGNAL_ORDER)


def _cooccurrence_signal(
    cand_ids: list[object],
    owned_items: np.ndarray,
    item_graph: ItemGraph,
) -> np.ndarray:
    """Sum of item-graph edges between each candidate and the owned set.

    Routes to the Rust extension when available. The native path folds
    the row-sum and per-candidate gather into one pass over selected
    rows, skipping the intermediate ``np.asarray(sum(axis=0))``.
    """
    if item_graph.n_items == 0 or owned_items.size == 0:
        return np.zeros(len(cand_ids), dtype=np.float64)
    owned_indices = [item_graph.item_index[i] for i in owned_items if i in item_graph.item_index]
    if not owned_indices:
        return np.zeros(len(cand_ids), dtype=np.float64)

    if NATIVE_AVAILABLE and kindling_native is not None:
        cand_slot_indices = [item_graph.item_index.get(cid, -1) for cid in cand_ids]
        adj = item_graph.adjacency
        result = kindling_native.cooccurrence_signal(
            adj.data.astype(np.float32, copy=False),
            adj.indices.astype(np.int32, copy=False),
            adj.indptr.astype(np.int32, copy=False),
            owned_indices,
            cand_slot_indices,
        )
        return np.asarray(result, dtype=np.float64)

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
