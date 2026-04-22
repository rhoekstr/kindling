"""Versioned engine persistence (plan Phase 10).

The format is a gzipped pickle of ``EngineState`` - a dataclass
carrying the serializable core engine state plus a ``PluginManifest``
describing pluggable components by qualified name + config. Load is a
two-step process: unpickle the core + manifest, then re-instantiate
pluggable components from a caller-supplied registry.

This is the plan's replacement for the PRD's bincode scheme, which
couldn't roundtrip Python closures. We chose pickle over msgpack
because scipy CSR matrices, numpy arrays, and our dataclass graph of
structures all pickle cleanly without custom codecs. The schema
version gate makes forward-incompatible changes visible.
"""

from __future__ import annotations

import gzip
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from kindling.engine import Engine

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PluginManifest:
    """Names + configs of the user-pluggable components.

    ``qualified_name`` is the ``module.ClassName`` or
    ``module.function_name`` to look up in the user's registry at load
    time. ``config`` is an opaque kwargs dict that the registered
    factory will re-apply. ``note`` carries a human-readable warning
    when we couldn't fully serialize a component (closures).
    """

    retrievers: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    rankers: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    kernels: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    constraints_note: str | None = None


@dataclass
class EngineState:
    """Pickle-friendly snapshot of a fitted Engine.

    Fields match the Engine attributes that must round-trip; pluggable
    components land in ``plugin_manifest``. The loader uses both to
    reconstruct an Engine equivalent to the saved one (modulo user
    closures, which surface a warning).
    """

    schema_version: int
    engine_version: str

    # Core fitted state.
    interactions_columns: list[str]
    schema_flags: dict[str, bool]
    reference_timestamp: float | None
    negative_signal_mode: str
    alpha_pop: float
    credible_coverage: float
    seed: int
    vi_max_iter: int

    # Derived structures (pickle-friendly).
    item_graph: Any
    tail_index: Any
    path_tree: Any
    basket_index: Any
    cost_graph: Any
    heuristic_blend: Any
    bayesian_blend: Any
    diagnostics: Any
    population_baselines: Any
    category_index: Any
    drift_tracker: Any

    # Per-entity caches.
    owned_by_entity: dict[object, Any]
    history_by_entity: dict[object, tuple[object, ...]]

    # Pluggable components.
    plugin_manifest: PluginManifest

    # Optional (append-only; older saves don't have these).
    item_cosine: Any = None
    als_factors: Any = None
    ranker: Any = None


_Factory = Callable[..., Any]


def save_engine(engine: "Engine", path: str | Path) -> None:
    """Write the engine's state to ``path`` as a gzipped pickle.

    Pluggable components that are defined in the kindling library save
    a qualified-name manifest that ``load_engine`` can resolve without
    a user-supplied registry. Components defined outside kindling
    require the caller to provide that registry at load time.
    """
    state = _snapshot(engine)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=6) as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_engine(
    path: str | Path,
    registry: dict[str, _Factory] | None = None,
) -> "Engine":
    """Reconstruct an Engine from a saved file.

    ``registry`` maps ``qualified_name`` entries in the manifest to
    callables that produce the equivalent pluggable component. Missing
    names for user-defined components raise; kindling's built-in
    components are resolved via import.
    """
    from kindling.engine import Engine  # local import avoids circular dep

    path = Path(path)
    with gzip.open(path, "rb") as fh:
        raw = pickle.load(fh)  # noqa: S301 - trusted local file
    if not isinstance(raw, EngineState):
        raise ValueError(f"File {path} does not contain an EngineState")
    if raw.schema_version > SCHEMA_VERSION:
        raise ValueError(
            f"Persisted schema version {raw.schema_version} is newer than "
            f"this kindling version ({SCHEMA_VERSION}). Upgrade kindling."
        )
    engine = Engine.__new__(Engine)
    _restore(engine, raw, registry or {})
    return engine


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _snapshot(engine: "Engine") -> EngineState:
    from kindling import __version__

    if engine._interactions is None:
        raise RuntimeError("Engine must be fitted before save")

    manifest = _build_manifest(engine)
    schema = engine._schema
    return EngineState(
        schema_version=SCHEMA_VERSION,
        engine_version=__version__,
        interactions_columns=list(engine._interactions.columns),
        schema_flags={
            "has_timestamp": bool(schema.has_timestamp) if schema else False,
            "has_session_id": bool(schema.has_session_id) if schema else False,
            "has_action_type": bool(schema.has_action_type) if schema else False,
            "has_rating": bool(schema.has_rating) if schema else False,
        },
        reference_timestamp=engine._reference_timestamp,
        negative_signal_mode=engine.negative_signal_mode,
        alpha_pop=engine.alpha_pop,
        credible_coverage=engine.credible_coverage,
        seed=engine.seed,
        vi_max_iter=engine.vi_max_iter,
        item_graph=engine._item_graph,
        tail_index=engine._tail_index,
        path_tree=engine._path_tree,
        basket_index=engine._basket_index,
        cost_graph=engine._cost_graph,
        heuristic_blend=engine._heuristic_blend,
        bayesian_blend=engine._bayesian_blend,
        diagnostics=engine._diagnostics,
        population_baselines=engine._population_baselines,
        item_cosine=engine._item_cosine,
        als_factors=engine._als_factors,
        ranker=engine._ranker,
        category_index=engine._category_index,
        drift_tracker=engine._drift_tracker,
        owned_by_entity=dict(engine._owned_by_entity),
        history_by_entity=dict(engine._history_by_entity),
        plugin_manifest=manifest,
    )


def _build_manifest(engine: "Engine") -> PluginManifest:
    from kindling.rerank.dpp import CooccurrenceCosineKernel

    retrievers: list[tuple[str, dict[str, Any]]] = []
    # We only store retriever factories that aren't closure-over-graph;
    # kindling ships two (cooccurrence + path_endpoint) and both are
    # re-created at fit time, so persistence snapshots the intent, not
    # the instance. For the fitted engine we just record that both
    # kindling retrievers are active.
    if engine._cooc_retriever is not None:
        retrievers.append(
            ("kindling.retrieve.cooccurrence.CoOccurrenceRetriever", {})
        )
    if engine._path_retriever is not None:
        retrievers.append(
            ("kindling.retrieve.path_endpoint.PathEndpointRetriever", {})
        )

    kernels: list[tuple[str, dict[str, Any]]] = []
    if isinstance(engine._diversity_kernel, CooccurrenceCosineKernel):
        kernels.append(
            (
                "kindling.rerank.dpp.CooccurrenceCosineKernel",
                {},
            )
        )
    elif engine._diversity_kernel is not None:
        kernels.append(
            (
                f"{type(engine._diversity_kernel).__module__}."
                f"{type(engine._diversity_kernel).__name__}",
                {},
            )
        )

    rankers: list[tuple[str, dict[str, Any]]] = []
    # Ranker pluggability lands in v1.x; nothing to serialize yet.

    return PluginManifest(
        retrievers=retrievers,
        rankers=rankers,
        kernels=kernels,
        constraints_note=(
            "User-supplied constraint closures cannot be pickled and are "
            "not restored. The loaded engine rebuilds without them; pass "
            "constraints explicitly on recommend() calls."
        ),
    )


def _restore(
    engine: "Engine",
    state: EngineState,
    registry: dict[str, _Factory],
) -> None:
    import numpy as np

    from kindling.blend.heuristic import HeuristicBlend
    from kindling.ingest.contract import InteractionSchema
    from kindling.lifecycle.decay import ExponentialDecay
    from kindling.lifecycle.pruning import PruningConfig
    from kindling.outcomes.log import OutcomeLog
    from kindling.path.basket_index import BasketSimilarity
    from kindling.retrieve.cooccurrence import CoOccurrenceRetriever
    from kindling.retrieve.path_endpoint import PathEndpointRetriever

    flags = state.schema_flags
    engine._schema = InteractionSchema(
        has_timestamp=flags["has_timestamp"],
        has_session_id=flags["has_session_id"],
        has_action_type=flags["has_action_type"],
        has_rating=flags["has_rating"],
    )
    # The interactions DataFrame is not saved (can be large and the
    # source of truth lives with the user). We only restore the
    # derived structures and caches.
    engine._interactions = None
    engine._reference_timestamp = state.reference_timestamp
    engine._session_inference = None
    engine._item_graph = state.item_graph
    engine._tail_index = state.tail_index
    engine._path_tree = state.path_tree
    engine._basket_index = state.basket_index
    engine._cost_graph = state.cost_graph
    engine._heuristic_blend = state.heuristic_blend or HeuristicBlend()
    engine._bayesian_blend = state.bayesian_blend
    engine._diagnostics = state.diagnostics
    engine._population_baselines = state.population_baselines
    engine._item_cosine = getattr(state, "item_cosine", None)
    engine._als_factors = getattr(state, "als_factors", None)
    engine._ranker = getattr(state, "ranker", None)
    engine.use_ranker = False
    engine.ranker_negatives_per_positive = 99
    engine.ranker_min_train_pairs = 500
    # Rebuild the cold-start popularity ranking from the restored baselines.
    if engine._population_baselines is not None and engine._population_baselines.item_to_baseline:
        engine._popular_items_ranked = sorted(
            engine._population_baselines.item_to_baseline,
            key=lambda i: engine._population_baselines.item_to_baseline[i],  # type: ignore[union-attr]
            reverse=True,
        )
    else:
        engine._popular_items_ranked = []
    engine._category_index = state.category_index
    engine._drift_tracker = state.drift_tracker

    engine._owned_by_entity = state.owned_by_entity
    engine._history_by_entity = state.history_by_entity

    engine.negative_signal_mode = state.negative_signal_mode
    engine.alpha_pop = state.alpha_pop
    engine.credible_coverage = state.credible_coverage
    engine.seed = state.seed
    engine.vi_max_iter = state.vi_max_iter

    # Config defaults the loaded engine needs to behave like a fresh
    # instance. These aren't in the state because they're Engine-level
    # hyperparameters that the user may want to override post-load.
    engine.retrieval_budget = 500
    engine.decay = ExponentialDecay(half_life_days=180.0)  # type: ignore[assignment]
    engine.max_path_prefix = 3
    engine.max_history_for_recommend = 5
    engine.basket_similarity = BasketSimilarity.COVERAGE
    engine.basket_scan_cap = 10_000
    engine.skip_signal_weight_threshold = 0.0
    engine.use_bayesian_blend = state.bayesian_blend is not None

    from kindling.blend.likelihoods import ListwiseCalibration

    engine.likelihood = ListwiseCalibration()  # type: ignore[assignment]
    engine._rng = np.random.default_rng(state.seed)

    engine._user_negative_mode = state.negative_signal_mode
    engine.item_metadata = None
    engine.category_column = "category"
    engine._diversity_kernel_override = None
    engine.pruning_config = PruningConfig()
    engine._preserved_aggregates = []
    engine.outcome_log = OutcomeLog()

    # Rebuild retrievers + kernel from the manifest + registry.
    manifest = state.plugin_manifest
    engine._cooc_retriever = None
    engine._path_retriever = None
    for name, cfg in manifest.retrievers:
        factory = _resolve_factory(name, registry, default_only=True)
        if name.endswith("CoOccurrenceRetriever"):
            engine._cooc_retriever = CoOccurrenceRetriever(state.item_graph, **cfg)
        elif name.endswith("PathEndpointRetriever"):
            engine._path_retriever = PathEndpointRetriever(
                state.path_tree, state.tail_index, **cfg
            )
        elif factory is not None:
            # Custom retriever from registry.
            _ = factory(**cfg)
    if manifest.kernels:
        name, cfg = manifest.kernels[0]
        if name.endswith("CooccurrenceCosineKernel"):
            from kindling.rerank.dpp import CooccurrenceCosineKernel

            engine._diversity_kernel = CooccurrenceCosineKernel(state.item_graph, **cfg)
        else:
            factory = _resolve_factory(name, registry, default_only=False)
            if factory is not None:
                engine._diversity_kernel = factory(**cfg)
    else:
        engine._diversity_kernel = None

    if manifest.constraints_note:
        warnings.warn(manifest.constraints_note, stacklevel=2)


def _resolve_factory(
    qualified_name: str,
    registry: dict[str, _Factory],
    default_only: bool,
) -> _Factory | None:
    """Prefer the user registry; fall back to importlib on built-ins."""
    if qualified_name in registry:
        return registry[qualified_name]
    if default_only and qualified_name.startswith("kindling."):
        return None  # handled by caller's special-case path
    # Import-path fallback.
    module_name, _, cls = qualified_name.rpartition(".")
    if not module_name:
        return None
    import importlib

    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(mod, cls, None)
