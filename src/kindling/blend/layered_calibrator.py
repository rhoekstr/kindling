"""Fit-time auto-calibrator for the layered (cooc + adaptive boosting) scorer.

The sweep data showed that the optimal (z_threshold, boost_multiplier)
varies by dataset shape:

  grocery-deep (session-rich):   z=2.5, b=3.0
  ml1m (rating-burst):           z=2.5, b=5.0
  amazon-beauty (sparse):        z=3.0, b=5.0
  yelp (no timestamps):          ~any (only 1 layer fires)

Rather than ship one global default and accept ~1% NDCG left on the
table per dataset, the calibrator runs a small grid sweep at fit time
on a leave-one-out slice of training data and picks the cell with the
highest held-out NDCG.

Methodology:

1. Sample ``n_users`` users with at least ``min_user_interactions``
   interactions in training.
2. For each, hold out their most-recent item; use the rest as the
   "owned" set.
3. For each (z, boost) cell, score the cooc-retrieved candidate pool
   under that config and check the rank of the held-out item.
4. Pick the cell that maximizes NDCG@k.

The engine has already seen the held-out items during fit (they're in
the training data), so absolute NDCG values are inflated. That's fine
- we only use the **relative** ranking among (z, boost) cells, and
the inflation is uniform across cells.

Cost: small. ``n_users=100`` × 12 cells × cheap signal lookups ≈
few hundred milliseconds on grocery, ~10 seconds on ml1m. Negligible
vs total fit (multi-minute typically).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from kindling.blend.layered import LayeredConfig, layered_score

if TYPE_CHECKING:
    from kindling.engine import Engine


# Default sweep grid — the sweep data showed these cells span the
# observed regimes without exploring useless extremes.
DEFAULT_Z_GRID: tuple[float, ...] = (2.0, 2.5, 3.0)
DEFAULT_BOOST_GRID: tuple[float, ...] = (1.0, 3.0, 5.0)


@dataclass
class CalibrationResult:
    """Output of ``calibrate()``.

    Attributes
    ----------
    best_config:
        The chosen ``LayeredConfig`` (with z_threshold and
        boost_multiplier set to the grid winner).
    grid_results:
        List of ``{z, boost, ndcg, mrr, n_eval}`` dicts for every
        cell. Useful for diagnostics / surfacing the calibration
        trace in ``engine.posterior_summary()``.
    n_users_evaluated:
        How many users contributed to the held-out evaluation.
    elapsed_seconds:
        Wall-time of the calibration sweep.
    fallback_to_default:
        True when the grid couldn't find a clear winner (all cells
        tied within tolerance) and the calibrator fell back to the
        default config.
    """

    best_config: LayeredConfig
    grid_results: list[dict[str, float]] = field(default_factory=list)
    n_users_evaluated: int = 0
    elapsed_seconds: float = 0.0
    fallback_to_default: bool = False


def calibrate(
    engine: "Engine",
    z_grid: tuple[float, ...] = DEFAULT_Z_GRID,
    boost_grid: tuple[float, ...] = DEFAULT_BOOST_GRID,
    n_users: int = 200,
    min_user_interactions: int = 5,
    held_out_per_user: int = 3,
    retrieval_budget: int = 200,
    k: int = 10,
    rng_seed: int = 0,
    tie_tolerance: float = 0.003,
    sparse_data_threshold: float = 20.0,
    sparse_data_boost_ceiling: float = 3.0,
    min_lift_over_cooc_only: float = 0.0,
) -> CalibrationResult:
    """Sweep (z, boost) grid on leave-one-out training data.

    Parameters
    ----------
    engine:
        Fitted Engine. Reads ``_item_graph``, ``_session_cooc_graph``,
        ``_temporal_graph``, ``_basket_index`` for layer scoring.
    z_grid, boost_grid:
        Grid of values to evaluate.
    n_users:
        Sample size. 100 is enough for a stable winner on any dataset
        we've tested; bigger samples increase calibration cost
        linearly.
    min_user_interactions:
        Skip users with fewer than this many training interactions
        (can't construct a hold-out + owned-set otherwise).
    retrieval_budget:
        Candidate pool size for each held-out user. 200 is enough to
        capture the held-out item via cooc retrieval in most cases;
        bigger is more honest but slower.
    k:
        NDCG@k.
    rng_seed:
        Deterministic user sampling.
    tie_tolerance:
        Cells whose NDCG falls within this delta of the best are
        treated as tied. When the top is fully tied, fall back to
        the default config to avoid arbitrary picks.
    sparse_data_threshold:
        When the average events-per-user is below this number, the
        calibrator caps boost_multiplier at ``sparse_data_boost_ceiling``
        to avoid amplifying noise. The growth-curve probe showed that
        on sparse data (5% grocery: 10 events/user) the calibrator
        otherwise picks aggressive boosts and loses to cooc-alone by
        5-6% NDCG. Default 20.0 events/user.
    sparse_data_boost_ceiling:
        Maximum boost_multiplier allowed when sparse-data threshold
        triggers. Default 3.0.
    min_lift_over_cooc_only:
        Minimum NDCG lift the best cell must have over cooc-only
        ranking on the held-out slice. When no cell beats cooc-only
        by this margin, the calibrator falls back to a degenerate
        config (boost_multiplier=0) that effectively disables
        boosting and returns plain cooc ranking. This protects
        against scenarios where the layered approach can't beat
        cooc but the calibrator's grid sweep still picks SOMETHING.
    """
    from kindling.benchmarks.metrics import aggregate
    from kindling.benchmarks.probe_layered import (
        _cooc_scores,
        _path_basket_scores,
        _session_cooc_scores,
        _temporal_cooc_scores,
    )
    from kindling.retrieve.cooccurrence import CoOccurrenceRetriever

    t0 = time.perf_counter()

    if engine._item_graph is None:
        # Engine not fitted; return defaults.
        return CalibrationResult(
            best_config=LayeredConfig(),
            fallback_to_default=True,
            elapsed_seconds=0.0,
        )

    rng = np.random.default_rng(rng_seed)

    # Pick eligible users: must have >= min_user_interactions in their
    # training history.
    eligible_users: list[object] = []
    for entity, owned_arr in engine._owned_by_entity.items():
        if owned_arr.size >= min_user_interactions:
            eligible_users.append(entity)
    if not eligible_users:
        return CalibrationResult(
            best_config=LayeredConfig(),
            fallback_to_default=True,
            elapsed_seconds=time.perf_counter() - t0,
        )

    # Deterministic sample.
    if len(eligible_users) > n_users:
        sampled_idx = rng.choice(len(eligible_users), size=n_users, replace=False)
        users = [eligible_users[int(i)] for i in sampled_idx]
    else:
        users = eligible_users

    cooc_retriever = CoOccurrenceRetriever(engine._item_graph)

    # Pre-compute per-user (cand_ids, signal_scores, held_out_set).
    # Random held-out (not most-recent) avoids over-rewarding recency-
    # correlated layers. Multiple held-out items per user reduces
    # noise vs single-LOO when cells score within ~0.5% of each other
    # on the eval set.
    cached: list[dict[str, object]] = []
    for entity in users:
        owned_full = engine._owned_by_entity.get(entity, np.array([]))
        history_full = engine._history_by_entity.get(entity, ())
        if owned_full.size < min_user_interactions or not history_full:
            continue
        # Hold out ``held_out_per_user`` distinct items.
        n_hold = min(held_out_per_user, max(1, len(history_full) - min_user_interactions + 1))
        if n_hold < 1:
            continue
        held_idx = rng.choice(len(history_full), size=n_hold, replace=False)
        held_out_items = {history_full[int(i)] for i in held_idx}
        owned_held = np.asarray(
            [it for it in owned_full.tolist() if it not in held_out_items]
        )
        if owned_held.size == 0:
            continue
        history_held = tuple(h for h in history_full if h not in held_out_items)
        # Candidates from the leave-out owned set.
        candidates = cooc_retriever.retrieve(owned_held, retrieval_budget)
        if not candidates:
            continue
        cand_ids = [c.item_id for c in candidates]
        cached.append({
            "entity": entity,
            "cand_ids": cand_ids,
            "held_out_items": held_out_items,
            "cooc": _cooc_scores(engine, cand_ids, owned_held),
            "path_basket": _path_basket_scores(engine, cand_ids, history_held),
            "session_cooc": _session_cooc_scores(engine, cand_ids, owned_held),
            "temporal_cooc": _temporal_cooc_scores(engine, cand_ids, owned_held),
        })

    if not cached:
        return CalibrationResult(
            best_config=LayeredConfig(),
            fallback_to_default=True,
            elapsed_seconds=time.perf_counter() - t0,
        )

    # Sparse-data guard: average events-per-user across the eligible
    # population. When low, cap boost_multiplier to avoid amplifying
    # noise. (Growth-curve probe on grocery showed 5% data
    # avg=10/user gave -5.6% NDCG with default boost=5.0.)
    avg_events = float(np.mean([
        engine._owned_by_entity.get(c["entity"], np.array([])).size
        for c in cached
    ]))
    if avg_events < sparse_data_threshold:
        boost_grid = tuple(b for b in boost_grid if b <= sparse_data_boost_ceiling)
        if not boost_grid:
            boost_grid = (sparse_data_boost_ceiling,)

    # Cooc-only baseline on the same held-out slice. Used to ensure
    # the chosen cell actually beats cooc-alone by a meaningful
    # margin. If not, fall back to boost_multiplier=0.
    cooc_per_entity: list[tuple[list[object], set[object]]] = []
    for c in cached:
        primary = c["cooc"]
        order = np.argsort(-primary)
        top = [c["cand_ids"][int(i)] for i in order[:k] if primary[int(i)] > 0.0]
        cooc_per_entity.append((top, c["held_out_items"]))
    cooc_baseline = aggregate(
        cooc_per_entity,
        catalog_size=engine._item_graph.n_items,
        k=k,
    )
    cooc_only_ndcg = float(cooc_baseline.ndcg_at_k)

    # Sweep grid.
    grid_results: list[dict[str, float]] = []
    for z in z_grid:
        for b in boost_grid:
            cfg = LayeredConfig(z_threshold=z, boost_multiplier=b)
            per_entity: list[tuple[list[object], set[object]]] = []
            for c in cached:
                primary = c["cooc"]
                layers = [c["path_basket"], c["session_cooc"], c["temporal_cooc"]]
                composite = layered_score(primary, layers, config=cfg)
                order = np.argsort(-composite)
                top = [c["cand_ids"][int(i)] for i in order[:k] if composite[int(i)] > 0.0]
                per_entity.append((top, c["held_out_items"]))
            metrics = aggregate(
                per_entity,
                catalog_size=engine._item_graph.n_items,
                k=k,
            )
            grid_results.append({
                "z": float(z),
                "boost": float(b),
                "ndcg": float(metrics.ndcg_at_k),
                "mrr": float(metrics.mrr),
            })

    # Pick the winner. If multiple cells tie within tolerance, prefer
    # the known-good default (z=2.5, b=3.0) if it's in the tied set;
    # otherwise tie-break to higher z (more selective) and lower
    # boost (less aggressive). The default-preference protects
    # against overfitting on noisy calibration samples.
    sorted_grid = sorted(grid_results, key=lambda r: -r["ndcg"])
    best_ndcg = sorted_grid[0]["ndcg"]

    # Lift-over-cooc-only check: if NO cell on the grid beats the
    # cooc-only baseline by min_lift_over_cooc_only, layered isn't
    # adding anything on this dataset. Return a degenerate config
    # with boost_multiplier=0 so the engine effectively does cooc-
    # only ranking. This protects sparse / no-information regimes
    # like yelp where the calibrator otherwise "wins" by random.
    if best_ndcg - cooc_only_ndcg < min_lift_over_cooc_only:
        return CalibrationResult(
            best_config=LayeredConfig(
                z_threshold=LayeredConfig().z_threshold,
                boost_multiplier=0.0,  # disable boosting
            ),
            grid_results=grid_results,
            n_users_evaluated=len(cached),
            elapsed_seconds=time.perf_counter() - t0,
            fallback_to_default=True,
        )

    tied = [r for r in sorted_grid if best_ndcg - r["ndcg"] <= tie_tolerance]
    if len(tied) == len(grid_results):
        # Fully degenerate - fall back to default.
        return CalibrationResult(
            best_config=LayeredConfig(),
            grid_results=grid_results,
            n_users_evaluated=len(cached),
            elapsed_seconds=time.perf_counter() - t0,
            fallback_to_default=True,
        )

    # Default-preference: if the post-sweep default (z=2.5, b=3.0) is
    # in the tied set, take it.
    default = LayeredConfig()
    default_in_tied = next(
        (
            r for r in tied
            if r["z"] == default.z_threshold and r["boost"] == default.boost_multiplier
        ),
        None,
    )
    if default_in_tied is not None and len(tied) > 1:
        chosen = default_in_tied
    else:
        # Tie-break: higher z first, then lower boost.
        tied_sorted = sorted(tied, key=lambda r: (-r["z"], r["boost"]))
        chosen = tied_sorted[0]
    return CalibrationResult(
        best_config=LayeredConfig(
            z_threshold=chosen["z"],
            boost_multiplier=chosen["boost"],
        ),
        grid_results=grid_results,
        n_users_evaluated=len(cached),
        elapsed_seconds=time.perf_counter() - t0,
        fallback_to_default=False,
    )
