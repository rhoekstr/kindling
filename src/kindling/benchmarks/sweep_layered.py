"""Parameter sweep for the layered (cooc + adaptive boosting) scorer.

Validates the (z_threshold=2.0, boost_multiplier=3.0) defaults
empirically. Also surfaces meaningfulness diagnostics per layer per
dataset so we can decide which refinement signals belong on which
dataset shapes.

For one dataset:

1. Fit engine once.
2. Compute per-(entity, candidate) signal scores once: cooc baseline,
   path_basket, session_cooccurrence, temporal_cooccurrence.
3. For each (z_threshold, boost_multiplier) cell in the grid, score
   the same candidate pool with the layered_cooc+stack variant and
   record NDCG / Recall / fire-rates.
4. Output: a 2D grid of NDCG values per (z, boost) plus per-layer
   meaningfulness diagnostics.

CLI:
    python -m kindling.benchmarks.sweep_layered \\
        --dataset synthetic-grocery-deep \\
        --output bench/reports/sweep_layered_grocery.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from kindling import Engine, __version__
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.blend.layer_scoring import (
    _cooc_scores,
    _path_basket_scores,
    _session_cooc_scores,
    _temporal_cooc_scores,
)
from kindling.blend.layered import (
    LayeredConfig,
    diagnostic_report,
    is_layer_meaningful,
    layered_score,
)
from kindling.retrieve.cooccurrence import CoOccurrenceRetriever


def _evaluate_grid_cell(
    engine: Engine,
    eval_entities: list[object],
    train_items: pd.Series,
    test_items: pd.Series,
    cached_scores: dict[object, dict[str, np.ndarray]],
    config: LayeredConfig,
    cooc_retriever: CoOccurrenceRetriever,
    catalog_size: int,
    retrieval_budget: int,
    k: int,
) -> tuple[float, float, float, dict[str, float]]:
    """Run the layered+stack variant for one (z_threshold, boost) point.

    Reuses precomputed signal scores per (entity, candidate_pool) so
    the sweep doesn't refit the engine or re-score signals K times.
    """
    per_entity: list[tuple[list[object], set[object]]] = []
    fire_accum: dict[str, list[float]] = {}
    n_with_relevant = 0
    recall_topk_hits = 0

    for entity in eval_entities:
        scores_for_entity = cached_scores.get(entity)
        if scores_for_entity is None:
            per_entity.append(([], set()))
            continue
        primary = scores_for_entity["cooc"]
        layers = [
            scores_for_entity["path_basket"],
            scores_for_entity["session_cooc"],
            scores_for_entity["temporal_cooc"],
        ]
        cand_ids = scores_for_entity["cand_ids"]

        rep = diagnostic_report(
            primary,
            {
                "path_basket": layers[0],
                "session_cooc": layers[1],
                "temporal_cooc": layers[2],
            },
            config=config,
        )
        for layer_name in ("path_basket", "session_cooc", "temporal_cooc"):
            if layer_name in rep["layers"] and not rep["layers"][layer_name].get("would_skip", False):
                fire_accum.setdefault(f"{layer_name}_fire_rate", []).append(
                    rep["layers"][layer_name]["fire_rate"]
                )

        composite = layered_score(primary, layers, config=config)
        order = np.argsort(-composite)
        top = [cand_ids[int(i)] for i in order[:k] if composite[int(i)] > 0.0]

        train_owned = train_items.get(entity, set())
        test_owned = test_items.get(entity, set())
        relevant = test_owned - train_owned
        per_entity.append((top, relevant))
        if relevant:
            n_with_relevant += 1
            if set(top) & relevant:
                recall_topk_hits += 1

    metrics = aggregate(per_entity, catalog_size=catalog_size, k=k)
    avg_fire = {k: float(np.mean(v)) for k, v in fire_accum.items() if v}
    return (
        metrics.ndcg_at_k,
        metrics.mrr,
        recall_topk_hits / max(n_with_relevant, 1),
        avg_fire,
    )


def _precompute_scores(
    engine: Engine,
    cooc_retriever: CoOccurrenceRetriever,
    eval_entities: list[object],
    retrieval_budget: int,
) -> dict[object, dict[str, np.ndarray]]:
    """Cache per-(entity, candidate_pool) signal scores so the sweep
    can iterate parameters without re-scoring."""
    cached: dict[object, dict[str, np.ndarray]] = {}
    for entity in eval_entities:
        owned = engine._owned_by_entity.get(entity, np.array([]))
        history = engine._history_by_entity.get(entity, ())
        candidates = cooc_retriever.retrieve(owned, retrieval_budget)
        if not candidates:
            continue
        cand_ids = [c.item_id for c in candidates]
        cached[entity] = {
            "cand_ids": cand_ids,
            "cooc": _cooc_scores(engine, cand_ids, owned),
            "path_basket": _path_basket_scores(engine, cand_ids, history),
            "session_cooc": _session_cooc_scores(engine, cand_ids, owned),
            "temporal_cooc": _temporal_cooc_scores(engine, cand_ids, owned),
        }
    return cached


def run_sweep(
    dataset: str,
    z_thresholds: list[float] | None = None,
    boost_multipliers: list[float] | None = None,
    max_eval_entities: int = 500,
    retrieval_budget: int = 500,
    k: int = 10,
    test_fraction: float = 0.1,
    skip_heavy_signals: bool = False,
) -> dict[str, object]:
    z_thresholds = z_thresholds or [1.5, 2.0, 2.5, 3.0]
    boost_multipliers = boost_multipliers or [1.0, 2.0, 3.0, 5.0]

    split = _load_dataset(dataset, test_fraction=test_fraction)
    print(f"\n=== sweep_layered: {dataset} ({len(split.train):,} train) ===", flush=True)

    test_items = cast(
        pd.Series,
        split.test.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )
    train_items = cast(
        pd.Series,
        split.train.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )
    eval_entities_all = sorted(
        set(train_items.index).intersection(test_items.index)
    )
    step = max(1, len(eval_entities_all) // max_eval_entities)
    eval_entities = eval_entities_all[::step][:max_eval_entities]
    catalog_size = int(split.train["item_id"].nunique())

    print(f"  fitting engine ...", flush=True)
    t0 = time.perf_counter()
    engine_kwargs: dict[str, object] = {}
    if not skip_heavy_signals:
        from kindling.personas import KMeansClustering, PersonaConfig
        engine_kwargs["persona_config"] = PersonaConfig(
            enabled=True,
            clustering=KMeansClustering(n_clusters=30, random_state=0),
            min_activation_users=100,
        )
    engine = Engine(**engine_kwargs).fit(split.train)
    print(f"  engine fit: {time.perf_counter() - t0:.1f}s", flush=True)

    cooc_retriever = CoOccurrenceRetriever(engine._item_graph)
    print(f"  pre-computing per-entity signal scores ...", flush=True)
    t0 = time.perf_counter()
    cached_scores = _precompute_scores(
        engine, cooc_retriever, eval_entities, retrieval_budget,
    )
    print(f"  precompute: {time.perf_counter() - t0:.1f}s "
          f"({len(cached_scores)} entities cached)", flush=True)

    # --- Per-layer meaningfulness diagnostics on a sample of entities ---
    layer_meaningfulness: dict[str, dict[str, int]] = {
        "path_basket": {"ok": 0, "skip": 0},
        "session_cooc": {"ok": 0, "skip": 0},
        "temporal_cooc": {"ok": 0, "skip": 0},
    }
    sample_size = min(100, len(cached_scores))
    sample_entities = list(cached_scores.keys())[:sample_size]
    for ent in sample_entities:
        s = cached_scores[ent]
        for layer_name in ("path_basket", "session_cooc", "temporal_cooc"):
            ok, _ = is_layer_meaningful(s[layer_name], s["cooc"])
            layer_meaningfulness[layer_name]["ok" if ok else "skip"] += 1

    # --- Run grid sweep ---
    grid: list[dict[str, object]] = []
    print(f"  sweeping z_threshold x boost_multiplier ...", flush=True)
    for z in z_thresholds:
        for b in boost_multipliers:
            cfg = LayeredConfig(z_threshold=z, boost_multiplier=b)
            ndcg, mrr, recall_topk, fire_rates = _evaluate_grid_cell(
                engine, eval_entities, train_items, test_items,
                cached_scores, cfg, cooc_retriever, catalog_size,
                retrieval_budget, k,
            )
            grid.append({
                "z_threshold": z,
                "boost_multiplier": b,
                "ndcg_at_k": ndcg,
                "mrr": mrr,
                "recall_topk": recall_topk,
                "fire_rates": fire_rates,
            })
            print(
                f"    z={z:.1f} b={b:.1f}  NDCG={ndcg:.4f} R@K={recall_topk:.3f} "
                f"MRR={mrr:.3f}  fire={fire_rates}",
                flush=True,
            )

    # --- Cooc-alone baseline for the sweep grid ---
    cooc_per_entity: list[tuple[list[object], set[object]]] = []
    for entity in eval_entities:
        scores_for_entity = cached_scores.get(entity)
        if scores_for_entity is None:
            cooc_per_entity.append(([], set()))
            continue
        cand_ids = scores_for_entity["cand_ids"]
        primary = scores_for_entity["cooc"]
        order = np.argsort(-primary)
        top = [cand_ids[int(i)] for i in order[:k] if primary[int(i)] > 0.0]
        train_owned = train_items.get(entity, set())
        test_owned = test_items.get(entity, set())
        cooc_per_entity.append((top, test_owned - train_owned))
    cooc_metrics = aggregate(cooc_per_entity, catalog_size=catalog_size, k=k)

    # Find best grid cell.
    best = max(grid, key=lambda r: r["ndcg_at_k"])

    return {
        "dataset": dataset,
        "k": k,
        "n_eval_entities": len(eval_entities),
        "kindling_version": __version__,
        "cooc_alone_ndcg": cooc_metrics.ndcg_at_k,
        "cooc_alone_mrr": cooc_metrics.mrr,
        "best_cell": best,
        "delta_vs_cooc": best["ndcg_at_k"] - cooc_metrics.ndcg_at_k,
        "default_cell": next(
            r for r in grid
            if r["z_threshold"] == 2.0 and r["boost_multiplier"] == 3.0
        ) if any(
            r["z_threshold"] == 2.0 and r["boost_multiplier"] == 3.0
            for r in grid
        ) else None,
        "layer_meaningfulness_sample": layer_meaningfulness,
        "grid": grid,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep z_threshold x boost_multiplier on the layered scorer."
    )
    parser.add_argument(
        "--dataset",
        default="synthetic-grocery-deep",
        choices=[
            "movielens-1m", "synthetic-grocery", "synthetic-grocery-deep",
            "retailrocket", "instacart", "gowalla", "yelp2018",
            "tafeng", "dunnhumby", "amazon-beauty", "amazon-book",
        ],
    )
    parser.add_argument("--max-eval-entities", type=int, default=500)
    parser.add_argument("--retrieval-budget", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--z-thresholds", default="1.5,2.0,2.5,3.0",
        help="Comma-separated list of z_threshold values to sweep.",
    )
    parser.add_argument(
        "--boost-multipliers", default="1.0,2.0,3.0,5.0",
        help="Comma-separated list of boost_multiplier values to sweep.",
    )
    parser.add_argument("--skip-heavy-signals", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    z_list = [float(x) for x in args.z_thresholds.split(",") if x.strip()]
    b_list = [float(x) for x in args.boost_multipliers.split(",") if x.strip()]

    report = run_sweep(
        dataset=args.dataset,
        z_thresholds=z_list,
        boost_multipliers=b_list,
        max_eval_entities=args.max_eval_entities,
        retrieval_budget=args.retrieval_budget,
        k=args.k,
        skip_heavy_signals=args.skip_heavy_signals,
    )
    pretty = json.dumps(report, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty + "\n")
        print(f"\nWrote {args.output}")
    else:
        print(pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
