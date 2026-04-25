"""Probe with auto-calibrated layered scorer.

Same flow as ``probe_layered`` but the (z_threshold, boost_multiplier)
config is chosen by the auto-calibrator at fit time rather than fixed
to defaults. Tests whether auto-calibration converges to the cells the
manual sweep identified per dataset.

CLI:
    python -m kindling.benchmarks.probe_layered_adaptive \\
        --dataset synthetic-grocery-deep \\
        --output bench/reports/probe_layered_adaptive_grocery.json
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
from kindling.benchmarks.probe_layered import (
    LayeredCell,
    _cooc_scores,
    _evaluate_blend_baseline,
    _path_basket_scores,
    _session_cooc_scores,
    _temporal_cooc_scores,
)
from kindling.blend.layered import LayeredConfig, layered_score
from kindling.blend.layered_calibrator import calibrate
from kindling.retrieve.cooccurrence import CoOccurrenceRetriever


def _evaluate_with_config(
    config: LayeredConfig,
    variant: str,
    eval_entities: list[object],
    train_items: pd.Series,
    test_items: pd.Series,
    engine: Engine,
    catalog_size: int,
    retrieval_budget: int,
    k: int,
    dataset: str,
) -> LayeredCell:
    cooc_retriever = CoOccurrenceRetriever(engine._item_graph)
    per_entity: list[tuple[list[object], set[object]]] = []
    latencies: list[float] = []
    recall_topk_hits = 0
    recall_budget_hits = 0
    n_with_relevant = 0

    for entity in eval_entities:
        owned = engine._owned_by_entity.get(entity, np.array([]))
        history = engine._history_by_entity.get(entity, ())
        t0 = time.perf_counter()
        candidates = cooc_retriever.retrieve(owned, retrieval_budget)
        cand_ids = [c.item_id for c in candidates]
        if not cand_ids:
            top: list[object] = []
        else:
            primary = _cooc_scores(engine, cand_ids, owned)
            layers = [
                _path_basket_scores(engine, cand_ids, history),
                _session_cooc_scores(engine, cand_ids, owned),
                _temporal_cooc_scores(engine, cand_ids, owned),
            ]
            composite = layered_score(primary, layers, config=config)
            order = np.argsort(-composite)
            top = [cand_ids[int(i)] for i in order[:k] if composite[int(i)] > 0.0]
        latencies.append((time.perf_counter() - t0) * 1000.0)

        train_owned = train_items.get(entity, set())
        test_owned = test_items.get(entity, set())
        relevant = test_owned - train_owned
        per_entity.append((top, relevant))
        if relevant:
            n_with_relevant += 1
            all_retrieved = {c.item_id for c in candidates}
            if all_retrieved & relevant:
                recall_budget_hits += 1
            if set(top) & relevant:
                recall_topk_hits += 1

    m = aggregate(per_entity, catalog_size=catalog_size, k=k)
    return LayeredCell(
        dataset=dataset,
        variant=variant,
        ndcg_at_k=m.ndcg_at_k,
        recall_topk=recall_topk_hits / max(n_with_relevant, 1),
        recall_budget=recall_budget_hits / max(n_with_relevant, 1),
        mrr=m.mrr,
        p95_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        n_eval_entities=len(eval_entities),
    )


def run_probe(
    dataset: str,
    max_eval_entities: int = 500,
    retrieval_budget: int = 500,
    k: int = 10,
    test_fraction: float = 0.1,
    skip_heavy_signals: bool = False,
    calibration_users: int = 100,
) -> dict[str, object]:
    split = _load_dataset(dataset, test_fraction=test_fraction)
    print(f"\n=== probe_layered_adaptive: {dataset} ({len(split.train):,} train) ===", flush=True)

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

    # Auto-calibrate.
    print(f"  auto-calibrating layered (z_grid x boost_grid) ...", flush=True)
    cal = calibrate(engine, n_users=calibration_users, retrieval_budget=200)
    print(
        f"  calibrator: z={cal.best_config.z_threshold} "
        f"b={cal.best_config.boost_multiplier} "
        f"({cal.elapsed_seconds:.1f}s, n_users={cal.n_users_evaluated}, "
        f"fallback={cal.fallback_to_default})",
        flush=True,
    )

    cells: list[LayeredCell] = []

    # Cooc-alone baseline.
    cell = _evaluate_with_config(
        config=LayeredConfig(z_threshold=999.0, boost_multiplier=0.0),  # no boosts
        variant="cooc_alone",
        eval_entities=eval_entities, train_items=train_items, test_items=test_items,
        engine=engine, catalog_size=catalog_size,
        retrieval_budget=retrieval_budget, k=k, dataset=dataset,
    )
    cells.append(cell)
    print(f"  cooc_alone           NDCG={cell.ndcg_at_k:.4f} R@K={cell.recall_topk:.3f}", flush=True)

    # Default-config layered.
    cell = _evaluate_with_config(
        config=LayeredConfig(),  # post-sweep defaults: z=2.5, b=3.0
        variant="layered_default",
        eval_entities=eval_entities, train_items=train_items, test_items=test_items,
        engine=engine, catalog_size=catalog_size,
        retrieval_budget=retrieval_budget, k=k, dataset=dataset,
    )
    cells.append(cell)
    print(f"  layered_default      NDCG={cell.ndcg_at_k:.4f} R@K={cell.recall_topk:.3f}", flush=True)

    # Auto-calibrated layered.
    cell = _evaluate_with_config(
        config=cal.best_config,
        variant=f"layered_adaptive(z={cal.best_config.z_threshold},b={cal.best_config.boost_multiplier})",
        eval_entities=eval_entities, train_items=train_items, test_items=test_items,
        engine=engine, catalog_size=catalog_size,
        retrieval_budget=retrieval_budget, k=k, dataset=dataset,
    )
    cells.append(cell)
    print(f"  layered_adaptive     NDCG={cell.ndcg_at_k:.4f} R@K={cell.recall_topk:.3f}", flush=True)

    # Bayesian blend baseline.
    blend_cell = _evaluate_blend_baseline(
        eval_entities, train_items, test_items, engine, catalog_size, k, dataset,
    )
    cells.append(blend_cell)
    print(f"  bayesian_blend       NDCG={blend_cell.ndcg_at_k:.4f} R@K={blend_cell.recall_topk:.3f}", flush=True)

    return {
        "dataset": dataset,
        "k": k,
        "n_eval_entities": len(eval_entities),
        "kindling_version": __version__,
        "calibration": {
            "best_z": cal.best_config.z_threshold,
            "best_boost": cal.best_config.boost_multiplier,
            "n_users_evaluated": cal.n_users_evaluated,
            "elapsed_seconds": cal.elapsed_seconds,
            "fallback_to_default": cal.fallback_to_default,
            "grid_results": cal.grid_results,
        },
        "cells": [c.as_dict() for c in cells],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe layered scoring with fit-time auto-calibration."
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
    parser.add_argument("--calibration-users", type=int, default=100)
    parser.add_argument("--skip-heavy-signals", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_probe(
        dataset=args.dataset,
        max_eval_entities=args.max_eval_entities,
        retrieval_budget=args.retrieval_budget,
        k=args.k,
        skip_heavy_signals=args.skip_heavy_signals,
        calibration_users=args.calibration_users,
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
