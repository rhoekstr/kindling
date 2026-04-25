"""End-to-end comparison: Engine default (Bayesian blend) vs Engine
with layered_scoring=True (cooc + adaptive boosting auto-calibrated).

Validates that the engine integration of the layered architecture
matches what the manual probes showed. Sanity check before promoting
layered to default.

CLI:
    python -m kindling.benchmarks.probe_engine_layered --dataset synthetic-grocery-deep
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


def _eval_engine(engine, eval_entities, train_items, test_items, k):
    per_entity = []
    latencies = []
    n_with_relevant = 0
    recall_topk_hits = 0
    for entity in eval_entities:
        t0 = time.perf_counter()
        recs = engine.recommend(entity_id=entity, n=k)
        rec_items = [r.item_id for r in recs]
        latencies.append((time.perf_counter() - t0) * 1000.0)
        train_owned = train_items.get(entity, set())
        test_owned = test_items.get(entity, set())
        relevant = test_owned - train_owned
        per_entity.append((rec_items, relevant))
        if relevant:
            n_with_relevant += 1
            if set(rec_items) & relevant:
                recall_topk_hits += 1
    m = aggregate(per_entity, catalog_size=engine._item_graph.n_items, k=k)
    return {
        "ndcg_at_k": m.ndcg_at_k,
        "mrr": m.mrr,
        "recall_topk": recall_topk_hits / max(n_with_relevant, 1),
        "p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
    }


def run(dataset: str, max_eval_entities: int = 500, k: int = 10, test_fraction: float = 0.1) -> dict:
    split = _load_dataset(dataset, test_fraction=test_fraction)
    print(f"\n=== probe_engine_layered: {dataset} ({len(split.train):,} train) ===", flush=True)

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

    print(f"  fitting Engine() (default Bayesian blend)...", flush=True)
    t0 = time.perf_counter()
    engine_b = Engine().fit(split.train)
    fit_b = time.perf_counter() - t0
    print(f"    fit {fit_b:.1f}s; evaluating ...", flush=True)
    results_b = _eval_engine(engine_b, eval_entities, train_items, test_items, k)
    print(f"    bayesian: NDCG={results_b['ndcg_at_k']:.4f} R@K={results_b['recall_topk']:.3f} "
          f"MRR={results_b['mrr']:.3f} p95={results_b['p95_ms']:.1f}ms", flush=True)

    print(f"  fitting Engine(layered_scoring=True)...", flush=True)
    t0 = time.perf_counter()
    engine_l = Engine(layered_scoring=True).fit(split.train)
    fit_l = time.perf_counter() - t0
    cal = engine_l._layered_calibration
    print(f"    fit {fit_l:.1f}s; calibrator picked z={engine_l.layered_config.z_threshold} "
          f"b={engine_l.layered_config.boost_multiplier}; evaluating ...", flush=True)
    results_l = _eval_engine(engine_l, eval_entities, train_items, test_items, k)
    print(f"    layered:  NDCG={results_l['ndcg_at_k']:.4f} R@K={results_l['recall_topk']:.3f} "
          f"MRR={results_l['mrr']:.3f} p95={results_l['p95_ms']:.1f}ms", flush=True)

    delta_ndcg = results_l["ndcg_at_k"] - results_b["ndcg_at_k"]
    rel_delta = delta_ndcg / max(results_b["ndcg_at_k"], 1e-9)
    print(f"  delta: NDCG {delta_ndcg:+.4f} ({rel_delta*100:+.2f}%)", flush=True)

    return {
        "dataset": dataset,
        "n_eval_entities": len(eval_entities),
        "kindling_version": __version__,
        "bayesian_blend": {**results_b, "fit_seconds": fit_b},
        "layered_adaptive": {
            **results_l,
            "fit_seconds": fit_l,
            "calibrated_z": engine_l.layered_config.z_threshold,
            "calibrated_boost": engine_l.layered_config.boost_multiplier,
            "calibration_seconds": engine_l._fit_timings.get("layered_calibration"),
        },
        "delta_ndcg": delta_ndcg,
        "delta_rel_pct": rel_delta * 100,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="synthetic-grocery-deep",
        choices=[
            "movielens-1m", "synthetic-grocery", "synthetic-grocery-deep",
            "retailrocket", "instacart", "gowalla", "yelp2018",
            "tafeng", "dunnhumby", "amazon-beauty", "amazon-book",
        ],
    )
    parser.add_argument("--max-eval-entities", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(args.dataset, args.max_eval_entities, args.k)
    pretty = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty + "\n")
        print(f"\nWrote {args.output}")
    else:
        print(pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
