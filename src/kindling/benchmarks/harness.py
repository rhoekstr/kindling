"""Benchmark harness — trains an Engine on train data, evaluates on held-out
test data, and emits a metric report.

CLI:
    python -m kindling.benchmarks.harness --dataset movielens-1m
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from kindling import __version__
from kindling.benchmarks.metrics import MetricReport, aggregate
from kindling.engine import Engine
from kindling.loaders import movielens


@dataclass(frozen=True)
class BenchRun:
    dataset: str
    fit_seconds: float
    recommend_seconds: float
    metrics: MetricReport
    engine_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "fit_seconds": self.fit_seconds,
            "recommend_seconds": self.recommend_seconds,
            "engine_version": self.engine_version,
            "metrics": self.metrics.as_dict(),
        }


def run_movielens_1m(
    k: int = 10,
    max_eval_entities: int | None = 2000,
    test_fraction: float = 0.1,
) -> BenchRun:
    """Load ML-1M, fit the engine, evaluate on the chronological tail."""
    split = movielens.load_1m(test_fraction=test_fraction)
    engine = Engine()

    fit_start = time.perf_counter()
    engine.fit(split.train)
    fit_seconds = time.perf_counter() - fit_start

    # Per-entity ground truth = items the entity interacted with in the test
    # window that they did NOT already have in train. Standard setup for
    # chronological split recommender evaluation.
    train_items_by_entity = split.train.groupby("entity_id", sort=False)["item_id"].apply(
        lambda s: set(s.tolist())
    )
    test_items_by_entity = split.test.groupby("entity_id", sort=False)["item_id"].apply(
        lambda s: set(s.tolist())
    )

    eval_entities = sorted(
        set(train_items_by_entity.index).intersection(test_items_by_entity.index)
    )
    if max_eval_entities is not None and len(eval_entities) > max_eval_entities:
        # Deterministic subsample for bench stability.
        step = len(eval_entities) // max_eval_entities
        eval_entities = eval_entities[::step][:max_eval_entities]

    per_entity: list[tuple[list[object], set[object]]] = []
    rec_start = time.perf_counter()
    for entity in eval_entities:
        train_owned = train_items_by_entity.get(entity, set())
        test_owned = test_items_by_entity.get(entity, set())
        relevant = test_owned - train_owned
        recs = engine.recommend(entity_id=entity, n=k)
        rec_items = [r.item_id for r in recs]
        per_entity.append((rec_items, relevant))
    recommend_seconds = time.perf_counter() - rec_start

    metrics = aggregate(
        per_entity,
        catalog_size=engine.item_graph.n_items,
        k=k,
    )

    return BenchRun(
        dataset="movielens-1m",
        fit_seconds=fit_seconds,
        recommend_seconds=recommend_seconds,
        metrics=metrics,
        engine_version=__version__,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run kindling's benchmark harness on a reference dataset."
    )
    parser.add_argument(
        "--dataset",
        default="movielens-1m",
        choices=["movielens-1m"],
        help="Reference dataset to benchmark against",
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--max-eval-entities",
        type=int,
        default=2000,
        help="Cap number of evaluated entities for wall-time control. Set to -1 to evaluate all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write a JSON report. Prints to stdout if unset.",
    )
    args = parser.parse_args(argv)

    max_entities = None if args.max_eval_entities < 0 else args.max_eval_entities
    if args.dataset == "movielens-1m":
        result = run_movielens_1m(k=args.k, max_eval_entities=max_entities)
    else:
        raise SystemExit(f"Unknown dataset: {args.dataset}")

    report = result.as_dict()
    pretty = json.dumps(report, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty + "\n")
        print(f"Wrote {args.output}")
    else:
        print(pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
