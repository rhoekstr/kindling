"""Growth-curve benchmark: how do accuracy and latency behave as the
training set grows from sparse to dense?

For each fraction in the sweep, we take the chronological prefix of the
training data (the first ``fraction * n_train`` interactions), train
every model, and evaluate on the same held-out test window. The
evaluation entities are held constant across fractions - entities that
exist at the smallest fraction. This isolates the effect of data volume
from the effect of evaluation population changes.

Kindling's design thesis: calibrated-uncertainty + data-adaptive priors
should make it degrade more gracefully than baselines as data thins
out. This harness is the honest empirical test of that thesis.

CLI:
    python -m kindling.benchmarks.growth_curve \\
        --dataset movielens-1m \\
        --fractions 0.1,0.25,0.5,0.75,1.0 \\
        --output bench/reports/growth_curve_movielens.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from kindling import __version__
from kindling.benchmarks.baselines import (
    ImplicitALSBaseline,
    ItemItemKNN,
    PopularityBaseline,
)
from kindling.benchmarks.comparison import Recommender, _EngineAdapter, _load_dataset
from kindling.benchmarks.metrics import MetricReport, aggregate


@dataclass(frozen=True)
class FractionResult:
    fraction: float
    n_train_interactions: int
    n_train_entities: int
    n_train_items: int
    model_name: str
    fit_seconds: float
    recommend_p50_ms: float
    recommend_p95_ms: float
    metrics: MetricReport

    def as_dict(self) -> dict[str, object]:
        return {
            "fraction": self.fraction,
            "n_train_interactions": self.n_train_interactions,
            "n_train_entities": self.n_train_entities,
            "n_train_items": self.n_train_items,
            "model_name": self.model_name,
            "fit_seconds": self.fit_seconds,
            "recommend_p50_ms": self.recommend_p50_ms,
            "recommend_p95_ms": self.recommend_p95_ms,
            "metrics": self.metrics.as_dict(),
        }


def _chronological_prefix(train: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Return the first ``fraction`` of train interactions by time order.

    Train is already chronological from the loader, so this is a slice.
    """
    n = int(round(len(train) * fraction))
    return train.iloc[:n].reset_index(drop=True)


def _build_models(include_als: bool) -> list[Recommender]:
    models: list[Recommender] = [
        _EngineAdapter(),
        PopularityBaseline(),
        ItemItemKNN(k_neighbors=200),
    ]
    if include_als:
        models.append(ImplicitALSBaseline(factors=64, iterations=15))
    return models


def _evaluate_at_fraction(
    model: Recommender,
    train_subset: pd.DataFrame,
    eval_entities: list[object],
    test_items_by_entity: pd.Series,
    train_items_by_entity: pd.Series,
    catalog_size: int,
    k: int,
) -> tuple[float, float, float, MetricReport]:
    fit_start = time.perf_counter()
    model.fit(train_subset)
    fit_seconds = time.perf_counter() - fit_start

    per_entity: list[tuple[list[object], set[object]]] = []
    latencies_ms: list[float] = []
    for entity in eval_entities:
        train_owned = train_items_by_entity.get(entity, set())
        test_owned = test_items_by_entity.get(entity, set())
        relevant = test_owned - train_owned
        t0 = time.perf_counter()
        rec_items = model.recommend(entity, n=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        per_entity.append((rec_items, relevant))

    metrics = aggregate(per_entity, catalog_size=catalog_size, k=k)
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    return fit_seconds, p50, p95, metrics


def run_growth_curve(
    fractions: list[float],
    dataset: str = "movielens-1m",
    k: int = 10,
    max_eval_entities: int = 1000,
    test_fraction: float = 0.1,
    include_als: bool = True,
) -> dict[str, object]:
    split = _load_dataset(dataset, test_fraction)

    train_items_by_entity = cast(
        pd.Series,
        split.train.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )
    test_items_by_entity = cast(
        pd.Series,
        split.test.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )

    # Eval entities = intersection of (train_has_entity) and (test_has_entity),
    # fixed across all fractions. For the smallest fraction an entity may not
    # appear in the subsample - those entities still count toward metrics (the
    # model returns an empty list, which hits 0 on accuracy metrics).
    eval_entities_all: list[object] = sorted(
        set(train_items_by_entity.index).intersection(test_items_by_entity.index)
    )
    step = max(1, len(eval_entities_all) // max_eval_entities)
    eval_entities: list[object] = eval_entities_all[::step][:max_eval_entities]

    results: list[FractionResult] = []
    for frac in fractions:
        print(f"\n=== fraction={frac:.2f} ===", flush=True)
        train_subset = _chronological_prefix(split.train, frac)
        n_train_entities = int(train_subset["entity_id"].nunique())
        n_train_items = int(train_subset["item_id"].nunique())
        print(
            f"  train: {len(train_subset):,} interactions, "
            f"{n_train_entities:,} entities, {n_train_items:,} items",
            flush=True,
        )
        # Build fresh models per fraction (no warm-start across fractions).
        for model in _build_models(include_als):
            print(f"  {model.name}...", flush=True)
            fit_s, p50, p95, metrics = _evaluate_at_fraction(
                model,
                train_subset=train_subset,
                eval_entities=eval_entities,
                test_items_by_entity=test_items_by_entity,
                train_items_by_entity=train_items_by_entity,
                catalog_size=n_train_items,
                k=k,
            )
            results.append(
                FractionResult(
                    fraction=frac,
                    n_train_interactions=len(train_subset),
                    n_train_entities=n_train_entities,
                    n_train_items=n_train_items,
                    model_name=model.name,
                    fit_seconds=fit_s,
                    recommend_p50_ms=p50,
                    recommend_p95_ms=p95,
                    metrics=metrics,
                )
            )

    return {
        "dataset": dataset,
        "k": k,
        "n_eval_entities": len(eval_entities),
        "kindling_version": __version__,
        "results": [r.as_dict() for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep training-data fraction and measure accuracy + latency."
    )
    parser.add_argument(
        "--dataset",
        default="movielens-1m",
        choices=["movielens-1m", "synthetic-grocery", "synthetic-grocery-deep"],
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--fractions",
        default="0.1,0.25,0.5,0.75,1.0",
        help="Comma-separated fractions of the train set to sweep.",
    )
    parser.add_argument("--max-eval-entities", type=int, default=1000)
    parser.add_argument("--no-als", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    report = run_growth_curve(
        fractions=fractions,
        dataset=args.dataset,
        k=args.k,
        max_eval_entities=args.max_eval_entities,
        include_als=not args.no_als,
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
