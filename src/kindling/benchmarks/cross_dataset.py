"""Cross-dataset architecture benchmark (plan Phase 7 expansion).

Iterates the kindling reference + extended dataset suite (movielens-1m,
synthetic-grocery-deep, retailrocket, instacart, gowalla, yelp2018,
tafeng, dunnhumby, amazon-beauty, amazon-book) and runs the three
scoring architectures (Bayesian / gating / RRF) on each. Datasets whose
files aren't on disk are skipped with a structured ``skipped`` entry so
the report still parses.

This is the harness for the "different cooc-vs-signal-diversity profiles
will tell us when gating actually wins" follow-up work flagged in
``ADR-scoring-architecture.md``.

CLI:
    python -m kindling.benchmarks.cross_dataset \\
        --output bench/reports/cross_dataset_architecture.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import cast

import pandas as pd

from kindling import __version__
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.scoring_architecture import ArchResult, _evaluate
from kindling.engine import Engine
from kindling.gate import GatingConfig


DEFAULT_DATASETS = (
    "movielens-1m",
    "synthetic-grocery-deep",
    "retailrocket",
    "instacart",
    "gowalla",
    "yelp2018",
    "tafeng",
    "dunnhumby",
    "amazon-beauty",
    "amazon-book",
)


def _skip_record(dataset: str, reason: str) -> dict[str, object]:
    return {
        "dataset": dataset,
        "status": "skipped",
        "reason": reason,
    }


def _run_one(
    dataset: str,
    max_eval_entities: int,
    test_fraction: float,
    k: int,
    gating_epochs: int,
    skip_gating: bool,
    skip_rrf: bool,
) -> dict[str, object]:
    """Run the three architectures on a single dataset.

    Returns a status block (loaded / skipped / errored). Skipping on
    missing data is structured so the combined report still parses.
    """
    print(f"\n=== {dataset} ===", flush=True)
    try:
        split = _load_dataset(dataset, test_fraction=test_fraction)
    except Exception as exc:  # NotAvailableError or unexpected
        msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"  skipping {dataset}: {msg}", flush=True)
        return _skip_record(dataset, msg)

    train_items = cast(
        pd.Series,
        split.train.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )
    test_items = cast(
        pd.Series,
        split.test.groupby("entity_id", sort=False)["item_id"].apply(
            lambda s: set(s.tolist())
        ),
    )
    eval_entities_all: list[object] = sorted(
        set(train_items.index).intersection(test_items.index)
    )
    if not eval_entities_all:
        return _skip_record(dataset, "no entities present in both train and test")
    step = max(1, len(eval_entities_all) // max_eval_entities)
    eval_entities: list[object] = eval_entities_all[::step][:max_eval_entities]
    catalog_size = int(split.train["item_id"].nunique())

    results: list[ArchResult] = []

    # Bayesian.
    print(f"  fitting bayesian...", flush=True)
    t0 = time.perf_counter()
    engine_b = Engine().fit(split.train)
    fit_b = time.perf_counter() - t0
    metrics_b, p95_b = _evaluate(
        engine_b, "bayesian", eval_entities, train_items, test_items, catalog_size, k
    )
    results.append(
        ArchResult(
            method="bayesian",
            fit_seconds=fit_b,
            recommend_p95_ms=p95_b,
            ndcg_at_k=metrics_b.ndcg_at_k,
            recall_at_k=metrics_b.recall_at_k,
            mrr=metrics_b.mrr,
            n_eval_entities=metrics_b.n_entities_evaluated,
        )
    )

    # Gating.
    if not skip_gating:
        print(f"  fitting gating...", flush=True)
        t0 = time.perf_counter()
        gate_cfg = GatingConfig(
            enabled=True,
            n_epochs=gating_epochs,
            batch_size=512,
            min_users=100,
            seed=0,
        )
        try:
            engine_g = Engine(gating_config=gate_cfg).fit(split.train)
            fit_g = time.perf_counter() - t0
            metrics_g, p95_g = _evaluate(
                engine_g, "gating", eval_entities, train_items, test_items, catalog_size, k
            )
            results.append(
                ArchResult(
                    method="gating",
                    fit_seconds=fit_g,
                    recommend_p95_ms=p95_g,
                    ndcg_at_k=metrics_g.ndcg_at_k,
                    recall_at_k=metrics_g.recall_at_k,
                    mrr=metrics_g.mrr,
                    n_eval_entities=metrics_g.n_entities_evaluated,
                )
            )
        except Exception as exc:
            traceback.print_exc()
            print(f"  gating errored: {exc}; continuing", flush=True)

    # RRF reuses the Bayesian-fitted engine.
    if not skip_rrf:
        print(f"  evaluating rrf...", flush=True)
        try:
            metrics_r, p95_r = _evaluate(
                engine_b, "rrf", eval_entities, train_items, test_items, catalog_size, k
            )
            results.append(
                ArchResult(
                    method="rrf",
                    fit_seconds=fit_b,
                    recommend_p95_ms=p95_r,
                    ndcg_at_k=metrics_r.ndcg_at_k,
                    recall_at_k=metrics_r.recall_at_k,
                    mrr=metrics_r.mrr,
                    n_eval_entities=metrics_r.n_entities_evaluated,
                )
            )
        except Exception as exc:
            traceback.print_exc()
            print(f"  rrf errored: {exc}; continuing", flush=True)

    # Print quick summary inline.
    for r in results:
        print(
            f"  {r.method:<9} NDCG={r.ndcg_at_k:.4f} R@10={r.recall_at_k:.4f} "
            f"MRR={r.mrr:.4f} fit={r.fit_seconds:.1f}s p95={r.recommend_p95_ms:.0f}ms",
            flush=True,
        )

    return {
        "dataset": dataset,
        "status": "ok",
        "k": k,
        "n_eval_entities": len(eval_entities),
        "catalog_size": catalog_size,
        "n_train_interactions": int(len(split.train)),
        "n_test_interactions": int(len(split.test)),
        "results": [r.as_dict() for r in results],
    }


def run_cross_dataset(
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    max_eval_entities: int = 500,
    test_fraction: float = 0.1,
    k: int = 10,
    gating_epochs: int = 10,
    skip_gating: bool = False,
    skip_rrf: bool = False,
) -> dict[str, object]:
    per_dataset = []
    for ds in datasets:
        per_dataset.append(
            _run_one(
                dataset=ds,
                max_eval_entities=max_eval_entities,
                test_fraction=test_fraction,
                k=k,
                gating_epochs=gating_epochs,
                skip_gating=skip_gating,
                skip_rrf=skip_rrf,
            )
        )
    return {
        "kindling_version": __version__,
        "k": k,
        "max_eval_entities": max_eval_entities,
        "datasets": per_dataset,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-dataset architecture benchmark.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to run; missing data files are skipped gracefully.",
    )
    parser.add_argument("--max-eval-entities", type=int, default=500)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--gating-epochs", type=int, default=10)
    parser.add_argument("--skip-gating", action="store_true")
    parser.add_argument("--skip-rrf", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_cross_dataset(
        datasets=tuple(args.datasets),
        max_eval_entities=args.max_eval_entities,
        test_fraction=args.test_fraction,
        k=args.k,
        gating_epochs=args.gating_epochs,
        skip_gating=args.skip_gating,
        skip_rrf=args.skip_rrf,
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
