"""v1 ↔ v2 parity sweep — quality + perf side-by-side, per loader.

Phase 7 of the migration plan. The cutover gate (Phase 8) requires:
- v2 NDCG@10 ≥ v1 NDCG@10 on every loader within ε.
- ≥ +5% lift on at least 3 cold-start datasets (PRD acceptance gate).

This module orchestrates the comparison: fit both engines on the same
chronological train split, score the same eval users, compute aggregate
metrics, emit a JSON report.

Usage:

    python -m kindling.benchmarks.parity \\
        --loader synthetic_medium \\
        --output bench/reports/parity/<loader>.json

When the parity test fails on any loader, the report names the affected
metric so we know which boost layer or routing decision to investigate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kindling import Engine
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.perf import synthetic_dataset


@dataclass
class ParityReport:
    """v1 ↔ v2 side-by-side report."""

    loader: str
    n_train: int
    n_test: int
    n_users_evaluated: int
    k: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)  # variant → {ndcg, mrr, recall, ...}
    timing: dict[str, dict[str, float]] = field(default_factory=dict)   # variant → {fit_s, p50_ms, ...}
    deltas: dict[str, float] = field(default_factory=dict)               # metric → (v2 - v1) / v1
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


def _build_eval_set(
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_users: int = 500,
    seed: int = 0,
) -> dict[object, set[object]]:
    """For each test user with held-out items, build (entity → relevant_set).

    Filters to users present in both splits and caps at `max_users` for
    bounded eval cost.
    """
    rng = np.random.default_rng(seed)
    train_users = set(train["entity_id"].unique())
    eligible = test[test["entity_id"].isin(train_users)]
    by_user: dict[object, set[object]] = {}
    train_owned: dict[object, set[object]] = {}
    for u, g in train.groupby("entity_id"):
        train_owned[u] = set(g["item_id"].tolist())
    for u, g in eligible.groupby("entity_id"):
        held = set(g["item_id"].tolist()) - train_owned.get(u, set())
        if held:
            by_user[u] = held
    if len(by_user) > max_users:
        keys = list(by_user.keys())
        idx = rng.choice(len(keys), size=max_users, replace=False)
        by_user = {keys[int(i)]: by_user[keys[int(i)]] for i in idx}
    return by_user


def _evaluate(
    engine: Engine, eval_set: dict[object, set[object]], k: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Run engine.recommend per eval user, return (metrics, timing)."""
    per_entity: list[tuple[list[object], set[object]]] = []
    latencies_ms: list[float] = []
    for entity, relevant in eval_set.items():
        s = time.perf_counter()
        recs = engine.recommend(entity_id=entity, n=k)
        latencies_ms.append((time.perf_counter() - s) * 1000.0)
        per_entity.append(([r.item_id for r in recs], relevant))
    # Catalog size proxy: items in the engine's owned-items set.
    if engine.use_v2_core and engine._v2_engine is not None:
        catalog_size = engine._v2_engine._state.n_items if engine._v2_engine._state else 0
    else:
        catalog_size = engine._item_graph.n_items if engine._item_graph else 0
    catalog_size = max(catalog_size, 1)
    rep = aggregate(per_entity, catalog_size=catalog_size, k=k)
    metrics = {
        "ndcg_at_k": rep.ndcg_at_k,
        "recall_at_k": rep.recall_at_k,
        "precision_at_k": rep.precision_at_k,
        "mrr": rep.mrr,
        "hit_rate": rep.hit_rate,
        "coverage": rep.coverage,
    }
    arr = np.asarray(latencies_ms) if latencies_ms else np.array([0.0])
    timing = {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }
    return metrics, timing


def run(
    loader: str,
    test_fraction: float = 0.1,
    max_eval_users: int = 200,
    k: int = 10,
    seed: int = 0,
) -> ParityReport:
    """Run v1 ↔ v2 parity comparison on one loader."""
    if loader.startswith("synthetic"):
        # Build dataset + chronological-style split (random for synthetic).
        if loader == "synthetic_small":
            interactions = synthetic_dataset(200, 100, density=0.05, seed=seed)
        elif loader == "synthetic_medium":
            interactions = synthetic_dataset(2000, 500, density=0.03, seed=seed)
        elif loader == "synthetic_large":
            interactions = synthetic_dataset(10_000, 2000, density=0.02, seed=seed)
        else:
            raise ValueError(f"unknown synthetic dataset: {loader}")
        # Random per-user split (no timestamps in synthetic).
        rng = np.random.default_rng(seed)
        is_test = rng.random(len(interactions)) < test_fraction
        train = interactions[~is_test].reset_index(drop=True)
        test = interactions[is_test].reset_index(drop=True)
    else:
        # Real loader — delegate to the unified comparison harness loader.
        from kindling.benchmarks.comparison import _load_dataset

        split = _load_dataset(loader, test_fraction=test_fraction)
        train, test = split.train, split.test

    eval_set = _build_eval_set(train, test, max_users=max_eval_users, seed=seed)
    if not eval_set:
        raise RuntimeError("eval set is empty; check train/test overlap")

    report = ParityReport(
        loader=loader,
        n_train=len(train),
        n_test=len(test),
        n_users_evaluated=len(eval_set),
        k=k,
    )

    for variant in ("v1", "v2"):
        engine = Engine() if variant == "v1" else Engine(use_v2_core=True)
        t0 = time.perf_counter()
        engine.fit(train)
        fit_s = time.perf_counter() - t0
        metrics, latencies = _evaluate(engine, eval_set, k=k)
        report.metrics[variant] = metrics
        report.timing[variant] = {"fit_s": fit_s, **latencies}

    # Deltas: (v2 - v1) / v1, positive = v2 better.
    if "v1" in report.metrics and "v2" in report.metrics:
        for metric, v1_val in report.metrics["v1"].items():
            v2_val = report.metrics["v2"].get(metric, 0.0)
            if v1_val > 0:
                report.deltas[metric] = (v2_val - v1_val) / v1_val
            else:
                report.deltas[metric] = float("inf") if v2_val > 0 else 0.0
    return report


def render_markdown(report: ParityReport) -> str:
    """Side-by-side markdown rendering."""
    metrics = sorted(set(report.metrics.get("v1", {}).keys()) | set(report.metrics.get("v2", {}).keys()))
    lines = [
        f"# Parity sweep — {report.loader}",
        "",
        f"- **users evaluated**: {report.n_users_evaluated}",
        f"- **train**: {report.n_train:,}    **test**: {report.n_test:,}",
        f"- **timestamp**: {report.timestamp}",
        "",
        "## Quality (top-K = " + str(report.k) + ")",
        "",
        "| metric | v1 | v2 | Δ |",
        "|---|---:|---:|---:|",
    ]
    for m in metrics:
        v1 = report.metrics.get("v1", {}).get(m, 0.0)
        v2 = report.metrics.get("v2", {}).get(m, 0.0)
        delta = report.deltas.get(m, 0.0)
        sign = "✅" if delta >= -0.005 else "⚠️"
        lines.append(f"| `{m}` | {v1:.4f} | {v2:.4f} | {delta:+.2%} {sign} |")
    lines.extend([
        "",
        "## Timing",
        "",
        "| stage | v1 | v2 | speedup |",
        "|---|---:|---:|---:|",
    ])
    timing_keys = ["fit_s", "p50_ms", "p95_ms", "p99_ms"]
    for tk in timing_keys:
        v1 = report.timing.get("v1", {}).get(tk, 0.0)
        v2 = report.timing.get("v2", {}).get(tk, 0.0)
        speedup = v1 / v2 if v2 > 0 else float("inf")
        lines.append(f"| `{tk}` | {v1:.3f} | {v2:.3f} | {speedup:.1f}× |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", default="synthetic_medium")
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-eval-users", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run(
        args.loader,
        test_fraction=args.test_fraction,
        max_eval_users=args.max_eval_users,
        k=args.k,
        seed=args.seed,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(asdict(report), indent=2, default=str) + "\n")
        print(f"wrote {args.output}")
    else:
        print(json.dumps(asdict(report), indent=2, default=str))
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report))
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
