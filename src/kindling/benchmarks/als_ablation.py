"""ALS ablation: does ALS earn its fit cost?

Three v2 configurations on the same loader split:

    A. ALS-for-HDBSCAN, no boost  (current default — full ALS, item factors discarded)
    B. SVD-for-HDBSCAN, no boost  (cheap; ALS not run at all)
    C. SVD-for-HDBSCAN, ALS-as-boost  (SVD for clustering input + ALS run for boost layer)

Comparing A vs B isolates: does ALS-quality matter for HDBSCAN clustering?
Comparing B vs C isolates: does ALS-as-boost contribute meaningful signal?
A vs C isolates: should ALS feed HDBSCAN directly, or only as a boost?

Outputs side-by-side metrics + per-stage fit timing per variant.
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

from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set
from kindling.engine_v2 import EngineV2


@dataclass
class AblationReport:
    loader: str
    n_train: int
    n_test: int
    n_users_evaluated: int
    k: int
    variants: dict[str, dict[str, Any]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


def _evaluate(engine: EngineV2, eval_set, k: int) -> tuple[dict[str, float], dict[str, float]]:
    per_entity: list[tuple[list[object], set[object]]] = []
    latencies_ms: list[float] = []
    for entity, relevant in eval_set.items():
        s = time.perf_counter()
        recs = engine.recommend(entity_id=entity, n=k)
        latencies_ms.append((time.perf_counter() - s) * 1000.0)
        per_entity.append(([r.item_id for r in recs], relevant))
    catalog_size = engine._state.n_items if engine._state else 1
    rep = aggregate(per_entity, catalog_size=max(catalog_size, 1), k=k)
    metrics = {
        "ndcg_at_k": rep.ndcg_at_k,
        "recall_at_k": rep.recall_at_k,
        "mrr": rep.mrr,
        "hit_rate": rep.hit_rate,
        "coverage": rep.coverage,
    }
    arr = np.asarray(latencies_ms) if latencies_ms else np.array([0.0])
    timing = {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }
    return metrics, timing


VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "A_als_hdbscan": {
        "hdbscan_factor_method": "als",
        "als_as_boost": False,
    },
    "B_svd_hdbscan": {
        "hdbscan_factor_method": "svd",
        "als_as_boost": False,
    },
    "C_svd_plus_als_boost": {
        "hdbscan_factor_method": "svd",
        "als_as_boost": True,
    },
}


def run(loader: str, max_eval_users: int = 200, k: int = 10, seed: int = 0) -> AblationReport:
    from kindling.benchmarks.comparison import _load_dataset

    split = _load_dataset(loader, test_fraction=0.1)
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=max_eval_users, seed=seed)
    if not eval_set:
        raise RuntimeError("eval set empty")

    report = AblationReport(
        loader=loader,
        n_train=len(train),
        n_test=len(test),
        n_users_evaluated=len(eval_set),
        k=k,
    )
    for name, kwargs in VARIANT_SPECS.items():
        engine = EngineV2(retrieval_budget=500, random_state=seed, **kwargs)
        t0 = time.perf_counter()
        engine.fit(train)
        fit_s = time.perf_counter() - t0
        metrics, latencies = _evaluate(engine, eval_set, k=k)
        report.variants[name] = {
            "kwargs": kwargs,
            "fit_seconds": fit_s,
            "metrics": metrics,
            "latency": latencies,
            "personas_actual": engine._state.n_personas if engine._state else 0,
        }
    return report


def render_markdown(report: AblationReport) -> str:
    lines = [
        f"# ALS ablation — {report.loader}",
        "",
        f"- users evaluated: {report.n_users_evaluated}",
        f"- train / test: {report.n_train:,} / {report.n_test:,}",
        f"- k = {report.k}",
        f"- timestamp: {report.timestamp}",
        "",
        "## Quality",
        "",
        "| metric | A: ALS→HDBSCAN | B: SVD→HDBSCAN | C: SVD→HDBSCAN + ALS-boost |",
        "|---|---:|---:|---:|",
    ]
    for m in ("ndcg_at_k", "mrr", "recall_at_k", "hit_rate", "coverage"):
        a = report.variants["A_als_hdbscan"]["metrics"][m]
        b = report.variants["B_svd_hdbscan"]["metrics"][m]
        c = report.variants["C_svd_plus_als_boost"]["metrics"][m]
        lines.append(f"| `{m}` | {a:.4f} | {b:.4f} | {c:.4f} |")

    lines.extend([
        "",
        "## Fit timing + persona count",
        "",
        "| stage | A | B | C |",
        "|---|---:|---:|---:|",
        f"| fit_s | {report.variants['A_als_hdbscan']['fit_seconds']:.2f} | "
        f"{report.variants['B_svd_hdbscan']['fit_seconds']:.2f} | "
        f"{report.variants['C_svd_plus_als_boost']['fit_seconds']:.2f} |",
        f"| personas_found | {report.variants['A_als_hdbscan']['personas_actual']} | "
        f"{report.variants['B_svd_hdbscan']['personas_actual']} | "
        f"{report.variants['C_svd_plus_als_boost']['personas_actual']} |",
    ])

    lines.extend([
        "",
        "## Recommend latency (ms)",
        "",
        "| stage | A | B | C |",
        "|---|---:|---:|---:|",
    ])
    for stat in ("p50_ms", "p95_ms", "p99_ms"):
        a = report.variants["A_als_hdbscan"]["latency"][stat]
        b = report.variants["B_svd_hdbscan"]["latency"][stat]
        c = report.variants["C_svd_plus_als_boost"]["latency"][stat]
        lines.append(f"| `{stat}` | {a:.2f} | {b:.2f} | {c:.2f} |")

    lines.extend([
        "",
        "## Reading guide",
        "",
        "- **A vs B**: does ALS-quality matter for clustering input?",
        "- **B vs C**: does ALS-as-boost contribute lift?",
        "- **A vs C**: should ALS feed HDBSCAN, or only the boost layer?",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", default="movielens-1m")
    parser.add_argument("--max-eval-users", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run(args.loader, max_eval_users=args.max_eval_users, k=args.k, seed=args.seed)
    payload = json.dumps(asdict(report), indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
        print(f"wrote {args.output}")
    else:
        print(payload)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report))
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
