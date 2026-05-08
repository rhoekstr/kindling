"""Graph-MF ablation: cooc base vs graph_mf base vs graph_mf boost.

Three v2 configurations under the canonical (strided-500) methodology:

    A. cooc base (default), no graph_mf
    B. graph_mf as base (replaces cooc base)
    C. graph_mf as boost (cooc still base, graph_mf adds dense boost)

A vs B isolates: does graph-regularized MF outperform raw cooc as the
primary scorer?
A vs C isolates: does graph_mf as a refinement layer add lift?
B vs C: which placement of graph_mf is more effective?

Reuses parity._build_eval_set for the canonical strided-500 sample.
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
class GraphMfReport:
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
    "A_cooc_base": {
        "use_graph_mf": False,
    },
    "B_graph_mf_base": {
        "use_graph_mf": True,
        "graph_mf_role": "base",
    },
    "C_graph_mf_boost": {
        "use_graph_mf": True,
        "graph_mf_role": "boost",
    },
}


def run(loader: str, max_eval_users: int = 500, k: int = 10, seed: int = 0) -> GraphMfReport:
    from kindling.benchmarks.comparison import _load_dataset

    split = _load_dataset(loader, test_fraction=0.1)
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=max_eval_users, seed=seed)
    if not eval_set:
        raise RuntimeError("eval set empty")

    report = GraphMfReport(
        loader=loader,
        n_train=len(train),
        n_test=len(test),
        n_users_evaluated=len(eval_set),
        k=k,
    )
    for name, kwargs in VARIANT_SPECS.items():
        engine = EngineV2(retrieval_budget=500, random_state=seed, **kwargs)
        t0 = time.perf_counter()
        engine.fit(train, item_metadata=split.items)
        fit_s = time.perf_counter() - t0
        metrics, latencies = _evaluate(engine, eval_set, k=k)
        st = engine._state
        report.variants[name] = {
            "kwargs": kwargs,
            "fit_seconds": fit_s,
            "metrics": metrics,
            "latency": latencies,
            "graph_mf_active": st.gmf_user_factors is not None if st else False,
            "graph_mf_data_graph_kind": st.gmf_data_graph_kind if st else "none",
        }
    return report


def render_markdown(report: GraphMfReport) -> str:
    lines = [
        f"# Graph-MF ablation — {report.loader}",
        "",
        f"- users evaluated: {report.n_users_evaluated}",
        f"- train / test: {report.n_train:,} / {report.n_test:,}",
        f"- k = {report.k}",
        f"- timestamp: {report.timestamp}",
        "",
        "## Quality",
        "",
        "| metric | A: cooc base | B: graph_mf base | C: graph_mf boost |",
        "|---|---:|---:|---:|",
    ]
    for m in ("ndcg_at_k", "mrr", "recall_at_k", "hit_rate", "coverage"):
        a = report.variants["A_cooc_base"]["metrics"][m]
        b = report.variants["B_graph_mf_base"]["metrics"][m]
        c = report.variants["C_graph_mf_boost"]["metrics"][m]
        lines.append(f"| `{m}` | {a:.4f} | {b:.4f} | {c:.4f} |")
    lines.extend([
        "",
        "## Fit timing",
        "",
        "| stage | A | B | C |",
        "|---|---:|---:|---:|",
        f"| fit_s | {report.variants['A_cooc_base']['fit_seconds']:.2f} | "
        f"{report.variants['B_graph_mf_base']['fit_seconds']:.2f} | "
        f"{report.variants['C_graph_mf_boost']['fit_seconds']:.2f} |",
        "",
        "## Recommend latency (ms)",
        "",
        "| stage | A | B | C |",
        "|---|---:|---:|---:|",
    ])
    for stat in ("p50_ms", "p95_ms", "p99_ms"):
        a = report.variants["A_cooc_base"]["latency"][stat]
        b = report.variants["B_graph_mf_base"]["latency"][stat]
        c = report.variants["C_graph_mf_boost"]["latency"][stat]
        lines.append(f"| `{stat}` | {a:.2f} | {b:.2f} | {c:.2f} |")
    lines.extend([
        "",
        "## State",
        "",
        f"- B graph kind: `{report.variants['B_graph_mf_base']['graph_mf_data_graph_kind']}`",
        f"- C graph kind: `{report.variants['C_graph_mf_boost']['graph_mf_data_graph_kind']}`",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", default="movielens-1m")
    parser.add_argument("--max-eval-users", type=int, default=500)
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
