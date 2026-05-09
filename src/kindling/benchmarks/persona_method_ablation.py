"""Persona-method ablation: no personas vs SVD+HDBSCAN vs Louvain.

Three v2 configurations under the canonical (strided-500) methodology:

    A. No personas — cooc base only, no persona routing.
    B. SVD + HDBSCAN — factor-based density clustering (use_als="force_off"
       to use SVD; HDBSCAN over normalized factors).
    C. Louvain — community detection on the user-user projected graph.

A vs B isolates: do SVD-derived personas add lift over cooc-only?
A vs C isolates: do Louvain-derived personas add lift over cooc-only?
B vs C isolates: which persona method finds better communities on this data?

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
class PersonaReport:
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
    "A_no_personas": {
        # Disable personas via an unreachable threshold.
        "persona_min_users": 10_000_000,
    },
    "B_svd_hdbscan": {
        "persona_method": "hdbscan_factors",
        "use_als": "force_off",
    },
    "C_louvain": {
        "persona_method": "louvain_graph",
    },
}


def run(loader: str, max_eval_users: int = 500, k: int = 10, seed: int = 0) -> PersonaReport:
    from kindling.benchmarks.comparison import _load_dataset

    split = _load_dataset(loader, test_fraction=0.1)
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=max_eval_users, seed=seed)
    if not eval_set:
        raise RuntimeError("eval set empty")

    report = PersonaReport(
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
        st = engine._state
        s = engine.fit_summary()
        report.variants[name] = {
            "kwargs": kwargs,
            "fit_seconds": fit_s,
            "metrics": metrics,
            "latency": latencies,
            "n_personas": s["n_personas"],
            "persona_method_used": s["persona_method_used"],
            "signal_kind": s["signal_kind"],
        }
    return report


def render_markdown(report: PersonaReport) -> str:
    lines = [
        f"# Persona-method ablation — {report.loader}",
        "",
        f"- users evaluated: {report.n_users_evaluated}",
        f"- train / test: {report.n_train:,} / {report.n_test:,}",
        f"- k = {report.k}",
        f"- timestamp: {report.timestamp}",
        f"- signal_kind: {report.variants['A_no_personas']['signal_kind']}",
        "",
        "## Quality",
        "",
        "| metric | A: no personas | B: SVD+HDBSCAN | C: Louvain |",
        "|---|---:|---:|---:|",
    ]
    for m in ("ndcg_at_k", "mrr", "recall_at_k", "hit_rate", "coverage"):
        a = report.variants["A_no_personas"]["metrics"][m]
        b = report.variants["B_svd_hdbscan"]["metrics"][m]
        c = report.variants["C_louvain"]["metrics"][m]
        lines.append(f"| `{m}` | {a:.4f} | {b:.4f} | {c:.4f} |")
    lines.extend([
        "",
        "## Persona structure",
        "",
        "| stage | A | B | C |",
        "|---|---:|---:|---:|",
        f"| n_personas | {report.variants['A_no_personas']['n_personas']} | "
        f"{report.variants['B_svd_hdbscan']['n_personas']} | "
        f"{report.variants['C_louvain']['n_personas']} |",
        f"| persona_method_used | `{report.variants['A_no_personas']['persona_method_used']}` | "
        f"`{report.variants['B_svd_hdbscan']['persona_method_used']}` | "
        f"`{report.variants['C_louvain']['persona_method_used']}` |",
        "",
        "## Fit timing",
        "",
        "| stage | A | B | C |",
        "|---|---:|---:|---:|",
        f"| fit_s | {report.variants['A_no_personas']['fit_seconds']:.2f} | "
        f"{report.variants['B_svd_hdbscan']['fit_seconds']:.2f} | "
        f"{report.variants['C_louvain']['fit_seconds']:.2f} |",
        "",
        "## Recommend latency (ms)",
        "",
        "| stage | A | B | C |",
        "|---|---:|---:|---:|",
    ])
    for stat in ("p50_ms", "p95_ms", "p99_ms"):
        a = report.variants["A_no_personas"]["latency"][stat]
        b = report.variants["B_svd_hdbscan"]["latency"][stat]
        c = report.variants["C_louvain"]["latency"][stat]
        lines.append(f"| `{stat}` | {a:.2f} | {b:.2f} | {c:.2f} |")
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
