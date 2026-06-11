"""Louvain graph-construction variant ablation.

The user-user graph that Louvain operates on is built by projecting the
user-item bipartite. Two design choices materially shift what
communities Louvain finds:

  1. **Edge weight distribution.** Raw `Σ w_u·w_v` counts produce a
     heavy-tailed distribution where popular-item-sharing pairs
     dominate. log/cosine transforms compress the dynamic range so
     long-tail signal is visible.
  2. **User population.** Hubs (very-active users) connect to almost
     everyone and force communities to merge; degenerate users (1
     interaction) only add noise. Trimming both ends sharpens the graph.

This harness compares four variants under fixed Louvain settings:

  | variant | weight_transform | edge prune | user trim         |
  |---------|------------------|------------|-------------------|
  | R       | raw              | 0%         | 0% / 0%           |
  | L       | log              | 5%         | 0% / 0%           |
  | C       | cosine           | 0%         | 0% / 0%           |
  | T       | raw              | 0%         | 5% bottom / 5% top|

Reports landing-zone: bench/reports/consolidated/louvain_variant_<loader>.{json,md}
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
class VariantReport:
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


# All variants force Louvain (so each test is isolated to graph construction).
_BASE = {"persona_method": "louvain_graph"}

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "R_raw": {
        **_BASE,
        "louvain_weight_transform": "raw",
        "louvain_min_edge_percentile": 0.0,
        "louvain_user_trim_top": 0.0,
        "louvain_user_trim_bottom": 0.0,
    },
    "L_log_prune": {
        **_BASE,
        "louvain_weight_transform": "log",
        "louvain_min_edge_percentile": 0.05,
        "louvain_user_trim_top": 0.0,
        "louvain_user_trim_bottom": 0.0,
    },
    "C_cosine": {
        **_BASE,
        "louvain_weight_transform": "cosine",
        "louvain_min_edge_percentile": 0.0,
        "louvain_user_trim_top": 0.0,
        "louvain_user_trim_bottom": 0.0,
    },
    "T_user_trim": {
        **_BASE,
        "louvain_weight_transform": "raw",
        "louvain_min_edge_percentile": 0.0,
        "louvain_user_trim_top": 0.05,
        "louvain_user_trim_bottom": 0.05,
    },
}


def run(loader: str, max_eval_users: int = 500, k: int = 10, seed: int = 0) -> VariantReport:
    from kindling.benchmarks.comparison import _load_dataset

    split = _load_dataset(loader, test_fraction=0.1)
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=max_eval_users, seed=seed)
    if not eval_set:
        raise RuntimeError("eval set empty")

    report = VariantReport(
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
        s = engine.fit_summary()
        # Pull modularity out of the profile if available.
        modularity = None
        try:
            modularity = engine._state.profile.get("louvain_modularity")  # type: ignore[union-attr]
        except Exception:
            modularity = None
        report.variants[name] = {
            "kwargs": {k_: v for k_, v in kwargs.items() if k_ != "persona_method"},
            "fit_seconds": fit_s,
            "metrics": metrics,
            "latency": latencies,
            "n_personas": s["n_personas"],
            "persona_method_used": s["persona_method_used"],
            "louvain_modularity": modularity,
        }
    return report


def render_markdown(report: VariantReport) -> str:
    names = list(VARIANT_SPECS.keys())
    headers = " | ".join(["metric"] + names)
    sep = " | ".join(["---"] + ["---:"] * len(names))
    lines = [
        f"# Louvain graph-variant ablation — {report.loader}",
        "",
        f"- users evaluated: {report.n_users_evaluated}",
        f"- train / test: {report.n_train:,} / {report.n_test:,}",
        f"- k = {report.k}",
        f"- timestamp: {report.timestamp}",
        "",
        "Variant key:",
        "",
        "| variant | weight_transform | edge prune | user trim |",
        "|---|---|---|---|",
    ]
    for n in names:
        kw = report.variants[n]["kwargs"]
        wt = kw.get("louvain_weight_transform", "raw")
        pr = kw.get("louvain_min_edge_percentile", 0.0)
        ut = (kw.get("louvain_user_trim_bottom", 0.0), kw.get("louvain_user_trim_top", 0.0))
        lines.append(
            f"| `{n}` | `{wt}` | {pr:.2%} | {ut[0]:.0%} bot / {ut[1]:.0%} top |"
        )
    lines += [
        "",
        "## Quality",
        "",
        f"| {headers} |",
        f"| {sep} |",
    ]
    for m in ("ndcg_at_k", "mrr", "recall_at_k", "hit_rate", "coverage"):
        row = [f"`{m}`"] + [
            f"{report.variants[n]['metrics'][m]:.4f}" for n in names
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Persona structure",
        "",
        f"| stage | {' | '.join(names)} |",
        f"| --- | {' | '.join(['---:'] * len(names))} |",
        "| n_personas | "
        + " | ".join(str(report.variants[n]["n_personas"]) for n in names)
        + " |",
        "| modularity | "
        + " | ".join(
            f"{report.variants[n]['louvain_modularity']:.4f}"
            if report.variants[n]["louvain_modularity"] is not None
            else "—"
            for n in names
        )
        + " |",
        "| persona_method_used | "
        + " | ".join(f"`{report.variants[n]['persona_method_used']}`" for n in names)
        + " |",
        "",
        "## Fit timing (s)",
        "",
        f"| stage | {' | '.join(names)} |",
        f"| --- | {' | '.join(['---:'] * len(names))} |",
        "| fit_s | "
        + " | ".join(f"{report.variants[n]['fit_seconds']:.2f}" for n in names)
        + " |",
        "",
        "## Recommend latency (ms)",
        "",
        f"| stage | {' | '.join(names)} |",
        f"| --- | {' | '.join(['---:'] * len(names))} |",
    ]
    for stat in ("p50_ms", "p95_ms", "p99_ms"):
        lines.append(
            f"| `{stat}` | "
            + " | ".join(f"{report.variants[n]['latency'][stat]:.2f}" for n in names)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", default="amazon-beauty")
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
