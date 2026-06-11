"""Clustering × coherence sweep.

Compares clustering algorithms on the **same** post-hoc cohesion metric:
mean cooc[i,j] across each persona's distinctive-item set. The score is
algorithm-agnostic — Louvain, HDBSCAN-on-SVD, HDBSCAN-on-ALS,
γ-Louvain, cosine-Louvain, etc. all produce a partition and the same
metric ranks their personas.

This isolates one question: **which clustering method finds the most
internally-coherent personas?** Quality metrics (NDCG/MRR) are reported
alongside but the primary axis is the coherence distribution per
variant.

A fixed `coherence_filter_percentile = 0.5` is applied for the quality
metrics (drop the bottom half of personas by coherence; their members
route to cooc base). Per-variant we also report:
  - mean / median / p25 / p75 / min / max coherence
  - n_personas before filter
  - n_personas kept after filter
  - quality metrics on the canonical strided-500 sample

Reports landing-zone:
  bench/reports/consolidated/clustering_coherence_<loader>.{json,md}
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
from kindling.benchmarks.persona_diff import compute_persona_diff
from kindling.engine_v2 import EngineV2


@dataclass
class SweepReport:
    loader: str
    n_train: int
    n_test: int
    n_users_evaluated: int
    k: int
    coherence_filter_percentile: float
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


# Six clustering variants. All operate on the same retrieval pool +
# canonical eval set; the only thing that changes is *how* the persona
# partition is constructed.
def _variant_specs(coherence_filter: float) -> dict[str, dict[str, Any]]:
    base = {"coherence_filter_percentile": coherence_filter}
    return {
        "hdbscan_svd": {
            **base,
            "persona_method": "hdbscan_factors",
            "use_als": "force_off",   # → SVD as factor source
        },
        "hdbscan_als": {
            **base,
            "persona_method": "hdbscan_factors",
            "use_als": "force_on",
        },
        "louvain_raw": {
            **base,
            "persona_method": "louvain_graph",
            "louvain_weight_transform": "raw",
            "louvain_min_edge_percentile": 0.0,
        },
        "louvain_log_prune": {
            **base,
            "persona_method": "louvain_graph",
            "louvain_weight_transform": "log",
            "louvain_min_edge_percentile": 0.05,
        },
        "louvain_cosine": {
            **base,
            "persona_method": "louvain_graph",
            "louvain_weight_transform": "cosine",
        },
        "louvain_gamma_2": {
            **base,
            "persona_method": "louvain_graph",
            "louvain_resolution": 2.0,
        },
        "dc_sbm": {
            **base,
            "persona_method": "dc_sbm",
            # Raw weights + γ=2.0 warm-start. min_internal_fraction=0.0
            # because the algorithm-agnostic coherence filter is the
            # principled noise gate — internal-fraction routing turned
            # out to over-collapse dense graphs (everything → -1 → 1
            # block survives).
            "louvain_weight_transform": "raw",
            "dc_sbm_warmstart_resolution": 2.0,
            "dc_sbm_min_internal_fraction": 0.0,
            "dc_sbm_max_passes": 12,
        },
    }


def run(
    loader: str,
    max_eval_users: int = 500,
    k: int = 10,
    seed: int = 0,
    coherence_filter: float = 0.5,
) -> SweepReport:
    from kindling.benchmarks.comparison import _load_dataset

    split = _load_dataset(loader, test_fraction=0.1)
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=max_eval_users, seed=seed)
    if not eval_set:
        raise RuntimeError("eval set empty")

    report = SweepReport(
        loader=loader,
        n_train=len(train),
        n_test=len(test),
        n_users_evaluated=len(eval_set),
        k=k,
        coherence_filter_percentile=coherence_filter,
    )
    specs = _variant_specs(coherence_filter)
    for name, kwargs in specs.items():
        engine = EngineV2(retrieval_budget=500, random_state=seed, **kwargs)
        t0 = time.perf_counter()
        engine.fit(train)
        fit_s = time.perf_counter() - t0
        metrics, latencies = _evaluate(engine, eval_set, k=k)
        s = engine.fit_summary()
        coh = s["profile"].get("persona_coherence", {})
        # Persona-vs-cooc differentiation diagnostic — answers "are we
        # surfacing meaningfully different recs or just a subset of cooc?"
        diff = compute_persona_diff(
            engine, sample_users_per_persona=30, k=k, seed=seed,
        ).get("global", {})
        report.variants[name] = {
            "kwargs": {k_: v for k_, v in kwargs.items() if k_ != "coherence_filter_percentile"},
            "fit_seconds": fit_s,
            "metrics": metrics,
            "latency": latencies,
            "n_personas": s["n_personas"],
            "persona_method_used": s["persona_method_used"],
            "coherence": coh,
            "persona_vs_cooc_diff": diff,
        }
    return report


def render_markdown(report: SweepReport) -> str:
    names = list(report.variants.keys())
    lines = [
        f"# Clustering × coherence sweep — {report.loader}",
        "",
        f"- users evaluated: {report.n_users_evaluated}",
        f"- train / test: {report.n_train:,} / {report.n_test:,}",
        f"- k = {report.k}",
        f"- coherence_filter_percentile: {report.coherence_filter_percentile:.2f}",
        f"- timestamp: {report.timestamp}",
        "",
        "## Coherence distribution per variant",
        "",
        f"| variant | n_personas | n_kept | mean | median | p25 | p75 | min | max |",
        f"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in names:
        v = report.variants[n]
        coh = v.get("coherence", {})
        n_personas = v["n_personas"]
        n_kept = coh.get("n_personas_kept", "—")
        mean = coh.get("mean", 0.0)
        median = coh.get("median", 0.0)
        p25 = coh.get("p25", 0.0)
        p75 = coh.get("p75", 0.0)
        cmin = coh.get("min", 0.0)
        cmax = coh.get("max", 0.0)
        lines.append(
            f"| `{n}` | {n_personas} | {n_kept} | {mean:.3f} | {median:.3f} | "
            f"{p25:.3f} | {p75:.3f} | {cmin:.3f} | {cmax:.3f} |"
        )
    lines += [
        "",
        "## Persona vs cooc differentiation",
        "",
        "How different are persona-cooc top-K recs from global cooc top-K?",
        "- `jaccard@K`: set overlap of top-K (1 = identical, 0 = disjoint)",
        "- `kendall_tau`: rank agreement on shared items (-1 to 1)",
        "- `rank_shift`: mean global-cooc rank of items unique to persona top-K (large = persona surfaces things cooc would have ranked far down)",
        "- `frac_identical`: fraction of users where persona top-K = cooc top-K",
        "",
        f"| variant | jaccard@K | kendall_tau | rank_shift | frac_identical | n_users |",
        f"| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n in names:
        d = report.variants[n].get("persona_vs_cooc_diff", {})
        if d:
            lines.append(
                f"| `{n}` | {d.get('mean_jaccard_at_k', 0):.3f} | "
                f"{d.get('mean_kendall_tau', 0):.3f} | "
                f"{d.get('mean_rank_shift_unique', 0):.0f} | "
                f"{d.get('fraction_identical', 0):.2%} | "
                f"{d.get('n_users_sampled', 0)} |"
            )
        else:
            lines.append(f"| `{n}` | — | — | — | — | — |")
    lines += [
        "",
        "## Quality (with coherence filter applied)",
        "",
        f"| metric | {' | '.join(names)} |",
        f"| --- | {' | '.join(['---:'] * len(names))} |",
    ]
    for m in ("ndcg_at_k", "mrr", "recall_at_k", "hit_rate", "coverage"):
        row = [f"`{m}`"] + [
            f"{report.variants[n]['metrics'][m]:.4f}" for n in names
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
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
        f"| stat | {' | '.join(names)} |",
        f"| --- | {' | '.join(['---:'] * len(names))} |",
    ]
    for stat in ("p50_ms", "p95_ms", "p99_ms"):
        lines.append(
            f"| `{stat}` | "
            + " | ".join(f"{report.variants[n]['latency'][stat]:.2f}" for n in names)
            + " |"
        )
    lines += [
        "",
        "## Variant configs",
        "",
    ]
    for n in names:
        kw = report.variants[n]["kwargs"]
        lines.append(f"- `{n}`: `{kw}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", default="amazon-beauty")
    parser.add_argument("--max-eval-users", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coherence-filter", type=float, default=0.5,
                        help="percentile of personas to drop by coherence (0.0=no filter, 0.5=drop bottom half)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run(
        args.loader,
        max_eval_users=args.max_eval_users,
        k=args.k,
        seed=args.seed,
        coherence_filter=args.coherence_filter,
    )
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
