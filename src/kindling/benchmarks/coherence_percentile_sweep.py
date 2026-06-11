"""Coherence-filter percentile sweep on the best variant per dataset.

For each loader, fix the best clustering method (per the prior
clustering_coherence_sweep results) and vary `coherence_filter_percentile`
∈ {0.0, 0.25, 0.5, 0.75}. Tests whether the post-hoc filter is doing
useful work or whether keeping more / fewer personas helps quality.

Best variant per loader (from clustering_coherence_sweep.py results):
    movielens-1m   → louvain_gamma_2  (NDCG 0.2582)
    amazon-beauty  → dc_sbm           (NDCG 0.0294)
    amazon-book    → louvain_cosine   (NDCG 0.0253, ties hdbscan_svd
                                       but at much lower train cost)

Reports landing-zone:
    bench/reports/consolidated/coherence_percentile_<loader>.{json,md}
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
class PctReport:
    loader: str
    n_train: int
    n_test: int
    n_users_evaluated: int
    k: int
    best_variant: str
    best_kwargs: dict[str, Any]
    percentiles: list[float]
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


_BEST_PER_LOADER: dict[str, tuple[str, dict[str, Any]]] = {
    "movielens-1m": (
        "louvain_gamma_2",
        {"persona_method": "louvain_graph", "louvain_resolution": 2.0},
    ),
    "amazon-beauty": (
        "dc_sbm",
        {
            "persona_method": "dc_sbm",
            "louvain_weight_transform": "raw",
            "dc_sbm_warmstart_resolution": 2.0,
            "dc_sbm_min_internal_fraction": 0.0,
            "dc_sbm_max_passes": 12,
            # Pin to louvain init to match the original best result
            # (96 blocks via Louvain noise → singletons promotion).
            "dc_sbm_init_mode": "louvain",
        },
    ),
    "amazon-book": (
        "louvain_cosine",
        {"persona_method": "louvain_graph", "louvain_weight_transform": "cosine"},
    ),
}

_PERCENTILES = [0.0, 0.25, 0.5, 0.75]


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


def run(loader: str, max_eval_users: int = 500, k: int = 10, seed: int = 0) -> PctReport:
    from kindling.benchmarks.comparison import _load_dataset

    if loader not in _BEST_PER_LOADER:
        raise ValueError(f"no best-variant config for {loader!r}; add to _BEST_PER_LOADER")
    best_variant, base_kwargs = _BEST_PER_LOADER[loader]

    split = _load_dataset(loader, test_fraction=0.1)
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=max_eval_users, seed=seed)
    if not eval_set:
        raise RuntimeError("eval set empty")

    report = PctReport(
        loader=loader,
        n_train=len(train),
        n_test=len(test),
        n_users_evaluated=len(eval_set),
        k=k,
        best_variant=best_variant,
        best_kwargs=base_kwargs,
        percentiles=_PERCENTILES,
    )
    for pct in _PERCENTILES:
        kwargs = {**base_kwargs, "coherence_filter_percentile": pct}
        engine = EngineV2(retrieval_budget=500, random_state=seed, **kwargs)
        t0 = time.perf_counter()
        engine.fit(train)
        fit_s = time.perf_counter() - t0
        metrics, latencies = _evaluate(engine, eval_set, k=k)
        s = engine.fit_summary()
        coh = s["profile"].get("persona_coherence", {})
        diff = compute_persona_diff(
            engine, sample_users_per_persona=30, k=k, seed=seed,
        ).get("global", {})
        label = f"pct_{pct:.2f}"
        report.runs[label] = {
            "coherence_filter_percentile": pct,
            "fit_seconds": fit_s,
            "metrics": metrics,
            "latency": latencies,
            "n_personas": s["n_personas"],
            "coherence": coh,
            "persona_vs_cooc_diff": diff,
        }
    return report


def render_markdown(report: PctReport) -> str:
    pcts = report.percentiles
    cols = [f"pct={p:.2f}" for p in pcts]
    labels = [f"pct_{p:.2f}" for p in pcts]
    lines = [
        f"# Coherence-filter percentile sweep — {report.loader}",
        "",
        f"- best variant: `{report.best_variant}`",
        f"- users evaluated: {report.n_users_evaluated}",
        f"- train / test: {report.n_train:,} / {report.n_test:,}",
        f"- k = {report.k}",
        f"- timestamp: {report.timestamp}",
        f"- variant kwargs: `{report.best_kwargs}`",
        "",
        "## Quality",
        "",
        f"| metric | {' | '.join(cols)} |",
        f"| --- | {' | '.join(['---:'] * len(pcts))} |",
    ]
    for m in ("ndcg_at_k", "mrr", "recall_at_k", "hit_rate", "coverage"):
        row = [f"`{m}`"] + [
            f"{report.runs[lab]['metrics'][m]:.4f}" for lab in labels
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Persona structure",
        "",
        f"| stage | {' | '.join(cols)} |",
        f"| --- | {' | '.join(['---:'] * len(pcts))} |",
        "| n_personas | "
        + " | ".join(str(report.runs[lab]["n_personas"]) for lab in labels)
        + " |",
        "| n_kept | "
        + " | ".join(
            str(report.runs[lab]["coherence"].get("n_personas_kept", report.runs[lab]["n_personas"]))
            for lab in labels
        )
        + " |",
        "| mean_coherence | "
        + " | ".join(f"{report.runs[lab]['coherence'].get('mean', 0):.3f}" for lab in labels)
        + " |",
        "",
        "## Differentiation vs cooc",
        "",
        f"| stat | {' | '.join(cols)} |",
        f"| --- | {' | '.join(['---:'] * len(pcts))} |",
    ]
    for stat, key in [
        ("jaccard@K", "mean_jaccard_at_k"),
        ("kendall_tau", "mean_kendall_tau"),
        ("rank_shift", "mean_rank_shift_unique"),
        ("frac_identical", "fraction_identical"),
        ("n_users_sampled", "n_users_sampled"),
    ]:
        if key == "n_users_sampled":
            row = [f"`{stat}`"] + [
                str(report.runs[lab]["persona_vs_cooc_diff"].get(key, 0))
                for lab in labels
            ]
        elif key == "fraction_identical":
            row = [f"`{stat}`"] + [
                f"{report.runs[lab]['persona_vs_cooc_diff'].get(key, 0):.2%}"
                for lab in labels
            ]
        elif key == "mean_rank_shift_unique":
            row = [f"`{stat}`"] + [
                f"{report.runs[lab]['persona_vs_cooc_diff'].get(key, 0):.0f}"
                for lab in labels
            ]
        else:
            row = [f"`{stat}`"] + [
                f"{report.runs[lab]['persona_vs_cooc_diff'].get(key, 0):.3f}"
                for lab in labels
            ]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## Fit timing (s)",
        "",
        f"| stage | {' | '.join(cols)} |",
        f"| --- | {' | '.join(['---:'] * len(pcts))} |",
        "| fit_s | "
        + " | ".join(f"{report.runs[lab]['fit_seconds']:.2f}" for lab in labels)
        + " |",
    ]
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
