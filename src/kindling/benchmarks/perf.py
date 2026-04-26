"""v2 performance harness — per-stage timings + recommend latency + RSS.

Emits the JSON format from PRD §"Performance — measurement and gating":

    {
      "kindling_version": "0.x.0",
      "variant": "v1" | "v2",
      "loader": "<name>",
      "machine": {"cpu": "...", "ram_gb": ...},
      "fit": {
        "total_seconds": ...,
        "by_stage": {"profile": ..., "cooc": ..., ...},
        "peak_rss_mb": ...
      },
      "recommend": {
        "n_users_sampled": ...,
        "p50_ms": ..., "p95_ms": ..., "p99_ms": ...,
        "throughput_per_sec": ...
      }
    }

CLI:

    python -m kindling.benchmarks.perf \\
        --variant v2 \\
        --dataset synthetic_small \\
        --output bench/reports/perf/<sha>.json

Two execution modes:

- ``--variant v1`` runs the legacy Engine.
- ``--variant v2`` runs Engine(use_v2_core=True).

Datasets are pluggable. ``synthetic_small/medium/large`` are built in
for environments without the real loaders fitted yet (Phase 4 wires
the Polars loaders). When real loaders are available, ``--dataset
movielens-1m`` etc. forwards to them.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kindling import Engine, __version__


@dataclass
class PerfReport:
    """Output schema (matches PRD §"Performance" JSON spec)."""

    kindling_version: str
    rust_core_sha: str | None
    variant: str
    loader: str
    machine: dict[str, Any]
    fit: dict[str, Any]
    recommend: dict[str, Any]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


def synthetic_dataset(n_users: int, n_items: int, density: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic two-cluster dataset for measurement."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    n_per_user = max(3, int(density * n_items))
    for u in range(n_users):
        # Two-cluster taste structure to give the persona pipeline
        # something coherent to find.
        pref_lo, pref_hi = (0, n_items // 2) if u < n_users // 2 else (n_items // 2, n_items)
        size = min(n_per_user, pref_hi - pref_lo)
        items = rng.choice(np.arange(pref_lo, pref_hi), size=size, replace=False)
        for i in items:
            rows.append({"entity_id": f"u{u}", "item_id": int(i), "_interaction_weight": 1.0})
    return pd.DataFrame(rows)


def _machine_info() -> dict[str, Any]:
    """Best-effort system info."""
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 1)
    except (AttributeError, ValueError):
        ram_gb = None
    return {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ram_gb": ram_gb,
    }


def _rust_core_sha() -> str | None:
    """Return the git SHA of the kindling_core crate, if reachable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _peak_rss_mb() -> float:
    """Peak RSS for this process in megabytes."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # On Linux this is in kilobytes; on macOS in bytes. Detect.
    raw = ru.ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024


def measure_fit(engine: Engine, interactions: pd.DataFrame) -> dict[str, Any]:
    """Time fit + capture per-stage timings if surfaced by EngineV2."""
    gc.collect()
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    engine.fit(interactions)
    total = time.perf_counter() - t0
    rss_after = _peak_rss_mb()
    by_stage: dict[str, float] = {}
    if engine.use_v2_core and engine._v2_engine is not None:
        summary = engine._v2_engine.fit_summary()  # type: ignore[attr-defined]
        # EngineV2 doesn't yet record per-stage timings — only total.
        # Phase 6.next adds per-stage timing decorators.
        by_stage = {"total": float(summary.get("fit_seconds", total))}
        by_stage["profile_decisions"] = float(summary.get("profile", {}).get("density", 0.0))
    return {
        "total_seconds": total,
        "by_stage": by_stage,
        "peak_rss_mb": max(rss_before, rss_after),
    }


def measure_recommend(
    engine: Engine,
    interactions: pd.DataFrame,
    n_sample: int = 500,
    k: int = 10,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Sample users + measure per-recommend latency."""
    rng = np.random.default_rng(rng_seed)
    eligible = interactions["entity_id"].unique()
    sample_size = min(n_sample, len(eligible))
    if sample_size == 0:
        return {"n_users_sampled": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "throughput_per_sec": 0}
    sample_idx = rng.choice(len(eligible), size=sample_size, replace=False)
    sampled = [eligible[int(i)] for i in sample_idx]
    latencies_ms: list[float] = []
    t0 = time.perf_counter()
    for entity in sampled:
        s = time.perf_counter()
        engine.recommend(entity_id=entity, n=k)
        latencies_ms.append((time.perf_counter() - s) * 1000.0)
    elapsed = time.perf_counter() - t0
    arr = np.asarray(latencies_ms)
    return {
        "n_users_sampled": int(sample_size),
        "k": int(k),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "throughput_per_sec": float(sample_size / elapsed) if elapsed > 0 else 0.0,
    }


def run(
    variant: str,
    dataset: str,
    n_sample: int = 500,
    k: int = 10,
    seed: int = 0,
) -> PerfReport:
    """Build engine + dataset, measure, return PerfReport."""
    if dataset == "synthetic_small":
        interactions = synthetic_dataset(n_users=200, n_items=100, density=0.05, seed=seed)
    elif dataset == "synthetic_medium":
        interactions = synthetic_dataset(n_users=2000, n_items=500, density=0.03, seed=seed)
    elif dataset == "synthetic_large":
        interactions = synthetic_dataset(n_users=10_000, n_items=2_000, density=0.02, seed=seed)
    else:
        raise ValueError(
            f"unknown dataset {dataset!r}; available: synthetic_small/medium/large. "
            "Real loaders wire in via Phase 4."
        )
    if variant == "v1":
        engine = Engine()
    elif variant == "v2":
        engine = Engine(use_v2_core=True)
    else:
        raise ValueError(f"unknown variant {variant!r}; expected 'v1' or 'v2'")

    fit_metrics = measure_fit(engine, interactions)
    rec_metrics = measure_recommend(engine, interactions, n_sample=n_sample, k=k, rng_seed=seed)

    return PerfReport(
        kindling_version=__version__,
        rust_core_sha=_rust_core_sha(),
        variant=variant,
        loader=dataset,
        machine=_machine_info(),
        fit=fit_metrics,
        recommend=rec_metrics,
    )


def render_markdown(report: PerfReport) -> str:
    """Markdown view of one PerfReport. Stable layout for diffing."""
    lines: list[str] = [
        f"# Perf report — {report.variant} on {report.loader}",
        "",
        f"- **kindling**: {report.kindling_version}",
        f"- **rust_core_sha**: `{report.rust_core_sha or 'n/a'}`",
        f"- **timestamp**: {report.timestamp}",
        f"- **machine**: {report.machine.get('cpu')} ({report.machine.get('platform')})",
        "",
        "## Fit",
        "",
        f"- total: **{report.fit['total_seconds']:.3f}s**",
        f"- peak RSS: {report.fit['peak_rss_mb']:.0f} MB",
    ]
    if report.fit.get("by_stage"):
        lines.append("- by stage:")
        for stage, t in report.fit["by_stage"].items():
            lines.append(f"  - `{stage}`: {t:.3f}s")
    lines.extend([
        "",
        "## Recommend",
        "",
        f"- users sampled: {report.recommend['n_users_sampled']}",
        f"- top-K: {report.recommend.get('k', '?')}",
        f"- p50 / p95 / p99: **{report.recommend['p50_ms']:.2f}** / "
        f"{report.recommend['p95_ms']:.2f} / "
        f"{report.recommend['p99_ms']:.2f} ms",
        f"- throughput: {report.recommend['throughput_per_sec']:.0f} recs/sec",
    ])
    return "\n".join(lines) + "\n"


def diff_reports(baseline: PerfReport, candidate: PerfReport) -> dict[str, dict[str, Any]]:
    """Compute per-metric pct deltas (candidate - baseline) / baseline.

    Used by CI to surface regressions. Positive values = slower / more
    memory; negative values = improvement.
    """
    def pct(b: float, c: float) -> float:
        return ((c - b) / b * 100.0) if b > 0 else float("inf")

    return {
        "fit": {
            "total_seconds": pct(
                baseline.fit["total_seconds"], candidate.fit["total_seconds"]
            ),
            "peak_rss_mb": pct(baseline.fit["peak_rss_mb"], candidate.fit["peak_rss_mb"]),
        },
        "recommend": {
            "p50_ms": pct(baseline.recommend["p50_ms"], candidate.recommend["p50_ms"]),
            "p95_ms": pct(baseline.recommend["p95_ms"], candidate.recommend["p95_ms"]),
            "p99_ms": pct(baseline.recommend["p99_ms"], candidate.recommend["p99_ms"]),
            "throughput_per_sec": -pct(  # negative because higher throughput is better
                baseline.recommend["throughput_per_sec"],
                candidate.recommend["throughput_per_sec"],
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["v1", "v2"], default="v2")
    parser.add_argument("--dataset", default="synthetic_small")
    parser.add_argument("--n-sample", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None,
                       help="Write JSON to this path. Default: print to stdout.")
    parser.add_argument("--markdown", type=Path, default=None,
                       help="Also render markdown to this path.")
    args = parser.parse_args(argv)

    report = run(args.variant, args.dataset, n_sample=args.n_sample, k=args.k, seed=args.seed)
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
