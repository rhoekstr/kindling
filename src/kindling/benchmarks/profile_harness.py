"""Performance profiling harness (plan Phase 8).

Measures fit time, recommend latency (p50 / p95 / p99), and peak
memory against the PRD §13.1 targets:

- Cold recommend (no Bayesian): 50ms p95 at <= 10k items, budget 500.
- Warm recommend (Bayesian on): 150ms p95 at <= 100k items, budget 500.
- Initial fit: < 60s at <= 1M interactions, <= 10k items.
- Memory footprint: < 2 GB at 100k items + 10M interactions.

Also runs cProfile over a sample of recommend calls so the hot path is
visible in the report. Results go to ``bench/reports/profile_*.json``.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import json
import pstats
import resource
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from kindling import __version__
from kindling.engine import Engine
from kindling.loaders import synthetic


@dataclass(frozen=True)
class RecommendLatency:
    n_samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float


@dataclass(frozen=True)
class MemoryMeasurement:
    peak_rss_mb: float
    peak_tracemalloc_mb: float


@dataclass(frozen=True)
class ProfileRun:
    regime: str  # "cold" or "warm"
    n_items: int
    n_entities: int
    n_interactions: int
    fit_seconds: float
    recommend: RecommendLatency
    memory: MemoryMeasurement
    hot_paths: list[tuple[str, float]] = field(default_factory=list)
    engine_version: str = __version__

    def passes_prd(self) -> dict[str, bool]:
        if self.regime == "cold":
            return {
                "latency_p95_under_50ms": self.recommend.p95_ms < 50.0,
                "fit_under_60s": self.fit_seconds < 60.0,
            }
        return {
            "latency_p95_under_150ms": self.recommend.p95_ms < 150.0,
            "memory_under_2gb": self.memory.peak_rss_mb < 2048.0,
        }


def _synth(regime: str) -> tuple[Engine, list[object]]:
    """Build an engine fitted to a dataset sized for the regime and
    return it alongside the list of evaluation entity ids."""
    if regime == "cold":
        # PRD target: <= 10k items. Keep interactions compact so the
        # cold regime stays cold (no Bayesian).
        split = synthetic.make_grocery(
            n_entities=500,
            n_items_per_category=1000,
            n_categories=10,  # total 10k items
            n_sessions_per_entity=3,
            items_per_session=4,
            seed=0,
        )
        engine = Engine(use_bayesian_blend=False, vi_max_iter=0).fit(split.train)
    elif regime == "warm":
        # Closer to the PRD warm-regime target. 100k items + moderate
        # sparsity keeps memory under 2GB on laptop-class hardware.
        split = synthetic.make_grocery(
            n_entities=2000,
            n_items_per_category=10000,
            n_categories=10,  # total 100k items
            n_sessions_per_entity=5,
            items_per_session=6,
            seed=0,
        )
        engine = Engine(use_bayesian_blend=True, vi_max_iter=50).fit(split.train)
    else:
        raise ValueError(f"unknown regime {regime!r}")

    # Evaluation entities: sample 500 that exist in the fitted engine.
    entity_ids = list(engine._owned_by_entity.keys())[:500]
    return engine, entity_ids


def _measure_memory() -> MemoryMeasurement:
    # Peak RSS: whatever the process has ever needed. Reported in MB.
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes, Linux in kilobytes.
    rss_mb = rss_kb / 1024.0 / 1024.0 if sys.platform == "darwin" else rss_kb / 1024.0
    trace_peak_mb = 0.0
    if tracemalloc.is_tracing():
        _, trace_peak = tracemalloc.get_traced_memory()
        trace_peak_mb = trace_peak / 1024.0 / 1024.0
    return MemoryMeasurement(peak_rss_mb=rss_mb, peak_tracemalloc_mb=trace_peak_mb)


def _time_recommends(
    engine: Engine, entity_ids: list[object], n: int = 10
) -> RecommendLatency:
    durations_ms: list[float] = []
    for entity in entity_ids:
        t0 = time.perf_counter()
        engine.recommend(entity_id=entity, n=n)
        durations_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(durations_ms)
    return RecommendLatency(
        n_samples=len(arr),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        max_ms=float(arr.max()),
        mean_ms=float(arr.mean()),
    )


def _profile_hot_paths(
    engine: Engine, entity_ids: list[object], top_n: int = 10
) -> list[tuple[str, float]]:
    """Run cProfile over a sample of recommend calls and return the top-N
    cumulative-time functions."""
    pr = cProfile.Profile()
    pr.enable()
    for entity in entity_ids[:100]:
        engine.recommend(entity_id=entity, n=10)
    pr.disable()
    stream = io.StringIO()
    pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    hot: list[tuple[str, float]] = []
    stats_obj = pstats.Stats(pr)
    stats_dict: dict = getattr(stats_obj, "stats", {})  # type: ignore[type-arg]
    if not stats_dict:
        return hot
    sorted_items = sorted(stats_dict.items(), key=lambda kv: -kv[1][3])
    for func, (_cc, _nc, _tt, ct, _callers) in sorted_items[:top_n]:
        file_part, _, _ = func[0].rpartition("/")
        short = f"{file_part.split('/')[-1]}:{func[1]}({func[2]})"
        hot.append((short, float(ct)))
    return hot


def run_profile(regime: str) -> ProfileRun:
    """Fit, time recommends, and capture memory."""
    tracemalloc.start()
    gc.collect()

    fit_start = time.perf_counter()
    engine, entity_ids = _synth(regime)
    fit_seconds = time.perf_counter() - fit_start

    # Warm the recommend path before timing so imports / caches don't
    # distort the first measurement.
    for entity in entity_ids[:5]:
        engine.recommend(entity_id=entity, n=10)

    latency = _time_recommends(engine, entity_ids, n=10)
    memory = _measure_memory()
    hot = _profile_hot_paths(engine, entity_ids)
    tracemalloc.stop()

    # Pull the interaction counts for reporting.
    assert engine._interactions is not None
    n_interactions = len(engine._interactions)
    assert engine._item_graph is not None
    n_items = engine._item_graph.n_items
    n_entities = len(engine._owned_by_entity)

    return ProfileRun(
        regime=regime,
        n_items=n_items,
        n_entities=n_entities,
        n_interactions=n_interactions,
        fit_seconds=fit_seconds,
        recommend=latency,
        memory=memory,
        hot_paths=hot,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 8 performance profile against PRD targets."
    )
    parser.add_argument(
        "--regime",
        choices=["cold", "warm", "both"],
        default="both",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    regimes = ["cold", "warm"] if args.regime == "both" else [args.regime]
    runs = [run_profile(r) for r in regimes]
    payload = {
        "engine_version": __version__,
        "runs": [asdict(r) for r in runs],
        "prd_checks": {r.regime: r.passes_prd() for r in runs},
    }

    pretty = json.dumps(payload, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(pretty + "\n")
        print(f"Wrote {args.output}")
        for r in runs:
            prd = r.passes_prd()
            status = "PASS" if all(prd.values()) else "MISS"
            print(
                f"  {r.regime:5s} {status} p95={r.recommend.p95_ms:.1f}ms "
                f"fit={r.fit_seconds:.1f}s rss={r.memory.peak_rss_mb:.0f}MB "
                f"items={r.n_items} interactions={r.n_interactions}"
            )
    else:
        print(pretty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
