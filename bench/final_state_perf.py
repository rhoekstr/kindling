"""Final-state performance of the native-only engine.

Measures, per dataset, the production engine's headline numbers: fit time, the
native engine build time, single-recommend latency (p50/p95), batch-recommend
throughput (the parallel Rust path), peak RSS, and NDCG@10 (confirming the
native engine reproduces the frozen gates.toml baseline). Writes
``bench/reports/final_state_perf.json``.

Single recommend went from the Python loop (~200 ms p50 on ml1m, see
bench/reports/baselines_comparison.json) to the native path; batch is the
GIL-released parallel win. Accuracy is unchanged (the port is NDCG-identical).

Run:  python bench/final_state_perf.py [dataset ...]
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from kindling import Engine
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set

_CONFIG = {
    "movielens-1m": {},
    "amazon-beauty": {"ease_lambda": 250.0},
    "steam": {"cold_slots": 1},
}


def _rss_gb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**3 if sys.platform == "darwin" else 1024**2)


def measure(dataset: str) -> dict:
    cfg = _CONFIG.get(dataset, {})
    split = _load_dataset(dataset, 0.1)
    has_meta = getattr(split, "items", None) is not None

    t0 = time.perf_counter()
    eng = Engine(retrieval_budget=500, random_state=0, **cfg).fit(
        split.train, item_metadata=split.items if has_meta else None
    )
    fit_s = time.perf_counter() - t0
    st = eng._state

    t0 = time.perf_counter()
    eng._require_native()  # build the native engine (EASE-matrix copy etc.)
    build_s = time.perf_counter() - t0

    eval_set = _build_eval_set(split.train, split.test, max_users=500, seed=0)
    ents = [e for e in eval_set if (ow := st.owned_by_entity.get(e)) is not None and ow.size > 0]

    # Single-recommend latency.
    lat = []
    for e in ents:
        t0 = time.perf_counter()
        eng.recommend(e, 10)
        lat.append((time.perf_counter() - t0) * 1e3)
    lat_arr = np.array(lat)

    # Batch throughput.
    t0 = time.perf_counter()
    batch = eng.recommend_batch(ents, 10)
    batch_s = time.perf_counter() - t0

    per = [([r.item_id for r in batch[i]], eval_set[ents[i]]) for i in range(len(ents))]
    ndcg = float(aggregate(per, catalog_size=max(st.n_items, 1), k=10).ndcg_at_k)

    return {
        "dataset": dataset,
        "base": st.base_scorer_used,
        "n_items": int(st.n_items),
        "n_users": int(st.n_users),
        "fit_seconds": round(fit_s, 2),
        "native_build_seconds": round(build_s, 3),
        "single_recommend_p50_ms": round(float(np.percentile(lat_arr, 50)), 3),
        "single_recommend_p95_ms": round(float(np.percentile(lat_arr, 95)), 3),
        "batch_throughput_recs_per_s": round(len(ents) / batch_s, 1),
        "batch_total_ms": round(batch_s * 1e3, 1),
        "n_eval_users": len(ents),
        "ndcg_at_10": round(ndcg, 4),
        "peak_rss_gb": round(_rss_gb(), 2),
    }


def main(argv: list[str]) -> int:
    datasets = argv[1:] or ["movielens-1m", "amazon-beauty", "steam"]
    results = [measure(d) for d in datasets]
    out = Path(__file__).resolve().parent / "reports" / "final_state_perf.json"
    out.write_text(json.dumps({"results": results}, indent=2))
    hdr = f"{'dataset':16s} {'base':5s} {'fit_s':>7s} {'p50_ms':>7s} {'p95_ms':>7s} {'batch_r/s':>10s} {'ndcg@10':>8s} {'rss_gb':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['dataset']:16s} {r['base']:5s} {r['fit_seconds']:7.1f} "
            f"{r['single_recommend_p50_ms']:7.2f} {r['single_recommend_p95_ms']:7.2f} "
            f"{r['batch_throughput_recs_per_s']:10.0f} {r['ndcg_at_10']:8.4f} {r['peak_rss_gb']:7.2f}"
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
