"""Per-user-warmth segmentation: the direct test of "do we handle cold USERS
better?" At full data, bucket eval users by their train history length and
compare kindling vs standard algorithms WITHIN each bucket.

The density-warming view (run_warming_curve.py) conflated cold-users with a
thin global dataset. This isolates user coldness: every model sees the full
dataset; we slice the eval population by how much history each user has. If
kindling leads on the COLD (short-history) buckets, that supports the
"handles cold users better" thesis — this is where its short-history
machinery lives (last-item channel, user-CF [auto-on for sparse data],
EASE neighborhood).

Models: kindling (EngineV2) · implicit ALS · item-item kNN · popularity.

Run: DATASET=amazon-beauty .venv/bin/python bench/run_user_warmth.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from run_warming_curve import build_models

BUCKETS = {"1-4": (1, 5), "5-19": (5, 20), "20-49": (20, 50), "50+": (50, 10**9)}
REPORT_DIR = Path(__file__).parent / "reports"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    dataset = os.environ.get("DATASET", "amazon-beauty")
    k = int(os.environ.get("K", "10"))
    max_eval = int(os.environ.get("MAX_EVAL", "4000"))
    split = _load_dataset(dataset, test_fraction=0.1)
    train, test = split.train, split.test
    train_by = train.groupby("entity_id", sort=False)["item_id"].apply(lambda s: set(s))
    test_by = test.groupby("entity_id", sort=False)["item_id"].apply(lambda s: set(s))
    catalog = int(train["item_id"].nunique())
    eval_all = sorted(set(train_by.index) & set(test_by.index))
    step = max(1, len(eval_all) // max_eval)
    eval_entities = eval_all[::step][:max_eval]

    def bucket_of(n: int) -> str | None:
        for name, (lo, hi) in BUCKETS.items():
            if lo <= n < hi:
                return name
        return None

    ent_bucket = {e: bucket_of(len(train_by[e])) for e in eval_entities}
    bcounts = {b: sum(1 for e in eval_entities if ent_bucket[e] == b) for b in BUCKETS}
    log(f"{dataset}: eval_users {len(eval_entities)} catalog {catalog:,} "
        f"buckets {bcounts}")

    results: dict[str, dict] = {}
    for model in build_models(include_als=True):
        t0 = time.perf_counter()
        model.fit(train)
        # collect per-bucket (recs, relevant)
        per_bucket: dict[str, list] = {b: [] for b in BUCKETS}
        for ent in eval_entities:
            b = ent_bucket[ent]
            if b is None:
                continue
            rel = test_by[ent] - train_by[ent]
            recs = model.recommend(ent, n=k)
            per_bucket[b].append((recs, rel))
        row = {}
        for b in BUCKETS:
            if per_bucket[b]:
                m = aggregate(per_bucket[b], catalog_size=catalog, k=k)
                row[b] = {"ndcg": round(m.ndcg_at_k, 4), "recall": round(m.recall_at_k, 4),
                          "n": len(per_bucket[b])}
            else:
                row[b] = {"ndcg": None, "recall": None, "n": 0}
        results[model.name] = row
        log(f"  {model.name:14s} fit={time.perf_counter()-t0:5.1f}s | " +
            "  ".join(f"{b}:{row[b]['ndcg']}" for b in BUCKETS))

    out = {"dataset": dataset, "k": k, "catalog": catalog,
           "bucket_counts": bcounts, "results": results}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"user_warmth_{dataset}.json").write_text(json.dumps(out, indent=2) + "\n")
    log(f"[wrote] user_warmth_{dataset}.json")


if __name__ == "__main__":
    main()
