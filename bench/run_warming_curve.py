"""Warming curve: kindling (validated v2 stack) vs standard algorithms as a
dataset accumulates interactions ("warms").

Thesis (the positive story behind this session's negatives): kindling's
closed-form, no-training stack (wilson cooc / EASE + channels, auto-gated) should
deliver strong recommendations from very little data and fit in seconds, while
trained latent models (implicit ALS) need more data + iterations to converge. So
on the cold/early end of the curve kindling should lead on accuracy AND speed,
and stay competitive as data warms.

For each chronological prefix fraction of train (earliest events = a young
system), fit every model and evaluate on the SAME fixed held-out test window
over a FIXED eval population (entities in full train∩test). Records, per
(fraction, model): fit_seconds (speed), recommend p50 latency, recall@k,
ndcg@k (accuracy). NOT cold-ITEM start (that program is closed §4.9) — this is
cold-as-in-data-scarce-SYSTEM.

Models: kindling (EngineV2, validated defaults) · Popularity · ItemItemKNN ·
implicit ALS (the industry-standard trained MF baseline).

Run: DATASET=movielens-1m FRACTIONS=0.01,0.02,0.05,0.1,0.2,0.4,0.7,1.0 \
     .venv/bin/python bench/run_warming_curve.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kindling.benchmarks.baselines import (
    ImplicitALSBaseline,
    ItemItemKNN,
    PopularityBaseline,
)
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate

REPORT_DIR = Path(__file__).parent / "reports"


def load_split(name: str, tf: float = 0.1):
    """_load_dataset, plus the timestamp-less book-academic (52k/91k) split —
    full amazon-book (357k) OOMs / trips the kill wall on this box."""
    if name == "amazon-book-academic":
        from kindling.benchmarks.comparison import _load_academic_split
        b = Path("~/.cache/kindling/amazon-book").expanduser()
        return _load_academic_split(b / "train.txt", b / "test.txt",
                                    name=name, action_type="rate")
    return _load_dataset(name, test_fraction=tf)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class KindlingV2:
    name = "kindling"

    def fit(self, df: pd.DataFrame):
        from kindling.engine_v2 import EngineV2
        self._e = EngineV2(persona_min_users=10**9, retrieval_budget=500, random_state=0)
        self._e.fit(df)
        return self

    def recommend(self, entity_id, n: int = 10):
        return [r.item_id for r in self._e.recommend(entity_id, n=n)]


def build_models(include_als: bool):
    models = [KindlingV2(), PopularityBaseline(), ItemItemKNN(k_neighbors=200)]
    if include_als:
        models.append(ImplicitALSBaseline(factors=64, iterations=15))
    return models


def main() -> None:
    dataset = os.environ.get("DATASET", "movielens-1m")
    fractions = [float(x) for x in os.environ.get(
        "FRACTIONS", "0.01,0.02,0.05,0.1,0.2,0.4,0.7,1.0").split(",")]
    k = int(os.environ.get("K", "10"))
    max_eval = int(os.environ.get("MAX_EVAL", "1000"))
    include_als = os.environ.get("NO_ALS", "") == ""

    # WARM=random (default): nested random subsampling of interactions — each
    # level ADDS data, so profiles warm from cold (1-2 items at 1%) to full.
    # Isolates DATA DENSITY with a stable real-user eval population, avoiding
    # the chronological-prefix artifact (early prefixes lack the future users,
    # so only non-personalized popularity scores). WARM=chrono = earliest prefix.
    mode = os.environ.get("WARM", "random")
    split = load_split(dataset)
    train, test = split.train, split.test
    train_by = train.groupby("entity_id", sort=False)["item_id"].apply(lambda s: set(s))
    test_by = test.groupby("entity_id", sort=False)["item_id"].apply(lambda s: set(s))
    # Eval population = real users (in full train) with held-out test items;
    # fixed across all warmth levels.
    eval_all = sorted(set(train_by.index) & set(test_by.index))
    step = max(1, len(eval_all) // max_eval)
    eval_entities = eval_all[::step][:max_eval]
    catalog = int(train["item_id"].nunique())
    order = np.random.default_rng(0).permutation(len(train))
    log(f"{dataset}: warm={mode} train {len(train):,} test {len(test):,} "
        f"eval_entities {len(eval_entities)} catalog {catalog:,} fractions {fractions}")

    rows = []
    for frac in fractions:
        n = int(round(len(train) * frac))
        if mode == "chrono":
            sub = train.iloc[:n].reset_index(drop=True)
        else:
            sub = train.iloc[order[:n]].reset_index(drop=True)
        owned_sub = sub.groupby("entity_id", sort=False)["item_id"].apply(lambda s: set(s))
        n_nonempty = sum(1 for e in eval_entities if test_by.get(e, set()) - owned_sub.get(e, set()))
        for model in build_models(include_als):
            t0 = time.perf_counter()
            try:
                model.fit(sub)
            except Exception as e:  # noqa: BLE001
                log(f"  frac={frac} {model.name} FIT FAILED: {type(e).__name__}: {e}")
                continue
            fit_s = time.perf_counter() - t0
            per, lat = [], []
            for ent in eval_entities:
                rel = test_by.get(ent, set()) - owned_sub.get(ent, set())
                r0 = time.perf_counter()
                recs = model.recommend(ent, n=k)
                lat.append((time.perf_counter() - r0) * 1000.0)
                per.append((recs, rel))
            m = aggregate(per, catalog_size=catalog, k=k)
            row = {"fraction": frac, "n_train": int(len(sub)),
                   "n_train_items": int(sub["item_id"].nunique()),
                   "n_eval_nonempty": n_nonempty,
                   "model": model.name, "fit_seconds": round(fit_s, 3),
                   "p50_ms": round(float(np.percentile(lat, 50)), 3),
                   "recall@k": round(m.recall_at_k, 4), "ndcg@k": round(m.ndcg_at_k, 4),
                   "mrr": round(m.mrr, 4), "hit_rate": round(m.hit_rate, 4)}
            rows.append(row)
            log(f"  frac={frac:<5} {model.name:14s} fit={fit_s:6.1f}s "
                f"ndcg={row['ndcg@k']:.4f} recall={row['recall@k']:.4f} p50={row['p50_ms']:.1f}ms")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"warming_{dataset}.json"
    out.write_text(json.dumps({"dataset": dataset, "k": k, "catalog": catalog,
                               "n_eval": len(eval_entities), "rows": rows}, indent=2) + "\n")
    log(f"[wrote] {out}")


if __name__ == "__main__":
    main()
