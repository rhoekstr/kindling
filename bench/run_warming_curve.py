"""Warming curve: kindling vs standard algorithms as a dataset accumulates
interactions ("warms"). Feeds the growth-curve grid (bench/plot_growth_curves.py).

Thesis: kindling's closed-form, no-training stack (wilson cooc / EASE + channels,
auto-gated) delivers strong recommendations from little data and fits in seconds,
while trained latent models (implicit ALS) need more data + iterations to
converge. On the cold/early end kindling should lead the *personalized* baselines
on accuracy AND speed, and pull further ahead as data warms.

For each subsample fraction of train (WARM=random density default; WARM=chrono =
earliest prefix), fit every model and evaluate on the SAME held-out test window
over a FIXED eval population (entities in full train∩test). Records, per
(fraction, model): fit_seconds, recommend p50 latency, recall@k, ndcg@k. This is
cold-as-in-data-scarce-SYSTEM (cf. run_user_warmth.py for cold-USER buckets).

Models (MODELS env, comma-sep): kindling · ease (kindling base alone) ·
popularity · item_item_knn · implicit_als. MERGE=1 keeps existing rows for
(fraction, model) pairs not regenerated — so a line or the cold tail can be added
without re-running the slow baselines.

Run: DATASET=movielens-1m MODELS=ease MERGE=1 \
     PYTHONPATH=src .venv/bin/python bench/run_warming_curve.py
"""
from __future__ import annotations

import json
import os
import time
from functools import partial
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
    full amazon-book (357k) OOMs / trips the kill wall on this box — and the
    real-retail H&M log (validate_hm's loader, kagglehub cache)."""
    if name == "amazon-book-academic":
        from kindling.benchmarks.comparison import _load_academic_split
        b = Path("~/.cache/kindling/amazon-book").expanduser()
        return _load_academic_split(b / "train.txt", b / "test.txt",
                                    name=name, action_type="rate")
    if name == "hm":
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import validate_hm
        train, test, articles = validate_hm._load()
        return SimpleNamespace(train=train, test=test, items=articles)
    return _load_dataset(name, test_fraction=tf)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Kindling:
    name = "kindling"

    def fit(self, df: pd.DataFrame):
        from kindling import Engine
        self._e = Engine(retrieval_budget=500, random_state=0)
        self._e.fit(df)
        return self

    def recommend(self, entity_id, n: int = 10):
        return [r.item_id for r in self._e.recommend(entity_id, n=n)]


class Ease:
    """kindling's closed-form base alone — channels + gate off. Isolates the
    item-item base from kindling's blend, so the grid shows how much the
    channels add. The base auto-resolves to EASE on the reference catalogs
    (≤20k core items: ml1m / beauty / steam) and to wilson-cooc on larger ones
    (book), so this is the EASE baseline wherever EASE is feasible — forcing
    EASE on a 90k-item catalog would be an O(n³) blow-up."""

    name = "ease"

    def fit(self, df: pd.DataFrame):
        from kindling import Engine
        self._e = Engine(
            retrieval_budget=500, random_state=0,
            trend_alpha=0.0, user_cf_alpha=0.0, last_item_alpha=0.0,
            transition_alpha=0.0, channel_gate=False,
        )
        self._e.fit(df)
        return self

    def recommend(self, entity_id, n: int = 10):
        return [r.item_id for r in self._e.recommend(entity_id, n=n)]


_REGISTRY = {
    "kindling": Kindling,
    "ease": Ease,
    "popularity": PopularityBaseline,
    "item_item_knn": partial(ItemItemKNN, k_neighbors=200),
    "implicit_als": partial(ImplicitALSBaseline, factors=64, iterations=15),
}


def build_models(which: list[str]):
    return [_REGISTRY[m]() for m in which if m in _REGISTRY]


def main() -> None:
    dataset = os.environ.get("DATASET", "movielens-1m")
    fractions = [float(x) for x in os.environ.get(
        "FRACTIONS", "0.01,0.02,0.04,0.08,0.15,0.3,0.5,0.75,1.0").split(",")]
    k = int(os.environ.get("K", "10"))
    max_eval = int(os.environ.get("MAX_EVAL", "1000"))
    which = os.environ.get("MODELS", "kindling,ease,popularity,item_item_knn,implicit_als").split(",")
    merge = os.environ.get("MERGE", "") != ""

    # WARM=random (default): nested random subsampling of interactions — each
    # level ADDS data, so profiles warm from cold (1-2 items at 1%) to full.
    # Isolates DATA DENSITY with a stable real-user eval population, avoiding
    # the chronological-prefix artifact (early prefixes lack the future users,
    # so only non-personalized popularity scores). WARM=chrono = earliest prefix.
    mode = os.environ.get("WARM", "random")
    split = load_split(dataset)
    train, test = split.train, split.test
    train_by = train.groupby("entity_id", sort=False)["item_id"].apply(set)
    test_by = test.groupby("entity_id", sort=False)["item_id"].apply(set)
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
        n = round(len(train) * frac)
        if mode == "chrono":
            sub = train.iloc[:n].reset_index(drop=True)
        else:
            sub = train.iloc[order[:n]].reset_index(drop=True)
        owned_sub = sub.groupby("entity_id", sort=False)["item_id"].apply(set)
        n_nonempty = sum(1 for e in eval_entities if test_by.get(e, set()) - owned_sub.get(e, set()))
        for model in build_models(which):
            t0 = time.perf_counter()
            try:
                model.fit(sub)
            except Exception as e:
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
            row = {"fraction": frac, "n_train": len(sub),
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
    if merge and out.exists():
        # Keep existing rows except those for the (fraction, model) pairs just
        # regenerated; append the fresh ones. Lets us add a model line or fill
        # the cold tail without re-running the slow baselines.
        prev = json.loads(out.read_text())["rows"]
        fresh = {(r["fraction"], r["model"]) for r in rows}
        rows = [r for r in prev if (r["fraction"], r["model"]) not in fresh] + rows
        rows.sort(key=lambda r: (r["fraction"], r["model"]))
    out.write_text(json.dumps({"dataset": dataset, "k": k, "catalog": catalog,
                               "n_eval": len(eval_entities), "rows": rows}, indent=2) + "\n")
    log(f"[wrote] {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
