"""Real-world validation: kindling vs standard baselines on H&M.

H&M Personalized Fashion Recommendations (Kaggle competition): a real
retail purchase log — ~31M transactions, ~1.37M customers, ~105k articles —
with timestamps AND rich, readable product metadata (product type, colour,
department, garment group, text description). It's the one dataset that can
exercise the content / cold-slot path on real metadata, plus a fresh-domain
model comparison.

Protocol: realistic tier — chronological split, full-catalog ranking, k=12
(the competition's metric is MAP@12), sliced by user history length. kindling
vs popularity / item-kNN / implicit ALS / BPR. A recent time window keeps the
fit tractable; set HM_START to change it.

Run: python bench/validate_hm.py
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender

from kindling import Engine
from kindling.benchmarks.metrics import aggregate

HM = Path(
    os.environ.get(
        "HM_DIR",
        Path.home() / ".cache/kagglehub/competitions/h-and-m-personalized-fashion-recommendations",
    )
)
K = 12
N_EVAL = 8000
TEST_FRACTION = 0.1
HM_START = os.environ.get("HM_START", "2020-06-01")  # recent window for tractability
BUCKETS = ("1-4", "5-19", "20+", "all")
META_COLS = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "department_name",
    "index_name",
    "garment_group_name",
]


def _bucket(h: int) -> str:
    return "1-4" if h <= 4 else ("5-19" if h < 20 else "20+")


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tx = pd.read_csv(
        HM / "transactions_train.csv",
        usecols=["t_dat", "customer_id", "article_id"],
        parse_dates=["t_dat"],
    )
    tx = tx[tx["t_dat"] >= pd.Timestamp(HM_START)].copy()
    tx = tx.rename(
        columns={"customer_id": "entity_id", "article_id": "item_id", "t_dat": "timestamp"}
    )
    cut = tx["timestamp"].quantile(1 - TEST_FRACTION)
    train = tx[tx["timestamp"] <= cut].copy()
    test = tx[tx["timestamp"] > cut].copy()
    articles = pd.read_csv(HM / "articles.csv", usecols=["article_id", *META_COLS]).rename(
        columns={"article_id": "item_id"}
    )
    return train, test, articles


def main() -> None:
    t = time.perf_counter()
    train, test, articles = _load()
    print(
        f"loaded {time.perf_counter() - t:.0f}s  window>={HM_START}  "
        f"train={len(train):,}  users={train.entity_id.nunique():,}  "
        f"items={train.item_id.nunique():,}  test={len(test):,}",
        flush=True,
    )

    train_owned = train.groupby("entity_id")["item_id"].apply(lambda x: set(x.tolist()))
    test_by = test.groupby("entity_id")["item_id"].apply(lambda x: set(x.tolist()))
    train_users = set(train_owned.index)
    eval_users = [u for u in test_by.index if u in train_users and (test_by[u] - train_owned[u])]
    random.seed(0)
    eval_users = random.sample(eval_users, min(N_EVAL, len(eval_users)))
    print(f"eval users with held-out: {len(eval_users):,}", flush=True)

    users = sorted(train_users)
    items = sorted(train["item_id"].unique())
    u_row = {u: i for i, u in enumerate(users)}
    i_col = {it: j for j, it in enumerate(items)}
    col_item = np.array(items, dtype=object)
    ui = sp.csr_matrix(
        (
            np.ones(len(train), dtype=np.float32),
            (train["entity_id"].map(u_row).to_numpy(), train["item_id"].map(i_col).to_numpy()),
        ),
        shape=(len(users), len(items)),
    )
    pop_order = train["item_id"].value_counts().index.tolist()

    models: dict[str, object] = {}
    t = time.perf_counter()
    eng = Engine(random_state=0, open_catalog=False)
    eng.fit(train, item_metadata=articles)
    print(
        f"fit kindling {time.perf_counter() - t:.0f}s  base={eng.activation_plan.base_scorer}  "
        f"channels={eng.activation_plan.active_channels}",
        flush=True,
    )
    for name, ctor in [
        ("item-kNN", lambda: CosineRecommender(K=200)),
        ("ALS", lambda: AlternatingLeastSquares(factors=64, iterations=15, random_state=0)),
        ("BPR", lambda: BayesianPersonalizedRanking(factors=64, iterations=80, random_state=0)),
    ]:
        t = time.perf_counter()
        m = ctor()
        m.fit(ui, show_progress=False)
        models[name] = m
        print(f"fit {name} {time.perf_counter() - t:.0f}s", flush=True)

    names = ["kindling", "popularity", "item-kNN", "ALS", "BPR"]
    res: dict[str, dict[str, list]] = {m: {b: [] for b in BUCKETS} for m in names}

    def _imp(name: str, u: object) -> list:
        r = u_row[u]
        ids, _ = models[name].recommend(r, ui[r], N=K, filter_already_liked_items=True)
        return [col_item[c] for c in ids]

    for u in eval_users:
        owned = train_owned[u]
        held = test_by[u] - owned
        b = _bucket(len(owned))
        recs = {
            "kindling": [x.item_id for x in eng.recommend(entity_id=u, n=K)],
            "popularity": [i for i in pop_order if i not in owned][:K],
            "item-kNN": _imp("item-kNN", u),
            "ALS": _imp("ALS", u),
            "BPR": _imp("BPR", u),
        }
        for m, rec in recs.items():
            for key in (b, "all"):
                res[m][key].append((rec, held))

    n_items = eng._state.n_items
    print(f"\nH&M — NDCG@{K} by user history (n_items={n_items:,})\n")
    print(f"{'bucket':<7}{'n':>7}" + "".join(f"{m:>11}" for m in names))
    out: dict = {
        "dataset": "h-and-m",
        "window_start": HM_START,
        "metric": f"ndcg@{K}",
        "buckets": {},
    }
    for b in BUCKETS:
        n = len(res["kindling"][b])
        if not n:
            continue
        vals = {
            m: round(float(aggregate(res[m][b], catalog_size=n_items, k=K).ndcg_at_k), 4)
            for m in names
        }
        print(f"{b:<7}{n:>7}" + "".join(f"{vals[m]:>11.4f}" for m in names))
        out["buckets"][b] = {"n": n, **vals}
    Path("bench/reports/validate_hm.json").write_text(json.dumps(out, indent=2))
    print("\nWrote bench/reports/validate_hm.json")


if __name__ == "__main__":
    main()
