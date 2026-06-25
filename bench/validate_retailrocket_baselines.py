"""RetailRocket: kindling vs the standard recommender baselines.

Reproduces the warming-benchmark comparison (REFERENCE §3.5) on real
production churn data: kindling vs popularity, item-item kNN (cosine),
implicit ALS (the industry-standard trained MF), and BPR. Realistic tier —
no k-core, chronological split, full-catalog ranking, k=20, sliced by user
history length.

Run: python bench/validate_retailrocket_baselines.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender

from kindling import Engine
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate

K = 20
N_EVAL = 8000
BUCKETS = ("1-4", "5-19", "20+", "all")


def _bucket(h: int) -> str:
    return "1-4" if h <= 4 else ("5-19" if h < 20 else "20+")


def main() -> None:
    split = _load_dataset("retailrocket", 0.1)
    train, test = split.train, split.test
    train_owned = train.groupby("entity_id")["item_id"].apply(lambda x: set(x.tolist()))
    test_by = test.groupby("entity_id")["item_id"].apply(lambda x: set(x.tolist()))
    train_users = set(train_owned.index)
    eval_users = [u for u in test_by.index if u in train_users and (test_by[u] - train_owned[u])]
    random.seed(0)
    eval_users = random.sample(eval_users, min(N_EVAL, len(eval_users)))

    # --- shared sparse user x item matrix + index maps (for implicit models) ---
    users = sorted(train_users)
    items = sorted(train["item_id"].unique())
    u_row = {u: i for i, u in enumerate(users)}
    i_col = {it: j for j, it in enumerate(items)}
    col_item = np.array(items, dtype=object)
    rows = train["entity_id"].map(u_row).to_numpy()
    cols = train["item_id"].map(i_col).to_numpy()
    ui = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(users), len(items))
    )
    pop_order = train["item_id"].value_counts().index.tolist()

    # --- fit all models ---
    models: dict[str, object] = {}
    t = time.perf_counter()
    eng = Engine(persona_min_users=10**9, random_state=0, open_catalog=False)
    eng.fit(train)
    print(f"fit kindling {time.perf_counter() - t:.0f}s", flush=True)
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

    # --- per-model, per-bucket eval ---
    res: dict[str, dict[str, list]] = {
        m: {b: [] for b in BUCKETS} for m in ["kindling", "popularity", "item-kNN", "ALS", "BPR"]
    }

    def _imp_recs(name: str, u: object, owned: set) -> list:
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
            "item-kNN": _imp_recs("item-kNN", u, owned),
            "ALS": _imp_recs("ALS", u, owned),
            "BPR": _imp_recs("BPR", u, owned),
        }
        for m, rec in recs.items():
            for key in (b, "all"):
                res[m][key].append((rec, held))

    n_items = eng._state.n_items
    print(
        f"\nRetailRocket — NDCG@20 by user history (n_items={n_items:,}, "
        f"kindling channels={eng.activation_plan.active_channels})\n"
    )
    header = f"{'bucket':<7}{'n':>6}" + "".join(f"{m:>11}" for m in res)
    print(header)
    out: dict = {"dataset": "retailrocket", "metric": "ndcg@20", "buckets": {}}
    for b in BUCKETS:
        n = len(res["kindling"][b])
        if not n:
            continue
        cells, row_vals = "", {}
        for m in res:
            ndcg = round(float(aggregate(res[m][b], catalog_size=n_items, k=K).ndcg_at_k), 4)
            row_vals[m] = ndcg
            cells += f"{ndcg:>11.4f}"
        print(f"{b:<7}{n:>6}{cells}")
        out["buckets"][b] = {"n": n, **row_vals}
    Path("bench/reports/validate_retailrocket_baselines.json").write_text(json.dumps(out, indent=2))
    print("\nWrote bench/reports/validate_retailrocket_baselines.json")


if __name__ == "__main__":
    main()
