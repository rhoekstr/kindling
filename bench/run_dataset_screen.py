"""§7.6 dataset screen: is a candidate a viable home for cooc-embedding imputation?

Cheap go/no-go on the four properties that decide whether embedding imputation
can bank a cold win (no engine fit, no architecture work):

  1. catalog size       >20k → cooc-base path (clean scale-match home)
  2. content-coherence  metadata->cooc mapping-R^2 (reuses cooc_impute.fit_impute,
                        the production gate metric) — book 0.058 / steam 0.077 /
                        ml1m-genres 0.084 / ml-25m-genome 0.102 are the references
  3. warm-dominated     cold(<5) share of catalog — cold-DOMINATED floods (book)
  4. cold tail          enough warmth-0 held-out demand to measure recovery

Recency strength comes from run_cold_coverage.py (steam strong / movies weak).

Run: DATASET=ml-25m-unrestricted .venv/bin/python bench/run_dataset_screen.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from kindling.graph.cooc_impute import WARM_MIN, fit_impute
from kindling.item_features import ItemFeatureExtractor

REPORT_DIR = Path(__file__).parent / "reports"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_ml25m(content: str, sub_users: int):
    """content: 'genres' (all movies) or 'genome' (genome-covered subset)."""
    D = Path("~/.cache/kindling/ml-25m").expanduser()
    rng = np.random.default_rng(0)
    r = pd.read_csv(D / "ratings.csv", usecols=["userId", "movieId", "timestamp"],
                    dtype={"userId": np.int32, "movieId": np.int32, "timestamp": np.int64})
    u = r["userId"].unique()
    if len(u) > sub_users:
        r = r[r["userId"].isin(set(rng.choice(u, sub_users, replace=False).tolist()))]
    if content == "genome":
        g = pd.read_csv(D / "genome-scores.csv")
        gmovies = set(g["movieId"].unique().tolist())
        r = r[r["movieId"].isin(gmovies)]
        gk = g[g["relevance"] >= 0.5]
        tags = gk.groupby("movieId")["tagId"].apply(lambda s: "|".join(f"t{int(t)}" for t in s))
        meta = pd.DataFrame({"item_id": tags.index.values, "tags": tags.values})
        ccols = ["tags"]
    else:
        mv = pd.read_csv(D / "movies.csv").rename(columns={"movieId": "item_id"})
        meta = mv[["item_id", "genres"]]
        ccols = ["genres"]
    cut = r["timestamp"].quantile(0.90)
    train = r[r["timestamp"] <= cut].rename(columns={"userId": "entity_id", "movieId": "item_id"})
    test = r[r["timestamp"] > cut].rename(columns={"userId": "entity_id", "movieId": "item_id"})
    return train, test, meta, ccols


def main() -> None:
    dataset = os.environ.get("DATASET", "ml-25m-unrestricted")
    sub = int(os.environ.get("SUB_USERS", "40000"))
    content = "genome" if dataset.endswith("genome") else "genres"
    train, test, meta, ccols = load_ml25m(content, sub)

    item_ids = pd.Index(train["item_id"].unique())
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    n_items = len(item_ids)
    uidx = pd.factorize(train["entity_id"])[0]
    iidx = train["item_id"].map(item_to_idx).to_numpy()
    n_users = int(uidx.max()) + 1
    counts = np.bincount(iidx, minlength=n_items).astype(np.float64)
    n_warm = int((counts >= WARM_MIN).sum())
    cold_share = float((counts < WARM_MIN).mean())

    test_items = set(test["item_id"].unique())
    cold_demand = len(test_items - set(item_ids))

    # cooc over train items + content CSR over train items (mapping uses warm).
    X = sp.csr_matrix((np.ones(len(iidx), np.float32), (uidx, iidx)), shape=(n_users, n_items))
    X.data[:] = 1.0
    X.sum_duplicates()
    X.data[:] = 1.0
    t = time.perf_counter()
    G = (X.T @ X).tocsr()
    feats = ItemFeatureExtractor().fit_transform(
        meta[meta["item_id"].isin(set(item_ids))][["item_id", *ccols]], item_to_idx, n_items
    )
    F = sp.csr_matrix((feats.data, feats.indices, feats.indptr),
                      shape=(n_items, feats.n_features))
    model = fit_impute(G.data.astype(np.float32), G.indices.astype(np.int32),
                       G.indptr.astype(np.int32), F, counts, n_users, n_items=n_items)

    out = {
        "dataset": dataset, "content": content, "n_train_items": n_items,
        ">20k_cooc_base": n_items > 20_000, "n_warm(>=5)": n_warm,
        "cold_share_of_catalog": round(cold_share, 3),
        "cold_demand(warmth0_heldout)": cold_demand,
        "n_content_features": feats.n_features,
        "mapping_r2": model.r2, "neighbor_recovery@10": model.neighbor_recovery,
    }
    log(f"screen {dataset} ({content}) in {time.perf_counter()-t:.0f}s:")
    for k, v in out.items():
        log(f"    {k:30s} {v}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p = REPORT_DIR / f"screen_{dataset}.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    log(f"[wrote] {p}")


if __name__ == "__main__":
    main()
