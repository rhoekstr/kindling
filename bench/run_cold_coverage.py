"""Stage-0 diagnostic for the demand-aware cold-extension policy (REFERENCE §7.1).

In open-catalog mode the engine admits metadata-only items as cold candidates,
but a RAM cap forces it to keep only the top-`cap` — currently in salesRank /
loader order. §7.1 asks whether a *demand-aware* selection (recency, content-
proximity to recent demand) would cover more of the actual cold demand.

Before building anything, this measures WHERE the bottleneck is — pure set
membership, no engine fit:

  cold demand   = test items with ZERO train interactions (warmth-0 held-out)
  pool          = metadata-only items (in item_metadata, not in train)
  ceiling       = |cold ∩ pool| / |cold|   (recoverable by ANY policy at cap=∞;
                  the rest of cold demand has no metadata → unreachable)
  coverage(f)   = |cold ∩ top-(f·|pool|) by policy| / |cold|

Policies: salesrank/meta-order (current), recency (release-date desc),
content (cosine to the recent-train-demand centroid), random (floor). The gap
between the current policy and the ceiling is the headroom a demand-aware
policy could win; whether recency/content close it tells us if it is worth
building, and on which datasets.

Run: DATASET=steam .venv/bin/python bench/run_cold_coverage.py
     DATASET=ml-25m  | amazon-book-chrono
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from kindling.item_features import ItemFeatureExtractor

REPORT_DIR = Path(__file__).parent / "reports"
FRACTIONS = [0.1, 0.25, 0.5, 1.0]
RECENT_WINDOW = 0.10  # last 10% of train (by time) = "recent demand"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(dataset: str):
    """Return (train, test, items_df, content_cols, date_col|None, rank_col|None)."""
    if dataset == "ml-25m":
        D = Path("~/.cache/kindling/ml-25m").expanduser()
        sub = int(os.environ.get("SUB_USERS", "40000"))
        rng = np.random.default_rng(0)
        r = pd.read_csv(D / "ratings.csv",
                        usecols=["userId", "movieId", "timestamp"],
                        dtype={"userId": np.int32, "movieId": np.int32,
                               "timestamp": np.int64})
        u = r["userId"].unique()
        if len(u) > sub:
            r = r[r["userId"].isin(set(rng.choice(u, sub, replace=False).tolist()))]
        cut = r["timestamp"].quantile(0.90)
        train = r[r["timestamp"] <= cut].rename(
            columns={"userId": "entity_id", "movieId": "item_id"})
        test = r[r["timestamp"] > cut].rename(
            columns={"userId": "entity_id", "movieId": "item_id"})
        mv = pd.read_csv(D / "movies.csv")  # movieId,title,genres
        mv = mv.rename(columns={"movieId": "item_id"})
        mv["year"] = mv["title"].str.extract(r"\((\d{4})\)").astype(float)
        return train, test, mv, ["genres"], "year", None
    from kindling.benchmarks.comparison import _load_dataset
    s = _load_dataset(dataset, test_fraction=0.1)
    if dataset == "steam":
        return s.train, s.test, s.items, ["genres", "tags"], "release_date", None
    if dataset == "amazon-book-chrono":
        return s.train, s.test, s.items, ["categories", "title", "brand"], None, "sales_rank"
    raise SystemExit(f"unsupported dataset {dataset}")


def coverage_curve(order: np.ndarray, pool_ids: np.ndarray, cold: set, n_cold: int):
    """order = pool indices ranked best→worst; return coverage at each fraction."""
    ranked = pool_ids[order]
    in_cold = np.array([pid in cold for pid in ranked])
    cum = np.cumsum(in_cold)
    out = {}
    for f in FRACTIONS:
        k = max(1, int(f * len(ranked)))
        out[f] = round(float(cum[k - 1]) / n_cold, 4)
    return out


def main() -> None:
    dataset = os.environ.get("DATASET", "steam")
    train, test, items, content_cols, date_col, rank_col = load(dataset)
    train_items = set(train["item_id"].unique())
    test_items = set(test["item_id"].unique())
    meta_items = pd.Index(items["item_id"].dropna().unique())

    cold = test_items - train_items                 # warmth-0 held-out demand
    pool = pd.Index(meta_items).difference(pd.Index(list(train_items)))
    pool_ids = pool.to_numpy()
    n_cold = len(cold)
    recoverable = sum(1 for c in cold if c in set(pool_ids.tolist()))
    log(f"{dataset}: train_items={len(train_items):,} cold_demand={n_cold:,} "
        f"meta_pool={len(pool_ids):,}  ceiling={recoverable / max(n_cold,1):.4f} "
        f"(={recoverable}/{n_cold} cold items are in metadata at all)")

    meta = items.drop_duplicates("item_id").set_index("item_id")
    rng = np.random.default_rng(0)
    results = {"dataset": dataset, "n_cold": n_cold, "meta_pool": len(pool_ids),
               "ceiling": round(recoverable / max(n_cold, 1), 4), "policies": {}}

    # ── current policy: salesRank asc (book) else loader/meta order ──
    if rank_col and rank_col in meta.columns:
        sr = pd.to_numeric(meta.loc[pool_ids, rank_col], errors="coerce").to_numpy()
        base_order = np.argsort(np.nan_to_num(sr, nan=np.inf), kind="stable")
        base_name = "salesrank"
    else:
        base_order = np.arange(len(pool_ids))  # loader/metadata order
        base_name = "meta-order"
    results["policies"][base_name] = coverage_curve(base_order, pool_ids, cold, n_cold)

    # ── random floor ──
    rand_order = rng.permutation(len(pool_ids))
    results["policies"]["random"] = coverage_curve(rand_order, pool_ids, cold, n_cold)

    # ── recency: release-date desc (newest first) ──
    if date_col and date_col in meta.columns:
        d = pd.to_datetime(meta.loc[pool_ids, date_col], errors="coerce")
        ts = d.astype("int64").to_numpy().astype(float)  # ns since epoch
        ts[d.isna().to_numpy()] = -np.inf                # NaT → oldest
        rec_order = np.argsort(-ts, kind="stable")
        results["policies"]["recency"] = coverage_curve(rec_order, pool_ids, cold, n_cold)

    # ── content-proximity to the recent-train-demand centroid ──
    t = time.perf_counter()
    all_ids = pd.Index(list(train_items)).append(pd.Index(pool_ids)).unique()
    item_to_idx = {it: i for i, it in enumerate(all_ids)}
    meta_for_feat = items[items["item_id"].isin(set(all_ids))][
        ["item_id", *content_cols]
    ].copy()
    feats = ItemFeatureExtractor().fit_transform(meta_for_feat, item_to_idx, len(all_ids))
    F = sp.csr_matrix((feats.data, feats.indices, feats.indptr),
                      shape=(len(all_ids), feats.n_features))
    # recent-demand centroid: items in the last RECENT_WINDOW of train by time.
    if "timestamp" in train.columns:
        thr = train["timestamp"].quantile(1 - RECENT_WINDOW)
        recent = train[train["timestamp"] >= thr]["item_id"]
    else:
        recent = train["item_id"]
    rc = recent.map(item_to_idx).dropna().astype(int).to_numpy()
    if len(rc):
        centroid = np.asarray(F[rc].mean(axis=0)).ravel()
        pool_rows = np.array([item_to_idx[p] for p in pool_ids])
        cont_scores = np.asarray(F[pool_rows] @ centroid).ravel()
        cont_order = np.argsort(-cont_scores, kind="stable")
        results["policies"]["content"] = coverage_curve(cont_order, pool_ids, cold, n_cold)
    log(f"content features: {feats.n_features} feats, centroid in {time.perf_counter()-t:.0f}s")

    # ── report ──
    log(f"coverage of cold demand (ceiling={results['ceiling']}):")
    hdr = "  policy        " + "  ".join(f"@{int(f*100):>3}%" for f in FRACTIONS)
    log(hdr)
    for pol, cov in results["policies"].items():
        log(f"  {pol:12s}  " + "  ".join(f"{cov[f]:.4f}" for f in FRACTIONS))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"cold_coverage_{dataset}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    log(f"[wrote] {out}")


if __name__ == "__main__":
    main()
