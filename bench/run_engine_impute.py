"""Engine-level validation of embedding imputation (Phase-1 guardrail).

The standalone mechanism was validated in run_ml25m_lift.py; this confirms the
WIRED engine reproduces it — i.e. that cold_impute through EngineV2's reserved
cold slots lifts cold-item recall at no warm cost, and beats the content-space
ranker it replaces.

Three arms on the SAME chronological split, segment-sliced by held-out item
warmth:
  off      cold_slots=0                       (production baseline, cold unscorable)
  content  cold_slots=1 cold_impute=content   (the mechanism being replaced)
  impute   cold_slots=1 cold_impute=impute    (embedding imputation)

Metadata = genome tags thresholded to multi-hot (relevance >= TAG_REL), the
representation a real deployment would feed the schema-inferring extractor.
Cold (recent) movies are metadata-only → open-catalog makes them candidates.

Run: SUB_USERS=8000 .venv/bin/python bench/run_engine_impute.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kindling.engine_v2 import EngineV2

D = Path("~/.cache/kindling/ml-25m").expanduser()
K = 20
TAG_REL = 0.5     # genome relevance threshold for a tag to count as present
COLD_MAX = 4      # held-out item is "cold" at <= this many train interactions
TIERS = {"cold_1-4": (1, 5), "5-19": (5, 20), "20+": (20, 10**9)}


def load(sub_users: int):
    rng = np.random.default_rng(0)
    r = pd.read_csv(D / "ratings.csv", usecols=["userId", "movieId", "timestamp"],
                    dtype={"userId": np.int32, "movieId": np.int32, "timestamp": np.int64})
    g = pd.read_csv(D / "genome-scores.csv")
    genome_movies = set(g["movieId"].unique().tolist())
    users = r["userId"].unique()
    if len(users) > sub_users:
        keep = set(rng.choice(users, sub_users, replace=False).tolist())
        r = r[r["userId"].isin(keep)]
    r = r[r["movieId"].isin(genome_movies)]

    cut = r["timestamp"].quantile(0.90)
    train = r[r["timestamp"] <= cut].copy()
    test = r[r["timestamp"] > cut].copy()
    for df in (train, test):
        df.rename(columns={"userId": "entity_id", "movieId": "item_id"}, inplace=True)

    # genome -> multi-hot tag string per movie (warm + cold, so cold items have
    # metadata and join the catalog via open_catalog).
    keep_movies = set(train["item_id"]) | set(test["item_id"])
    gk = g[(g["relevance"] >= TAG_REL) & (g["movieId"].isin(keep_movies))]
    tags = gk.groupby("movieId")["tagId"].apply(
        lambda s: "|".join(f"t{int(t)}" for t in s)
    )
    meta = pd.DataFrame({"item_id": tags.index.values, "tags": tags.values})
    return train, test, meta


def evaluate(eng, train_by, test_by, counts, item_to_idx, eval_users):
    idisc = 1.0 / np.log2(np.arange(2, K + 2))
    tier_hit = {t: 0 for t in TIERS}
    tier_tot = {t: 0 for t in TIERS}
    ndcgs = []
    for u in eval_users:
        rel = test_by.get(u, set()) - train_by.get(u, set())
        if not rel:
            continue
        recs = [r.item_id for r in eng.recommend(u, n=K)]
        recset = set(recs)
        gains = np.array([1.0 if it in rel else 0.0 for it in recs])
        denom = idisc[: min(len(rel), K)].sum()
        ndcgs.append(float((gains * idisc[: len(gains)]).sum()) / denom if denom else 0.0)
        for it in rel:
            ci = item_to_idx.get(it)
            c = counts[ci] if ci is not None and ci < len(counts) else 0
            for t, (lo, hi) in TIERS.items():
                if lo <= c < hi:
                    tier_tot[t] += 1
                    tier_hit[t] += int(it in recset)
                    break
    return {
        "ndcg": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        **{t: (round(tier_hit[t] / tier_tot[t], 4) if tier_tot[t] else None) for t in TIERS},
        **{f"n_{t}": tier_tot[t] for t in TIERS},
    }


def main():
    sub = int(os.environ.get("SUB_USERS", "8000"))
    print(f"[load] ml-25m subsample={sub} ...", flush=True)
    train, test, meta = load(sub)
    train_by = train.groupby("entity_id")["item_id"].apply(set).to_dict()
    test_by = test.groupby("entity_id")["item_id"].apply(set).to_dict()
    counts_s = train.groupby("item_id").size()
    eval_users = sorted(set(train_by) & set(test_by))
    rng = np.random.default_rng(1)
    rng.shuffle(eval_users)
    eval_users = eval_users[:2000]
    print(f"[data] train_items={train['item_id'].nunique()} cold_meta_items="
          f"{meta['item_id'].nunique() - train['item_id'].nunique()} eval_users={len(eval_users)}",
          flush=True)

    arms = {
        "off": dict(cold_slots=0),
        "content": dict(cold_slots=1, cold_impute="content"),
        "impute": dict(cold_slots=1, cold_impute="impute"),
    }
    for name, kw in arms.items():
        t0 = time.perf_counter()
        eng = EngineV2(**kw).fit(train, item_metadata=meta)
        item_to_idx = eng._state.item_to_idx
        counts = np.zeros(eng._state.n_items)
        for it, c in counts_s.items():
            ci = item_to_idx.get(it)
            if ci is not None:
                counts[ci] = c
        res = evaluate(eng, train_by, test_by, counts, item_to_idx, eval_users)
        p = eng._state.profile
        extra = ""
        if "cold_impute_r2" in p:
            extra = f" r2={p['cold_impute_r2']} nbr={p['cold_impute_neighbor_recovery']} active={p['cold_impute_active']}"
        print(f"  {name:8s} base={p['base_scorer_used']} fit={time.perf_counter()-t0:.0f}s{extra}\n"
              f"           ndcg={res['ndcg']} | "
              + " ".join(f"{t}={res[t]}(n={res['n_'+t]})" for t in TIERS), flush=True)


if __name__ == "__main__":
    main()
