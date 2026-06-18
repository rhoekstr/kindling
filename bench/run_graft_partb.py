"""Part B: does grafting cold/barely-seen items' metadata edges into the cooc
graph lift their recall — and where does it help vs hurt the rest of the catalog?

Mechanism: for each ELIGIBLE item (train count <= threshold, has metadata),
synthesize symmetric cooc edges to its top-k metadata-similar WARM items, weighted
on the observed-cooc scale, then score every item by row-sum over the user's owned
set (the cooc base). Cold items become reachable through the SAME scorer as warm.

Two questions baked in:
  - eligibility sweep: graft items with count <= T for T in {3,5,10,20}. Do we
    benefit only from grafting the very cold, or larger bands too? Segmented
    recall by held-out warmth tier shows where lift/regression lands.
  - distributional threshold: T as an occurrence PERCENTILE (adapts to dataset
    density) vs absolute counts.

Arms: no_graft (baseline) / naive (uniform synthetic weight) / calibrated
(weight = metadata_sim). Net effect = overall NDCG + per-tier recall + coverage.

Run: FPR_DATASETS=steam FPR_META=multihot .venv/bin/python bench/run_graft_partb.py
Env: GRAFT_TOPK, GRAFT_CAP (synthetic weight cap ratio vs max observed),
     GRAFT_THRESHOLDS (csv abs counts), GRAFT_PCTILE (csv percentiles), GRAFT_K.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from kindling.graph.cooc_transform import apply_cooc_transform
from kindling.item_features import ItemFeatureExtractor
from run_graft_probe import load

REPORT = Path(__file__).parent / "reports" / "graft_partb.json"
K = 20
TIERS = {"1-2": (1, 3), "3-5": (3, 6), "6-10": (6, 11), "11-20": (11, 21), "21+": (21, 10**9)}


def build(name, meta_mode, subsample=None):
    train, test, items = load(name)
    if subsample and train["entity_id"].nunique() > subsample:
        rng = np.random.default_rng(0)
        keep = set(rng.choice(train["entity_id"].unique(), subsample, replace=False).tolist())
        train = train[train["entity_id"].isin(keep)]
        test = test[test["entity_id"].isin(keep)]
        print(f"  subsampled to {subsample} users -> {train['item_id'].nunique()} items")
    item_ids = pd.Index(train["item_id"].unique())
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    n_items = len(item_ids)
    iidx = train["item_id"].map(item_to_idx).to_numpy()
    uidx = pd.factorize(train["entity_id"])[0]
    d = np.bincount(iidx, minlength=n_items).astype(np.float64)

    feat = ItemFeatureExtractor().fit_transform(items, item_to_idx, n_items)
    F = sp.csr_matrix((feat.data, feat.indices, feat.indptr), shape=(n_items, feat.n_features))
    has = np.diff(F.indptr) > 0

    S = sp.csr_matrix((np.ones(len(iidx), np.float32), (uidx, iidx)),
                      shape=(int(uidx.max()) + 1, n_items))
    S.data[:] = 1.0
    S.sum_duplicates()
    S.data[:] = 1.0
    C = (S.T @ S).tocoo()
    m = C.row != C.col
    rows, cols = C.row[m], C.col[m]
    co = C.data[m].astype(np.float64)
    # Observed edges = wilson-normalized cooc (the shipped base), so grafted and
    # observed weights live on the same scale.
    obs = apply_cooc_transform(co.astype(np.float32), cols.astype(np.int32),
                               _indptr_from_rows(rows, n_items), d, int(S.shape[0]), "wilson")
    Cobs = sp.csr_matrix((obs, (rows, cols)), shape=(n_items, n_items))
    max_obs = float(obs.max())

    tb = train.groupby("entity_id", sort=False)["item_id"].apply(set)
    te = test.groupby("entity_id", sort=False)["item_id"].apply(set)
    cand = sorted(set(tb.index) & set(te.index))
    if len(cand) > 2000:
        cand = cand[:: len(cand) // 2000][:2000]
    return dict(n_items=n_items, d=d, F=F, has=has, Cobs=Cobs, max_obs=max_obs,
                item_to_idx=item_to_idx, tb=tb, te=te, cand=cand)


def _indptr_from_rows(rows, n):
    """CSR indptr from sorted-by-row COO rows (S.T@S COO is row-major)."""
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.add.at(indptr, rows + 1, 1)
    return np.cumsum(indptr).astype(np.int32)


def graft(g, threshold, mode, topk, cap, rng):
    n_items, d, F, has = g["n_items"], g["d"], g["F"], g["has"]
    eligible = np.where((d >= 1) & (d <= threshold) & has)[0]
    warm = np.where((d > threshold) & has)[0]
    if len(eligible) == 0 or len(warm) == 0:
        return g["Cobs"]
    Fw = F[warm]
    er, ec, ew = [], [], []
    for s in range(0, len(eligible), 512):
        blk = eligible[s:s + 512]
        sims = np.asarray((F[blk] @ Fw.T).todense())  # (blk, n_warm)
        for bi, c in enumerate(blk):
            row = sims[bi]
            if topk < len(row):
                top = np.argpartition(-row, topk)[:topk]
            else:
                top = np.arange(len(row))
            top = top[row[top] > 0]
            for w in top:
                wt = (row[w] if mode == "calibrated" else 1.0) * cap * g["max_obs"]
                er.append(c); ec.append(int(warm[w])); ew.append(wt)
    syn = sp.csr_matrix((np.array(ew + ew), (np.array(er + ec), np.array(ec + er))),
                        shape=(n_items, n_items))  # symmetric
    return (g["Cobs"] + syn).tocsr()


def evaluate(g, Cg):
    d, item_to_idx = g["d"], g["item_to_idx"]
    n_items = g["n_items"]
    idisc = 1.0 / np.log2(np.arange(2, K + 2))
    tier_hit = {t: 0 for t in TIERS}
    tier_tot = {t: 0 for t in TIERS}
    ndcgs = []
    rec_items = set()
    for e in g["cand"]:
        owned = np.array([item_to_idx[i] for i in g["tb"][e] if i in item_to_idx])
        rel = [item_to_idx[i] for i in (g["te"][e] - g["tb"][e]) if i in item_to_idx]
        if len(owned) == 0:
            continue
        scores = np.asarray(Cg[:, owned].sum(axis=1)).ravel()
        scores[owned] = -np.inf
        top = np.argpartition(-scores, K)[:K]
        top = top[np.argsort(-scores[top])]
        rec_items.update(top.tolist())
        if not rel:
            continue
        topset = set(top.tolist())
        relset = set(rel)
        g_top = np.array([1.0 if int(t) in relset else 0.0 for t in top])
        if g_top.sum() > 0:
            dcg = float((g_top * idisc).sum())
            ideal = float(idisc[: min(len(rel), K)].sum())
            ndcgs.append(dcg / ideal)
        else:
            ndcgs.append(0.0)
        for r in rel:
            for t, (lo, hi) in TIERS.items():
                if lo <= d[r] < hi:
                    tier_tot[t] += 1
                    if r in topset:
                        tier_hit[t] += 1
                    break
    return {
        "ndcg": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        "coverage": round(len(rec_items) / n_items, 4),
        **{f"rec_{t}": (round(tier_hit[t] / tier_tot[t], 4) if tier_tot[t] else None) for t in TIERS},
        **{f"n_{t}": tier_tot[t] for t in TIERS},
    }


def main():
    datasets = os.environ.get("FPR_DATASETS", "steam").split(",")
    meta_mode = os.environ.get("FPR_META", "multihot")
    topk = int(os.environ.get("GRAFT_TOPK", "20"))
    cap = float(os.environ.get("GRAFT_CAP", "1.0"))
    subsample = int(os.environ.get("GRAFT_SUBSAMPLE", "0")) or None
    pctiles = os.environ.get("GRAFT_PCTILE", "")
    abs_thresholds = [int(x) for x in os.environ.get("GRAFT_THRESHOLDS", "3,10,20").split(",")]
    out = []
    for name in datasets:
        print(f"\n########## {name} ##########")
        g = build(name, meta_mode, subsample)
        d = g["d"]
        if pctiles:
            # distributional threshold: density-relative occurrence percentiles.
            bands = [(f"p{p}(<={int(np.percentile(d[d >= 1], float(p)))})",
                      int(np.percentile(d[d >= 1], float(p)))) for p in pctiles.split(",")]
        else:
            bands = [(f"T{t}", t) for t in abs_thresholds]
        rng = np.random.default_rng(0)
        base = evaluate(g, g["Cobs"])
        print(f"  {'no_graft':22s} ndcg={base['ndcg']:.4f} cov={base['coverage']:.3f} | "
              + " ".join(f"{t}={base['rec_' + t]}" for t in TIERS))
        rows = [{"arm": "no_graft", **base}]
        for label, T in bands:
            for mode in ("naive", "calibrated"):
                Cg = graft(g, T, mode, topk, cap, rng)
                r = evaluate(g, Cg)
                rows.append({"arm": f"{mode}_{label}", "threshold": T, **r})
                print(f"  {mode + '_' + label:22s} ndcg={r['ndcg']:.4f} cov={r['coverage']:.3f} | "
                      + " ".join(f"{t}={r['rec_' + t]}" for t in TIERS))
        out.append({"dataset": name, "k": K, "topk": topk, "cap": cap,
                    "subsample": subsample, "bands": [b[0] for b in bands], "rows": rows})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[wrote] {REPORT}")


if __name__ == "__main__":
    main()
