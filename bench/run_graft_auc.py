"""Cold/barely-seen TRANSFER-AUC: metadata's prediction ability where grafting
actually operates.

The warm presence-AUC (run earlier) asked: can metadata retrieve a warm item's
TRAIN cooc partners? Grafting's real job is colder and forward-looking: for a
low-occurrence item, can metadata retrieve the warm items it co-occurs with in
the TEST window — partners it has little/no train cooc to lean on?

This measures, per item-warmth tier (train interaction count), a popularity-
controlled retrieval AUC:
  anchor c (train count in tier, has metadata)
  positives = train items co-held by c's TEST users  (c should graft to these)
  negatives = degree-matched (train popularity) non-partners with metadata
  AUC = P(metadata_sim(c, partner) > metadata_sim(c, matched non-partner))

Tiers span the spectrum from barely-seen (1-2) to warm (21-50, reference).
AUC 0.5 = metadata no better than chance at finding a cold item's partners.

Run: FPR_DATASETS=steam,amazon-book-chrono FPR_META=multihot .venv/bin/python bench/run_graft_auc.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from kindling.item_features import ItemFeatureExtractor
from run_graft_probe import _combined_text, load

REPORT = Path(__file__).parent / "reports" / "graft_auc.json"
TIERS = {"1-2": (1, 3), "3-5": (3, 6), "6-10": (6, 11), "11-20": (11, 21), "21-50": (21, 51)}


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def deg_matched_neg(pos, cand_sorted, cand_d_sorted, d, banned, rng, window=40):
    out = []
    for p in pos:
        i = int(np.searchsorted(cand_d_sorted, d[p]))
        lo, hi = max(0, i - window), min(len(cand_sorted), i + window)
        for _ in range(8):
            k = int(cand_sorted[rng.integers(lo, hi)]) if hi > lo else -1
            if k >= 0 and k not in banned:
                out.append(k)
                break
    return np.array(out, dtype=np.int64)


def probe(name, meta_mode="multihot", per_tier=400, min_pos=3, max_pos=200, seed=0):
    train, test, items = load(name)
    item_ids = pd.Index(train["item_id"].unique())
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    item_ids_arr = item_ids.to_numpy()
    n_items = len(item_ids)

    users = pd.Index(pd.concat([train["entity_id"], test["entity_id"]]).unique())
    u2i = {u: i for i, u in enumerate(users)}
    n_users = len(users)

    tr_u = train["entity_id"].map(u2i).to_numpy()
    tr_i = train["item_id"].map(item_to_idx).to_numpy()
    Strain = sp.csr_matrix((np.ones(len(tr_i), np.float32), (tr_u, tr_i)), shape=(n_users, n_items))
    Strain.data[:] = 1.0
    Strain.sum_duplicates()
    Strain.data[:] = 1.0
    d = np.bincount(tr_i, minlength=n_items).astype(np.float64)

    test_in = test[test["item_id"].isin(item_to_idx)]
    te_u = test_in["entity_id"].map(u2i).to_numpy()
    te_i = test_in["item_id"].map(item_to_idx).to_numpy()
    Stest = sp.csc_matrix((np.ones(len(te_i), np.float32), (te_u, te_i)), shape=(n_users, n_items))
    Stest.data[:] = 1.0
    Stest.sum_duplicates()
    Stest.data[:] = 1.0
    test_count = np.bincount(te_i, minlength=n_items)

    if meta_mode == "dense":
        from kindling.dense_content import embed_texts

        text_by_id = _combined_text(items)
        has = np.array([bool(text_by_id.get(it, "")) for it in item_ids_arr])
        V = np.zeros((n_items, 384), np.float32)
        widx = np.where(has)[0]
        emb = embed_texts([text_by_id.get(item_ids_arr[g], "") for g in widx])
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
        V[widx] = emb
        F = sp.csr_matrix(V)
        n_features = 384
    else:
        feat = ItemFeatureExtractor().fit_transform(items, item_to_idx, n_items)
        F = sp.csr_matrix((feat.data, feat.indices, feat.indptr), shape=(n_items, feat.n_features))
        has = np.diff(F.indptr) > 0
        n_features = feat.n_features

    meta_items = np.where(has & (d >= 1))[0]
    meta_sorted = meta_items[np.argsort(d[meta_items])]
    meta_d_sorted = d[meta_sorted]

    rng = np.random.default_rng(seed)
    tier_out = {}
    for tname, (lo, hi) in TIERS.items():
        cand = np.where(has & (d >= lo) & (d < hi) & (test_count > 0))[0]
        rng.shuffle(cand)
        aucs = []
        for c in cand:
            if len(aucs) >= per_tier:
                break
            coocrow = np.asarray((Stest[:, c].T @ Strain).todense()).ravel()
            partners = np.where(coocrow > 0)[0]
            partners = partners[(partners != c) & has[partners]]
            if len(partners) < min_pos:
                continue
            if len(partners) > max_pos:
                partners = rng.choice(partners, max_pos, replace=False)
            banned = set(np.where(coocrow > 0)[0].tolist()) | {int(c)}
            neg = deg_matched_neg(partners, meta_sorted, meta_d_sorted, d, banned, rng)
            if len(neg) == 0:
                continue
            cands = np.concatenate([partners, neg])
            sims = np.asarray((F[cands] @ F[c].T).todense()).ravel()
            aucs.append(auc(sims[: len(partners)], sims[len(partners):]))
        tier_out[tname] = {"auc": round(float(np.nanmean(aucs)), 3) if aucs else None,
                           "n_anchors": len(aucs)}

    out = {"dataset": name, "meta_mode": meta_mode, "n_items": int(n_items),
           "n_features": int(n_features), "tiers": tier_out}
    print(f"\n##### {name} [{meta_mode}] — cold/barely-seen transfer-AUC #####")
    print(f"  feats={n_features}")
    for t, v in tier_out.items():
        a = v["auc"]
        bar = "" if a is None else "  " + ("#" * int(max(0, (a - 0.5)) * 100))
        print(f"  tier {t:6s} train-count  AUC={str(a):5s}  (n={v['n_anchors']}){bar}")
    return out


def main():
    datasets = os.environ.get("FPR_DATASETS", "steam,amazon-book-chrono").split(",")
    meta_mode = os.environ.get("FPR_META", "multihot")
    out = []
    for ds in datasets:
        try:
            out.append(probe(ds, meta_mode=meta_mode))
        except Exception as e:  # noqa: BLE001
            print(f"[{ds}] FAILED: {type(e).__name__}: {e}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[wrote] {REPORT}")


if __name__ == "__main__":
    main()
