"""Phase-0 viability gate for metadata->cooc edge grafting (Stage 5 signal).

The linchpin of calibrated grafting is the Stage-5 effect model: does metadata
similarity predict cooc strength on WARM co-occurring pairs? If it does, a cold
item's synthetic edges can be denominated in cooc-weight units (the scale-match
the content blend never had). If it doesn't, calibrated synthetic weights are
noise and grafting is dead before we build it.

Pre-registered gate (fixed before running):
    Stage-5 ALIVE iff, on held-out warm co-occurring pairs, the metadata->cooc
    relationship clears  max(R^2_linear, R^2_binned) >= 0.10  OR
    Spearman(metadata_sim, cooc) >= 0.20  on the affinity (cosine-weight) target.

Multi-hot metadata (item_features). Datasets: steam + amazon-book-chrono.
Run: FPR_DATASETS=steam,amazon-book-chrono .venv/bin/python bench/run_graft_probe.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from kindling.item_features import ItemFeatureExtractor

REPORT = Path(__file__).parent / "reports" / "graft_probe.json"
GATE_R2, GATE_SPEARMAN = 0.10, 0.20


def load(name):
    if name == "steam":
        from kindling.loaders.steam import load_steam

        train, test, items = load_steam(test_fraction=0.1)
        return train, test, items
    if name == "amazon-book-chrono":
        from kindling.loaders.amazon_chrono import load_amazon_chrono, load_amazon_meta

        book = Path("~/.cache/kindling/amazon-book").expanduser()
        train, test = load_amazon_chrono(book / "reviews_Books_5.json.gz", cache_dir=book, test_fraction=0.1)
        items = load_amazon_meta(book / "meta_Books.json.gz", cache_dir=book,
                                 catalog=set(train["item_id"].unique()))
        return train, test, items
    raise ValueError(name)


def _r2(y, yhat):
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - yhat) ** 2).sum()) / ss if ss > 0 else 0.0


def effect(xtr, ytr, xte, yte):
    sp_ = float(spearmanr(xte, yte).statistic)
    b1, b0 = np.polyfit(xtr, ytr, 1)
    r2_lin = _r2(yte, b0 + b1 * xte)
    # nonparametric: quantile-binned mean of y over x (isotonic-ish ceiling)
    qs = np.quantile(xtr, np.linspace(0, 1, 11))
    edges = qs[1:-1]
    btr = np.digitize(xtr, edges)
    means = np.array([ytr[btr == b].mean() if (btr == b).any() else ytr.mean() for b in range(10)])
    r2_np = _r2(yte, means[np.digitize(xte, edges)])
    return {"spearman": round(sp_, 3), "r2_linear": round(r2_lin, 3), "r2_binned": round(r2_np, 3)}


def _combined_text(items):
    """One text string per item_id by concatenating all non-id columns."""
    items = items.drop_duplicates("item_id", keep="first")
    parts = []
    for col in items.columns:
        if col == "item_id":
            continue
        s = items[col]
        is_list = s.map(lambda v: isinstance(v, (list, tuple))).any()
        if is_list:
            parts.append(s.map(lambda v: " ".join(map(str, v)) if isinstance(v, (list, tuple)) else ""))
        elif s.dtype == object:  # text only — skip numeric (price, sales_rank, hours)
            parts.append(s.fillna("").astype(str))
    combined = parts[0] if parts else pd.Series([""] * len(items))
    for p in parts[1:]:
        combined = combined.str.cat(p, sep=" ")
    return dict(zip(items["item_id"], combined.str.strip()))


def probe(name, meta_mode="multihot", n_warm=4000, max_pairs=80000, seed=0):
    train, _test, items = load(name)
    item_ids = pd.Index(train["item_id"].unique())
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    item_ids_arr = item_ids.to_numpy()
    n_items = len(item_ids)

    text_by_id = _combined_text(items)
    has_text = np.array([bool(text_by_id.get(it, "")) for it in item_ids_arr])
    coverage = float(has_text.mean())

    iidx = train["item_id"].map(item_to_idx).to_numpy()
    uidx = pd.factorize(train["entity_id"])[0]
    d = np.bincount(iidx, minlength=n_items).astype(np.float64)

    order = np.argsort(-d)
    warm = order[has_text[order]][:n_warm]

    S = sp.csr_matrix((np.ones(len(iidx), np.float32), (uidx, iidx)),
                      shape=(int(uidx.max()) + 1, n_items))
    S.data[:] = 1.0
    S.sum_duplicates()
    S.data[:] = 1.0
    Ssub = S[:, warm]
    C = (Ssub.T @ Ssub).tocoo()
    m = C.row < C.col
    r, c, co = C.row[m], C.col[m], C.data[m].astype(np.float64)

    rng = np.random.default_rng(seed)
    if len(co) > max_pairs:
        sel = rng.choice(len(co), max_pairs, replace=False)
        r, c, co = r[sel], c[sel], co[sel]
    gi, gj = warm[r], warm[c]

    if meta_mode == "dense":
        from kindling.dense_content import embed_texts

        vecs = embed_texts([text_by_id.get(item_ids_arr[g], "") for g in warm])  # (n_warm, 384)
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        g2w = -np.ones(n_items, dtype=np.int64)
        g2w[warm] = np.arange(len(warm))
        n_features = vecs.shape[1]
        sim = (vecs[g2w[gi]] * vecs[g2w[gj]]).sum(axis=1)
    else:
        feat = ItemFeatureExtractor().fit_transform(items, item_to_idx, n_items)
        F = sp.csr_matrix((feat.data, feat.indices, feat.indptr), shape=(n_items, feat.n_features))
        n_features = feat.n_features
        sim = np.asarray(F[gi].multiply(F[gj]).sum(axis=1)).ravel()

    di, dj = d[gi], d[gj]
    cos_w = co / np.sqrt(di * dj)

    ntr = int(0.7 * len(sim))
    perm = rng.permutation(len(sim))
    tr, te = perm[:ntr], perm[ntr:]
    targets = {
        "cosine_weight": effect(sim[tr], cos_w[tr], sim[te], cos_w[te]),
        "raw_cocount": effect(sim[tr], co[tr], sim[te], co[te]),
    }
    aff = targets["cosine_weight"]
    alive = max(aff["r2_linear"], aff["r2_binned"]) >= GATE_R2 or aff["spearman"] >= GATE_SPEARMAN

    out = {
        "dataset": name, "meta_mode": meta_mode, "n_items": int(n_items),
        "n_warm_with_meta": int(len(warm)), "metadata_coverage": round(coverage, 3),
        "n_features": int(n_features),
        "n_pairs": int(len(sim)), "frac_pairs_sim_gt0": round(float((sim > 0).mean()), 3),
        "median_sim": round(float(np.median(sim)), 4),
        "targets": targets, "alive": bool(alive),
    }
    print(f"\n##### {name} [{meta_mode}] #####")
    print(f"  n_items={n_items} warm_w/meta={len(warm)} coverage={coverage:.2f} "
          f"feats={n_features} pairs={len(sim)} sim>0={out['frac_pairs_sim_gt0']:.2f}")
    for t, v in targets.items():
        print(f"  {t:14s} spearman={v['spearman']:+.3f}  R2_lin={v['r2_linear']:+.3f}  R2_binned={v['r2_binned']:+.3f}")
    print(f"  GATE (cosine_weight, R2>={GATE_R2} or rho>={GATE_SPEARMAN}): "
          f"{'ALIVE' if alive else 'DEAD'}")
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
