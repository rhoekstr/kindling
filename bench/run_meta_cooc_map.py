"""Metadata->cooc MAPPING quality: how much of the cooc structure can metadata
reconstruct? (Robert's reframe: cooc is the strong signal; metadata's value =
its ability to predict cooc structure, especially transferred to held-out items.)

  cooc_emb  = truncated-SVD of the PPMI cooc matrix (the affinity structure)
  meta_emb  = SVD of multi-hot item_features, OR dense MiniLM vectors
  map       = ridge regression  meta_emb -> cooc_emb  fit on a WARM item split
  metric    = transfer R^2 + cooc-neighbor recovery on HELD-OUT warm items

Higher = metadata captures more cooc structure = more valuable (and the map is
exactly what would place a cold item in cooc-space from its metadata).

Decisive checks: does this rank steam > book (the AUC inverted there), and does
it discriminate multi-hot vs dense on steam (the AUC couldn't)?

Run: FPR_DATASETS=steam,amazon-book-chrono,movielens-1m FPR_META=multihot .venv/bin/python bench/run_meta_cooc_map.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from kindling.item_features import ItemFeatureExtractor
from run_graft_probe import _combined_text, load as _graft_load


def load(name):
    if name in ("steam", "amazon-book-chrono"):
        return _graft_load(name)
    from kindling.benchmarks.comparison import _load_dataset

    s = _load_dataset(name, test_fraction=0.1)
    return s.train, s.test, s.items

REPORT = Path(__file__).parent / "reports" / "meta_cooc_map.json"
DIM = 64
N_WARM = int(os.environ.get("N_WARM", "20000"))
WARM_MIN = 5


def ppmi(C, d, n_users):
    """Positive PMI on the cooc COO, returned sparse."""
    co = C.data.astype(np.float64)
    di, dj = d[C.row], d[C.col]
    pmi = np.log(np.maximum(co * n_users / np.maximum(di * dj, 1.0), 1e-12))
    keep = pmi > 0
    return sp.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape)


def svd_emb(M, dim):
    dim = min(dim, min(M.shape) - 1)
    u, s, _ = svds(M.asfptype(), k=dim)
    return u * s  # rows = item embeddings


def ridge_transfer(X, Y, train, test, kind="ridge", lam=10.0):
    # Standardize both spaces on the train split (center + unit-variance) so the
    # map has an implicit intercept and R^2 weights all cooc dims equally.
    mx, sx = X[train].mean(0), np.maximum(X[train].std(0), 1e-9)
    my, sy = Y[train].mean(0), np.maximum(Y[train].std(0), 1e-9)
    Xs = (X - mx) / sx
    Ys = (Y - my) / sy
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor

        m = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=400, alpha=1e-3,
                         early_stopping=True, random_state=0).fit(Xs[train], Ys[train])
        Yhat = m.predict(Xs[test])
    else:
        Xtr, Ytr = Xs[train], Ys[train]
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(X.shape[1]), Xtr.T @ Ytr)
        Yhat = Xs[test] @ W
    Yte = Ys[test]
    ss_res = float(((Yte - Yhat) ** 2).sum())
    ss_tot = float(((Yte - Yte.mean(axis=0)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # cooc-neighbor recovery: do predicted positions retrieve the true neighbors?
    Yn = Yte / np.maximum(np.linalg.norm(Yte, axis=1, keepdims=True), 1e-9)
    Hn = Yhat / np.maximum(np.linalg.norm(Yhat, axis=1, keepdims=True), 1e-9)
    rec = []
    for i in range(min(400, len(test))):
        true_nn = np.argpartition(-(Yn @ Yn[i]), 11)[:11]
        pred_nn = np.argpartition(-(Yn @ Hn[i]), 10)[:10]
        rec.append(len(set(true_nn.tolist()) & set(pred_nn.tolist())) / 10)
    return r2, float(np.mean(rec))


def _load_enriched(name, content):
    """LLM-enriched metadata from cache as an items df (item_id + content text).
    content: 'kw' (keywords) or 'niche' (niche-positioning phrases)."""
    short = {"movielens-1m": "ml1m", "steam": "steam"}.get(name, name)
    suffix = "keywords" if content == "kw" else "niche"
    path = Path(__file__).parent / "cache" / f"{short}_{suffix}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame({"item_id": [r["item_id"] for r in rows],
                         "content": [" ".join(r.get("keywords", [])) for r in rows]})


def _load_aisle(name, shuffle=False):
    """LLM store-aisle labels (run_aisle_classify.py) as an items df with
    aisle + section categorical columns; optionally shuffle labels across items
    (a control: does the cooc-mapping come from the LABELS or just bucketing?)."""
    path = Path(__file__).parent / "cache" / f"{name}_aisle.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame([r for r in rows if r.get("aisle")])
    if shuffle:
        perm = np.random.default_rng(0).permutation(len(df))
        df[["aisle", "section"]] = df[["aisle", "section"]].to_numpy()[perm]
    return df[["item_id", "aisle", "section"]]


def meta_embedding(name, meta_mode, items, item_to_idx, n_items, warm):
    # meta_mode: native "multihot"/"dense", "<content>_<rep>", or "aisle"[_shuffle].
    if meta_mode.startswith("aisle"):
        adf = _load_aisle(name, shuffle=meta_mode.endswith("shuffle"))
        feat = ItemFeatureExtractor().fit_transform(adf, item_to_idx, n_items)
        F = sp.csr_matrix((feat.data, feat.indices, feat.indptr),
                          shape=(n_items, feat.n_features))
        return svd_emb(F[warm], DIM)
    rep = meta_mode
    if "_" in meta_mode:
        content, rep = meta_mode.split("_", 1)
        items = _load_enriched(name, content)
    if rep == "dense":
        from kindling.dense_content import embed_texts

        text = _combined_text(items)
        ids = pd.Index(item_to_idx).to_numpy()
        V = embed_texts([text.get(ids[w], "") for w in warm])
        return V.astype(np.float64)
    feat = ItemFeatureExtractor().fit_transform(items, item_to_idx, n_items)
    F = sp.csr_matrix((feat.data, feat.indices, feat.indptr), shape=(n_items, feat.n_features))
    return svd_emb(F[warm], DIM)


def probe(name, meta_mode, map_kind="ridge", seed=0):
    train, _test, items = load(name)
    item_ids = pd.Index(train["item_id"].unique())
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    n_items = len(item_ids)
    iidx = train["item_id"].map(item_to_idx).to_numpy()
    uidx = pd.factorize(train["entity_id"])[0]
    n_users = int(uidx.max()) + 1
    d = np.bincount(iidx, minlength=n_items).astype(np.float64)

    warm = np.where(d >= WARM_MIN)[0]
    warm = warm[np.argsort(-d[warm])][:N_WARM]
    gpos = {int(g): i for i, g in enumerate(warm)}

    S = sp.csr_matrix((np.ones(len(iidx), np.float32), (uidx, iidx)), shape=(n_users, n_items))
    S.data[:] = 1.0
    S.sum_duplicates()
    S.data[:] = 1.0
    Sw = S[:, warm]
    C = (Sw.T @ Sw).tocoo()
    keep = C.row != C.col
    C = sp.coo_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=(len(warm), len(warm)))
    cooc_emb = svd_emb(ppmi(C, d[warm], n_users), DIM)

    meta_emb = meta_embedding(name, meta_mode, items, item_to_idx, n_items, warm)
    # standardize columns
    meta_emb = (meta_emb - meta_emb.mean(0)) / np.maximum(meta_emb.std(0), 1e-9)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(warm))
    cut = int(0.7 * len(warm))
    tr, te = perm[:cut], perm[cut:]
    r2, rec = ridge_transfer(meta_emb, cooc_emb, tr, te, kind=map_kind)

    out = {"dataset": name, "meta_mode": meta_mode, "map": map_kind, "n_warm": int(len(warm)),
           "meta_dim": int(meta_emb.shape[1]), "transfer_r2": round(r2, 4),
           "neighbor_recovery@10": round(rec, 4)}
    print(f"  {name:20s} [{meta_mode:8s}/{map_kind:5s}] transfer_R2={r2:+.4f}  cooc-nbr-recovery@10={rec:.3f}  (warm={len(warm)})")
    return out


def main():
    datasets = os.environ.get("FPR_DATASETS", "steam,amazon-book-chrono,movielens-1m").split(",")
    modes = os.environ.get("FPR_META", "multihot,dense").split(",")
    maps = os.environ.get("FPR_MAP", "ridge").split(",")
    out = []
    print(f"metadata->cooc mapping quality (cooc_emb=PPMI-SVD{DIM}, warm>= {WARM_MIN}, top {N_WARM}):")
    for ds in datasets:
        for mode in modes:
            for mk in maps:
                try:
                    out.append(probe(ds, mode, map_kind=mk))
                except Exception as e:  # noqa: BLE001
                    print(f"  {ds} [{mode}/{mk}] FAILED: {type(e).__name__}: {e}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[wrote] {REPORT}")


if __name__ == "__main__":
    main()
