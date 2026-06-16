"""Cooc weight-smoothing reversal of the FPR probe.

FPR held the weights fixed (cosine) and swept the projection (stage 3) — and
the projection lost. This reverses it: hold projection = none (graph-direct
scoring, the part that works) and sweep stage 2 — the WEIGHT SMOOTHING — plus
the AGGREGATION it couples with.

Why 2-factor: on ml1m, cosine+sum scored like popularity (B1 0.2516 vs pop
0.2539). The leak isn't the edge transform — it's the sum-over-seen scorer:
a promiscuous (popular) candidate has nonzero edges to many items, so the SUM
re-imports popularity even after per-edge normalization. So we sweep the
transform AND the aggregation (sum / candidate-L2-normalized / max), because
breaking the leak needs the aggregation, not just the transform.

Everything is segment-sliced by held-out item train-popularity tier (1-4 /
5-19 / 20+), because on a global-chronological split popularity IS much of the
signal — a good smoother trades head for tail, and aggregate nDCG hides that.

EASE is the upper-reference smoother (the optimal global "do it all at once").

Run:
    .venv/bin/python bench/run_cooc_smoothing.py            # ml1m only (fast check)
    FPR_DATASETS=movielens-1m,amazon-beauty,steam .venv/bin/python bench/run_cooc_smoothing.py
Env: FPR_DATASETS (csv), FPR_EVAL (max eval users), FPR_AGG (csv subset),
     FPR_EASE (1/0), FPR_CDS_ALPHA, FPR_SPPMI_K, FPR_WILSON_Z.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from kindling.benchmarks.comparison import _load_academic_split, _load_dataset

REPORT = Path(__file__).parent / "reports" / "cooc_smoothing.json"
K = int(os.environ.get("FPR_K", "10"))
EPS = 1e-12


# --------------------------------------------------------------------------- #
# Weight smoothers. Each maps the raw co-count COO (a, di, dj, N) -> new weight.
# di, dj = item user-counts (popularity); N = n_users. Some zero out entries
# (PPMI family, LLR negative associations); callers eliminate_zeros after.
# --------------------------------------------------------------------------- #
def smooth_raw(a, di, dj, N, d):
    return a.copy()


def smooth_cosine(a, di, dj, N, d):
    return a / np.sqrt(di * dj)


def smooth_jaccard(a, di, dj, N, d):
    return a / (di + dj - a)


def smooth_ppmi(a, di, dj, N, d):
    pmi = np.log((a * N) / (di * dj))
    return np.maximum(pmi, 0.0)


def smooth_ppmi_cds(a, di, dj, N, d, alpha=0.75):
    # Context-distribution smoothing: rescale d^alpha to preserve total mass,
    # so rare items get a relatively larger effective marginal -> their PMI
    # boost shrinks (Levy-Goldberg's single biggest PMI fix).
    ds = d**alpha * (d.sum() / (d**alpha).sum())
    dsi, dsj = ds[smooth_ppmi_cds._ri], ds[smooth_ppmi_cds._ci]
    pmi = np.log((a * N) / (dsi * dsj))
    return np.maximum(pmi, 0.0)


def smooth_sppmi(a, di, dj, N, d, k=5.0):
    pmi = np.log((a * N) / (di * dj)) - np.log(k)
    return np.maximum(pmi, 0.0)


def smooth_llr(a, di, dj, N, d):
    # Dunning G^2 over the 2x2 (has i?) x (has j?) contingency table. Keep only
    # positive associations (observed co-count above independence expectation).
    def term(O, E):
        out = np.zeros_like(O)
        m = O > 0
        out[m] = O[m] * np.log(O[m] / np.maximum(E[m], EPS))
        return out

    a_ = a
    b_ = di - a
    c_ = dj - a
    d_ = N - di - dj + a
    Ea = di * dj / N
    Eb = di * (N - dj) / N
    Ec = (N - di) * dj / N
    Ed = (N - di) * (N - dj) / N
    g2 = 2.0 * (term(a_, Ea) + term(b_, Eb) + term(c_, Ec) + term(d_, Ed))
    g2[a_ < Ea] = 0.0  # drop negative (anti-) associations
    return g2


def smooth_wilson(a, di, dj, N, d, z=1.96):
    # Wilson lower bound on P(j|i) and P(i|j); take the conservative min. Auto-
    # shrinks low-count edges toward 0 — confidence-gating OBSERVED cooc (the
    # same principle the metadata edge-grafting will apply to SYNTHETIC cooc).
    def lb(phat, n):
        z2 = z * z
        return (phat + z2 / (2 * n) - z * np.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))) / (1 + z2 / n)

    lij = lb(a / di, di)
    lji = lb(a / dj, dj)
    return np.minimum(lij, lji)


SMOOTHERS = {
    "raw": smooth_raw,
    "cosine": smooth_cosine,
    "jaccard": smooth_jaccard,
    "ppmi": smooth_ppmi,
    "ppmi_cds": smooth_ppmi_cds,
    "sppmi": smooth_sppmi,
    "llr": smooth_llr,
    "wilson": smooth_wilson,
}


# --------------------------------------------------------------------------- #
def build_cooc(train, eval_cap):
    ents = train["entity_id"].to_numpy()
    its = train["item_id"].to_numpy()
    uent, ui = np.unique(ents, return_inverse=True)
    uitem, ii = np.unique(its, return_inverse=True)
    n_users, n_items = len(uent), len(uitem)
    S = sp.csr_matrix((np.ones(len(ui), np.float32), (ui, ii)), shape=(n_users, n_items))
    S.data[:] = 1.0
    S.sum_duplicates()
    S.data[:] = 1.0
    d = np.asarray(S.sum(axis=0)).ravel()  # item popularity (user count)
    C = (S.T @ S).tocoo()
    mask = C.row != C.col
    rows, cols, a = C.row[mask], C.col[mask], C.data[mask].astype(np.float64)
    item_ix = {it: i for i, it in enumerate(uitem)}
    entity_ix = {e: i for i, e in enumerate(uent)}
    owned = S  # users x items binary
    return dict(n_users=n_users, n_items=n_items, d=d, rows=rows, cols=cols, a=a,
                item_ix=item_ix, entity_ix=entity_ix, owned=owned, uitem=uitem)


def make_W(g, fn, **kw):
    smooth_ppmi_cds._ri, smooth_ppmi_cds._ci = g["rows"], g["cols"]
    di, dj = g["d"][g["rows"]], g["d"][g["cols"]]
    w = fn(g["a"], di, dj, g["n_users"], g["d"], **kw)
    keep = w > 0
    W = sp.csr_matrix((w[keep], (g["rows"][keep], g["cols"][keep])),
                      shape=(g["n_items"], g["n_items"]))
    return W


# --------------------------------------------------------------------------- #
def score_sum(W, owned_eval):
    return np.asarray((owned_eval @ W).todense())  # (n_eval, n_items)


def score_cand_l2(W, owned_eval):
    s = score_sum(W, owned_eval)
    rownorm = np.sqrt(np.asarray(W.multiply(W).sum(axis=1)).ravel())
    rownorm[rownorm == 0] = 1.0
    return s / rownorm[None, :]


def score_max(W, owned_eval):
    Wc = W.tocsc()
    out = np.zeros((owned_eval.shape[0], W.shape[0]), np.float64)
    for u in range(owned_eval.shape[0]):
        cols = owned_eval.getrow(u).indices
        if len(cols) == 0:
            continue
        out[u] = np.asarray(Wc[:, cols].max(axis=1).todense()).ravel()
    return out


AGGS = {"sum": score_sum, "cand_l2": score_cand_l2, "max": score_max}


# --------------------------------------------------------------------------- #
def gini(x):
    x = np.sort(x.astype(np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def evaluate(scores, eval_idx, owned_eval, relevant_idx, d, n_items, k=K):
    """scores: (n_eval, n_items). relevant_idx[u]: set of reachable item indices."""
    tiers = {"low_1_4": (1, 5), "mid_5_19": (5, 20), "warm_20+": (20, 10**9)}
    ndcgs, recalls = [], []
    tier_hit = {t: 0 for t in tiers}
    tier_tot = {t: 0 for t in tiers}
    rec_freq = np.zeros(n_items)
    idisc = 1.0 / np.log2(np.arange(2, k + 2))
    for u in range(scores.shape[0]):
        rel = relevant_idx[u]
        s = scores[u].copy()
        s[owned_eval.getrow(u).indices] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        rec_freq[top] += 1
        if not rel:
            continue
        hitmask = np.array([t in rel for t in top])
        if hitmask.any():
            dcg = idisc[hitmask].sum()
            ideal = idisc[: min(len(rel), k)].sum()
            ndcgs.append(dcg / ideal)
        else:
            ndcgs.append(0.0)
        recalls.append(hitmask.sum() / len(rel))
        topset = set(top.tolist())
        for r in rel:
            for t, (lo, hi) in tiers.items():
                if lo <= d[r] < hi:
                    tier_tot[t] += 1
                    if r in topset:
                        tier_hit[t] += 1
                    break
    return {
        "ndcg@10": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        "recall@10": round(float(np.mean(recalls)) if recalls else 0.0, 4),
        "coverage": round(float((rec_freq > 0).sum() / n_items), 4),
        "gini": round(gini(rec_freq), 4),
        **{f"rec_{t}": round(tier_hit[t] / tier_tot[t], 4) if tier_tot[t] else 0.0
           for t in tiers},
    }


# --------------------------------------------------------------------------- #
def run_dataset(name, eval_cap, aggs, want_ease):
    print(f"\n########## {name} (k={K}) ##########")
    if name == "amazon-book-academic":
        book = Path("~/.cache/kindling/amazon-book").expanduser()
        split = _load_academic_split(book / "train.txt", book / "test.txt",
                                     name="amazon-book-academic", action_type="purchase")
    else:
        split = _load_dataset(name, test_fraction=0.1)
    g = build_cooc(split.train, eval_cap)
    print(f"[graph] users={g['n_users']} items={g['n_items']} "
          f"cooc_edges={len(g['a'])} median_item_pop={int(np.median(g['d']))}")

    # eval users + reachable relevant item indices
    test = split.test
    train_by = split.train.groupby("entity_id", sort=False)["item_id"].apply(lambda s: set(s))
    test_by = test.groupby("entity_id", sort=False)["item_id"].apply(lambda s: set(s))
    cand = sorted(set(train_by.index) & set(test_by.index))
    if len(cand) > eval_cap:
        step = len(cand) // eval_cap
        cand = cand[::step][:eval_cap]
    rows_eval = [g["entity_ix"][e] for e in cand]
    owned_eval = g["owned"][rows_eval]
    item_ix = g["item_ix"]
    relevant_idx = []
    unreachable = 0
    for e in cand:
        rel = test_by[e] - train_by[e]
        ridx = set()
        for it in rel:
            j = item_ix.get(it)
            if j is None:
                unreachable += 1
            else:
                ridx.add(j)
        relevant_idx.append(ridx)
    print(f"[eval ] {len(cand)} users, unreachable_heldout={unreachable}")

    only = os.environ.get("FPR_SMOOTHERS")
    smoothers = {s: SMOOTHERS[s] for s in only.split(",")} if only else SMOOTHERS
    rows = []
    for sname, fn in smoothers.items():
        t0 = time.perf_counter()
        W = make_W(g, fn)
        W = W.maximum(W.T)  # symmetrize
        build_s = time.perf_counter() - t0
        for agg in aggs:
            scores = AGGS[agg](W, owned_eval)
            m = evaluate(scores, rows_eval, owned_eval, relevant_idx,
                         g["d"], g["n_items"])
            row = {"smoother": sname, "agg": agg, "nnz": int(W.nnz),
                   "build_s": round(build_s, 1), **m}
            rows.append(row)
            tag = f"{sname:9s}/{agg:8s}"
            print(f"  {tag} ndcg={m['ndcg@10']:.4f} rec={m['recall@10']:.4f} "
                  f"cov={m['coverage']:.3f} gini={m['gini']:.3f} | "
                  f"low={m['rec_low_1_4']:.3f} mid={m['rec_mid_5_19']:.3f} warm={m['rec_warm_20+']:.3f}")

    if want_ease and g["n_items"] <= 15000:
        try:
            t0 = time.perf_counter()
            Gram = np.asarray((g["owned"].T @ g["owned"]).todense(), np.float64)
            lam = 20.0 * float(np.mean(g["d"]))
            Gram[np.diag_indices_from(Gram)] += lam
            P = np.linalg.inv(Gram)
            B = -P / np.diag(P)[None, :]
            np.fill_diagonal(B, 0.0)
            scores = np.asarray(owned_eval.todense()) @ B
            m = evaluate(scores, rows_eval, owned_eval, relevant_idx, g["d"], g["n_items"])
            rows.append({"smoother": "EASE_ref", "agg": "closed_form", "nnz": -1,
                         "build_s": round(time.perf_counter() - t0, 1), **m})
            print(f"  {'EASE_ref':9s}/{'closed':8s} ndcg={m['ndcg@10']:.4f} rec={m['recall@10']:.4f} "
                  f"cov={m['coverage']:.3f} gini={m['gini']:.3f} | "
                  f"low={m['rec_low_1_4']:.3f} mid={m['rec_mid_5_19']:.3f} warm={m['rec_warm_20+']:.3f}")
        except (MemoryError, np.linalg.LinAlgError) as e:
            print(f"  EASE_ref skipped: {e}")

    return {"dataset": name, "n_items": g["n_items"], "n_users": g["n_users"],
            "cooc_edges": len(g["a"]), "rows": rows}


def main():
    datasets = os.environ.get("FPR_DATASETS", "movielens-1m").split(",")
    eval_cap = int(os.environ.get("FPR_EVAL", "2000"))
    aggs = os.environ.get("FPR_AGG", "sum,cand_l2,max").split(",")
    want_ease = os.environ.get("FPR_EASE", "1") == "1"
    out = {"protocol": "projection=none, global-chrono test_fraction=0.1, k=10",
           "aggregations": aggs, "datasets": []}
    for ds in datasets:
        try:
            out["datasets"].append(run_dataset(ds, eval_cap, aggs, want_ease))
        except Exception as e:  # noqa: BLE001
            print(f"[{ds}] FAILED: {type(e).__name__}: {e}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n[wrote] {REPORT}")


if __name__ == "__main__":
    main()
