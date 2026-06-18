"""ml-25m cold-lift validation: does the metadata layer actually recover cold
items on the good dataset (content-coherent + real cold tail + warm-dominated)?

Mechanism = EMBEDDING IMPUTATION (flood-free, the mapping pointed to it):
  cooc_emb   = PPMI-SVD of warm-item cooc (the strong signal)
  W          = ridge map  tag-genome -> cooc_emb  (fit on warm)
  cold item  -> predicted cooc-position  W . genome(cold)   (ONE vector, not k edges)
  score      = user_profile (mean owned cooc_emb) . item_position   (cosine in cooc-space)

Arms: baseline (cold items unscorable, as in production) vs impute (cold items
placed by predicted cooc-position). Segmented recall by held-out warmth tier +
net NDCG. Chronological split so recent/cold movies are genuinely held out.

Run: SUB_USERS=40000 .venv/bin/python bench/run_ml25m_lift.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from run_meta_cooc_map import DIM, ppmi, svd_emb

D = Path("~/.cache/kindling/ml-25m").expanduser()
K = 20
WARM_MIN = 5
COLD_MAX = 4
TIERS = {"cold_1-4": (1, 5), "5-9": (5, 10), "10-29": (10, 30), "30+": (30, 10**9)}


def main():
    sub = int(os.environ.get("SUB_USERS", "40000"))
    rng = np.random.default_rng(0)

    print("[load] ratings + genome ...", flush=True)
    r = pd.read_csv(D / "ratings.csv", usecols=["userId", "movieId", "timestamp"],
                    dtype={"userId": np.int32, "movieId": np.int32, "timestamp": np.int64})
    g = pd.read_csv(D / "genome-scores.csv")
    genome_movies = set(g["movieId"].unique().tolist())
    n_tags = int(g["tagId"].max())

    users = r["userId"].unique()
    if len(users) > sub:
        keep = set(rng.choice(users, sub, replace=False).tolist())
        r = r[r["userId"].isin(keep)]
    r = r[r["movieId"].isin(genome_movies)]

    # chronological split (global cutoff)
    cut_ts = r["timestamp"].quantile(0.90)
    tr_df = r[r["timestamp"] <= cut_ts]
    te_df = r[r["timestamp"] > cut_ts]

    movies = pd.Index(tr_df["movieId"].unique())
    m2i = {int(m): i for i, m in enumerate(movies)}
    n_items = len(movies)
    iidx = tr_df["movieId"].map(m2i).to_numpy()
    uidx = pd.factorize(tr_df["userId"])[0]
    n_users = int(uidx.max()) + 1
    d = np.bincount(iidx, minlength=n_items).astype(np.float64)

    warm = np.where(d >= WARM_MIN)[0]
    warm = warm[np.argsort(-d[warm])]
    cold = np.where((d >= 1) & (d <= COLD_MAX))[0]
    print(f"[graph] users={n_users} items={n_items} warm={len(warm)} cold(1-{COLD_MAX})={len(cold)}", flush=True)

    S = sp.csr_matrix((np.ones(len(iidx), np.float32), (uidx, iidx)), shape=(n_users, n_items))
    S.data[:] = 1.0
    S.sum_duplicates()
    S.data[:] = 1.0
    Sw = S[:, warm]
    C = (Sw.T @ Sw).tocoo()
    keep = C.row != C.col
    C = sp.coo_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=(len(warm), len(warm)))
    cooc_emb = svd_emb(ppmi(C, d[warm], n_users), DIM)  # (n_warm, DIM)

    # genome matrix for warm + cold
    def genome_mat(items_global):
        mv = movies.to_numpy()[items_global]
        sub_g = g[g["movieId"].isin(set(int(m) for m in mv))]
        pos = {int(m): i for i, m in enumerate(mv)}
        rows = sub_g["movieId"].map(pos).to_numpy()
        return sp.csr_matrix((sub_g["relevance"].to_numpy(), (rows, sub_g["tagId"].to_numpy() - 1)),
                             shape=(len(items_global), n_tags)).toarray()

    Gw = genome_mat(warm)
    Gc = genome_mat(cold)

    # standardize on warm, fit ridge genome->cooc_emb, predict cold positions
    mx, sx = Gw.mean(0), np.maximum(Gw.std(0), 1e-9)
    my, sy = cooc_emb.mean(0), np.maximum(cooc_emb.std(0), 1e-9)
    Xw = (Gw - mx) / sx
    Yw = (cooc_emb - my) / sy
    W = np.linalg.solve(Xw.T @ Xw + 10.0 * np.eye(n_tags), Xw.T @ Yw)
    pos_warm = Yw                      # standardized cooc_emb
    pos_cold = ((Gc - mx) / sx) @ W    # predicted standardized positions

    # catalog = warm ++ cold
    cat_global = np.concatenate([warm, cold])
    pos = np.vstack([pos_warm, pos_cold])
    is_cold = np.concatenate([np.zeros(len(warm), bool), np.ones(len(cold), bool)])
    g2cat = -np.ones(n_items, np.int64)
    g2cat[cat_global] = np.arange(len(cat_global))
    warm_cat = g2cat[warm]  # for profile building

    # eval
    tb = tr_df.groupby("userId", sort=False)["movieId"].apply(lambda s: set(s))
    te = te_df.groupby("userId", sort=False)["movieId"].apply(lambda s: set(s))
    evalu = sorted(set(tb.index) & set(te.index))
    rng.shuffle(evalu)
    evalu = evalu[:3000]
    idisc = 1.0 / np.log2(np.arange(2, K + 2))

    def run(score_cold):
        tier_hit = {t: 0 for t in TIERS}
        tier_tot = {t: 0 for t in TIERS}
        ndcgs, covset = [], set()
        for u in evalu:
            owned_g = np.array([m2i[m] for m in tb[u] if m in m2i])
            owned_warm_cat = g2cat[owned_g[d[owned_g] >= WARM_MIN]] if len(owned_g) else np.array([], int)
            owned_warm_cat = owned_warm_cat[owned_warm_cat >= 0]
            if len(owned_warm_cat) == 0:
                continue
            profile = pos[owned_warm_cat].mean(0)
            s = pos @ profile
            owned_cat = g2cat[owned_g]
            s[owned_cat[owned_cat >= 0]] = -np.inf
            if not score_cold:
                s[is_cold] = -np.inf
            top = np.argpartition(-s, K)[:K]
            top = top[np.argsort(-s[top])]
            covset.update(top.tolist())
            rel = [m2i[m] for m in (te[u] - tb[u]) if m in m2i]
            if not rel:
                continue
            topglob = set(cat_global[t] for t in top)
            g_top = np.array([1.0 if int(cat_global[t]) in set(rel) else 0.0 for t in top])
            ndcgs.append(float((g_top * idisc).sum()) / float(idisc[: min(len(rel), K)].sum()) if g_top.sum() else 0.0)
            for ri in rel:
                for t, (lo, hi) in TIERS.items():
                    if lo <= d[ri] < hi:
                        tier_tot[t] += 1
                        if ri in topglob:
                            tier_hit[t] += 1
                        break
        return {"ndcg": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
                "cov": round(len(covset) / len(cat_global), 4),
                **{t: (round(tier_hit[t] / tier_tot[t], 4) if tier_tot[t] else None) for t in TIERS},
                **{f"n_{t}": tier_tot[t] for t in TIERS}}

    base = run(score_cold=False)
    imp = run(score_cold=True)
    print("\nml-25m cold-lift (embedding imputation, cooc-space scoring):")
    print(f"  held-out n per tier: " + " ".join(f"{t}={base['n_' + t]}" for t in TIERS))
    for name, rr in [("baseline (cold off)", base), ("impute (cold on)", imp)]:
        print(f"  {name:20s} ndcg={rr['ndcg']:.4f} cov={rr['cov']:.3f} | "
              + " ".join(f"{t}={rr[t]}" for t in TIERS))


if __name__ == "__main__":
    main()
