"""Phase 2: low-rank EASE beyond the 20k gate, on the book-91k academic split.

Full EASE materializes a dense n×n Gram (66 GB at 91k) and a dense B (33 GB) —
the reason the engine gates EASE at ~20k. Low-rank EASE never forms either:

    G = XᵀX                       (sparse item-item Gram)
    G ≈ Σ_{k≤r} λ_k q_k q_kᵀ      (top-r eigenpairs, scipy eigsh on sparse G)
    P = (G + λI)⁻¹
      ≈ Σ_{k≤r} q_k q_kᵀ/(λ_k+λ) + (1/λ)(I − QQᵀ)   (small dirs → 1/λ)
    B = −P/diag(P), zero diag      (never materialized)

Per-user scoring (history h, owned excluded) collapses to one (r,)→(n,) matvec:
    hQ   = Σ_{i∈h} q_i            (r,)
    hP   = [hQ · (1/(λ_k+λ) − 1/λ)] @ Qᵀ    (n,)   (the h_j/λ term drops on owned)
    s    = −hP / diag(P)
    diag(P)_j = Σ_k q_{jk}²/(λ_k+λ) + (1/λ)(1 − Σ_k q_{jk}²)

Memory: Q is n×r (91k×512 ≈ 0.37 GB). The full-rank limit r=n is exact EASE.

Bar: beat the wilson cooc base (0.0482 NDCG@20 / 0.0592 Recall@20, same split &
eval set, run_book_academic.py BOOK_TRANSFORM=wilson) by enough to justify it.

Run: EASE_R=128,256 EASE_EVAL=2000 .venv/bin/python bench/run_ease_large.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.utils.extmath import randomized_svd

from kindling.benchmarks.comparison import _load_academic_split
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set

K = 20
REPORT = Path(__file__).parent / "reports" / "ease_large.json"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ranks = [int(x) for x in os.environ.get("EASE_R", "128,256,512").split(",")]
    n_eval = int(os.environ.get("EASE_EVAL", "2000"))
    lam_env = os.environ.get("EASE_LAMBDA", "")

    book = Path("~/.cache/kindling/amazon-book").expanduser()
    split = _load_academic_split(
        book / "train.txt", book / "test.txt", name="amazon-book-academic",
        action_type="rate",
    )
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=n_eval, seed=0)
    log(f"split: train {len(train):,} items {train.item_id.nunique():,} "
        f"eval_users {len(eval_set)}")

    item_ids = pd.Index(train["item_id"].unique())
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    n_items = len(item_ids)
    uidx, uniq_u = pd.factorize(train["entity_id"])
    iidx = train["item_id"].map(item_to_idx).to_numpy()
    n_users = len(uniq_u)
    entity_to_user = {e: i for i, e in enumerate(uniq_u)}

    X = sp.csr_matrix((np.ones(len(iidx), np.float32), (uidx, iidx)),
                      shape=(n_users, n_items))
    X.data[:] = 1.0
    X.sum_duplicates()
    X.data[:] = 1.0
    nnz = X.nnz
    lam_default = 20.0 * nnz / n_items
    lambdas = [float(x) for x in lam_env.split(",")] if lam_env else [lam_default]
    log(f"X nnz={nnz:,}  lambda_default={lam_default:.1f}  lambdas={lambdas}  ranks={ranks}")

    t = time.perf_counter()
    G = (X.T @ X).tocsr().astype(np.float64)
    log(f"Gram: shape {G.shape} nnz {G.nnz:,} ({time.perf_counter()-t:.0f}s)")

    # owned indices per eval entity (for exclusion + history vector).
    owned_by = {}
    for e in eval_set:
        u = entity_to_user.get(e)
        if u is None:
            continue
        owned_by[e] = X.indices[X.indptr[u]:X.indptr[u + 1]]

    out = []
    solver = os.environ.get("EASE_SOLVER", "eigsh")  # "eigsh" (exact) | "randomized" (fast)
    for r in ranks:
        t = time.perf_counter()
        if solver == "randomized":
            # G is symmetric PSD → randomized SVD recovers its top eigenpairs
            # far faster than Lanczos at high r (a few power passes).
            u, s, _ = randomized_svd(G, n_components=min(r, n_items - 2),
                                     n_iter=5, random_state=0)
            vals, vecs = s, u  # singular values == eigenvalues for SPD G
        else:
            vals, vecs = eigsh(G, k=min(r, n_items - 2), which="LM")
        order = np.argsort(-vals)
        vals, vecs = vals[order], vecs[:, order]  # (n, r)
        q2 = vecs ** 2
        sum_q2 = q2.sum(1)  # (n,)
        log(f"{solver} r={r}: top eig {vals[0]:.1f} .. {vals[-1]:.3f} "
            f"({time.perf_counter()-t:.0f}s)")

        for lam in lambdas:
            dinv = 1.0 / (vals + lam)              # (r,)
            diagP = q2 @ dinv + (1.0 / lam) * (1.0 - sum_q2)  # (n,)
            coef_scale = dinv - 1.0 / lam          # (r,)
            te = time.perf_counter()
            per = []
            for e, rel in eval_set.items():
                owned = owned_by.get(e)
                if owned is None or owned.size == 0:
                    per.append(([], rel))
                    continue
                hq = vecs[owned].sum(0)            # (r,)
                hp = (hq * coef_scale) @ vecs.T    # (n,)
                s = -hp / diagP
                s[owned] = -np.inf
                top = np.argpartition(-s, K)[:K]
                top = top[np.argsort(-s[top])]
                per.append(([item_ids[i] for i in top], rel))
            rep = aggregate(per, catalog_size=n_items, k=K)
            row = {"rank": r, "lambda": round(lam, 1),
                   "recall@20": round(rep.recall_at_k, 4),
                   "ndcg@20": round(rep.ndcg_at_k, 4),
                   "mrr": round(rep.mrr, 4), "hr": round(rep.hit_rate, 4)}
            out.append(row)
            log(f"  r={r} lam={lam:.1f}: Recall@20={row['recall@20']} "
                f"NDCG@20={row['ndcg@20']}  (eval {time.perf_counter()-te:.0f}s)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    log("wilson cooc reference: Recall@20 0.0592  NDCG@20 0.0482  (run_book_academic)")
    log(f"[wrote] {REPORT}")


if __name__ == "__main__":
    main()
