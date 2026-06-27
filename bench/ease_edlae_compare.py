# ruff: noqa: N803, N806  (linear-algebra matrix names: X, G, B, P)
"""EASE vs EDLAE (EASE+) across datasets: fit time + accuracy, exclude-seen eval.

Both are single dense item-item inversions (EDLAE adds an O(n) diagonal term), so
this quantifies the fit-time delta and finds any dataset where EDLAE < EASE.
EASE-applicable catalogs only (<= 20k train items).

Run: PYTHONPATH=src .venv/bin/python bench/ease_edlae_compare.py <dataset...>
"""

from __future__ import annotations

import sys
import time
from math import log2

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "bench")
from run_warming_curve import load_split

DATASETS = ["movielens-1m", "amazon-beauty", "steam", "tafeng", "dunnhumby"]
MAX_ITEMS = 20_000
LAMBDAS = [100.0, 250.0, 500.0, 1000.0, 2000.0]
DELTAS = [0.05, 0.1, 0.25, 0.5, 1.0]  # EDLAE denoising strength


def _eval(B, owned_rows, eval_rows, test_sets, k=10):
    nd = []
    for r, owned in zip(eval_rows, owned_rows):
        if not owned:
            continue
        scores = B[owned].sum(axis=0)
        scores[owned] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        rel = test_sets[r]
        dcg = sum(1 / log2(i + 2) for i, c in enumerate(top) if c in rel)
        idcg = sum(1 / log2(i + 2) for i in range(min(len(rel), k)))
        nd.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(nd)) if nd else 0.0


def run(ds: str, max_users: int = 1500):
    sp_ = load_split(ds, 0.1)
    tr, te = sp_.train, sp_.test
    items = {it: c for c, it in enumerate(tr["item_id"].unique())}
    n_items = len(items)
    if n_items > MAX_ITEMS:
        print(f"{ds:14s} SKIP — {n_items:,} items > {MAX_ITEMS:,} (cooc base, not EASE)")
        return
    users = {u: r for r, u in enumerate(tr["entity_id"].unique())}
    rows = tr["entity_id"].map(users).to_numpy()
    cols = tr["item_id"].map(items).to_numpy()
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(users), n_items))
    G = np.asarray((X.T @ X).todense(), dtype=np.float64)

    owned_by = {u: [] for u in users}
    for u, c in zip(tr["entity_id"].to_numpy(), cols):
        owned_by[u].append(int(c))
    test_by = {}
    for u, g in te.groupby("entity_id"):
        if u in users:
            test_by[users[u]] = {items[i] for i in g["item_id"] if i in items}
    erows = [r for r in test_by if test_by[r]]
    rng = np.random.default_rng(0)
    if len(erows) > max_users:
        erows = [erows[i] for i in rng.choice(len(erows), max_users, replace=False)]
    inv_u = {r: u for u, r in users.items()}
    owned_rows = [owned_by[inv_u[r]] for r in erows]

    eye = np.eye(n_items)
    # EASE — tune lambda
    best_e = (-1, None, 0.0)
    t0 = time.perf_counter()
    P = np.linalg.inv(G + LAMBDAS[0] * eye)
    ease_fit = time.perf_counter() - t0  # one inversion
    for lam in LAMBDAS:
        P = np.linalg.inv(G + lam * eye)
        d = np.diag(P).copy()
        B = -P / d[None, :]
        np.fill_diagonal(B, 0.0)
        nd = _eval(B, owned_rows, erows, test_by)
        if nd > best_e[0]:
            best_e = (nd, lam, ease_fit)
    # EDLAE — tune (lambda, delta)
    best_d = (-1, None, None, 0.0)
    t0 = time.perf_counter()
    Gd = G.copy()
    np.fill_diagonal(Gd, np.diag(Gd) + LAMBDAS[0] + DELTAS[0] * np.diag(G))
    _ = np.linalg.inv(Gd)
    edlae_fit = time.perf_counter() - t0
    for lam in LAMBDAS:
        for delta in DELTAS:
            Gd = G.copy()
            np.fill_diagonal(Gd, np.diag(Gd) + lam + delta * np.diag(G))
            P = np.linalg.inv(Gd)
            d = np.diag(P).copy()
            B = -P / d[None, :]
            np.fill_diagonal(B, 0.0)
            nd = _eval(B, owned_rows, erows, test_by)
            if nd > best_d[0]:
                best_d = (nd, lam, delta, edlae_fit)
    delta_pct = 100 * (best_d[0] - best_e[0]) / max(best_e[0], 1e-9)
    flag = "  <-- EDLAE WORSE" if best_d[0] < best_e[0] else ""
    print(f"{ds:14s} items={n_items:6,}  EASE ndcg={best_e[0]:.4f} (λ{best_e[1]:g}, fit={ease_fit:.2f}s)  "
          f"EDLAE ndcg={best_d[0]:.4f} (λ{best_d[1]:g} δ{best_d[2]:g}, fit={edlae_fit:.2f}s)  "
          f"Δacc={delta_pct:+.2f}%  Δfit={edlae_fit-ease_fit:+.2f}s{flag}")


def main(argv):
    for ds in (argv[1:] or DATASETS):
        try:
            run(ds)
        except Exception as e:
            print(f"{ds:14s} ERR {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    main(sys.argv)
