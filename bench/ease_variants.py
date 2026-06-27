# ruff: noqa: N802, N803, N806  (linear-algebra matrix names: X, G, B, P, T)
"""EASE-variant assessment (Stage 3): EASE vs EDLAE vs RLAE on the ml-1m split.

DIY closed-form item-item linear autoencoders, scored with the SAME valid-masked
eval used for the kindling-vs-RecBole comparison (mask each user's train+valid
items, predict their held-out test basket). The RecBole EASE/ADMMSLIM/SLIMElastic
numbers (bench/recbole_runner.py) are reported alongside as an independent anchor.

NOTE: EDLAE and RLAE here are *our* closed forms (clearly-defined relaxations of
EASE), not guaranteed byte-identical to Steck (2020); the point is the relative
lift over EASE on our data. EASE itself is the canonical Steck (2019) form.

  EASE   B = -P/diag(P), P=(XᵀX+λI)⁻¹, diag(B)=0
  EDLAE  popularity-scaled diagonal ridge: G[i,i] += λ + δ·G[i,i]   (denoising)
  RLAE   relaxed zero-diagonal: B = I - P·diag(1/(diag(P)+ρ))       (ρ=0 → EASE)

Run: PYTHONPATH=src .venv/bin/python bench/ease_variants.py
"""

from __future__ import annotations

import json
from math import log2
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("recbole_data")


def _load():
    with open(D / "split_truth.json") as fh:
        truth = {u: set(v) for u, v in json.load(fh).items()}
    train = pd.read_csv(D / "split_train.csv")
    full = pd.read_csv(D / "ml-1m/ml-1m.inter", sep="\t")
    full.columns = ["user", "item", "rating", "ts"]
    full_by = {str(u): set(map(str, g)) for u, g in full.groupby("user")["item"]}
    train_by = {str(u): set(map(str, g)) for u, g in train.groupby("entity_id")["item_id"]}
    valid_by = {u: (full_by.get(u, set()) - train_by.get(u, set()) - truth.get(u, set())) for u in truth}
    items = sorted(train["item_id"].astype(str).unique())
    i2c = {it: c for c, it in enumerate(items)}
    users = sorted(train["entity_id"].astype(str).unique())
    u2r = {u: r for r, u in enumerate(users)}
    rows = train["entity_id"].astype(str).map(u2r).to_numpy()
    cols = train["item_id"].astype(str).map(i2c).to_numpy()
    X = np.zeros((len(users), len(items)), dtype=np.float64)
    X[rows, cols] = 1.0
    return X, i2c, u2r, truth, train_by, valid_by, items


def _gram(X):
    return X.T @ X


def ease_B(G, lam):
    P = np.linalg.inv(G + lam * np.eye(G.shape[0]))
    d = np.diag(P).copy()
    B = -P / d[None, :]
    np.fill_diagonal(B, 0.0)
    return B


def edlae_B(G, lam, delta):
    G2 = G.copy()
    np.fill_diagonal(G2, np.diag(G2) + lam + delta * np.diag(G))
    P = np.linalg.inv(G2)
    d = np.diag(P).copy()
    B = -P / d[None, :]
    np.fill_diagonal(B, 0.0)
    return B


def rlae_B(G, lam, rho):
    P = np.linalg.inv(G + lam * np.eye(G.shape[0]))
    gamma = 1.0 / (np.diag(P) + rho)
    B = -P * gamma[None, :]
    np.fill_diagonal(B, 1.0 - np.diag(P) * gamma)
    return B


def score(B, X, i2c, u2r, truth, train_by, valid_by, k=10, max_users=3000):
    ev = [u for u in truth if u in u2r][:max_users]
    nd = []
    for u in ev:
        s = X[u2r[u]] @ B
        for it in train_by.get(u, set()) | valid_by.get(u, set()):
            c = i2c.get(it)
            if c is not None:
                s[c] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        inv = {v: kk for kk, v in i2c.items()}
        rec = [inv[c] for c in top]
        T = truth[u]
        dcg = sum(1 / log2(i + 2) for i, p in enumerate(rec) if p in T)
        idcg = sum(1 / log2(i + 2) for i in range(min(len(T), k)))
        nd.append(dcg / idcg if idcg else 0.0)
    return round(float(np.mean(nd)), 4)


def main():
    X, i2c, u2r, truth, train_by, valid_by, items = _load()
    G = _gram(X)
    args = (X, i2c, u2r, truth, train_by, valid_by)
    print(f"ml-1m split: users={X.shape[0]} items={X.shape[1]}")
    best = {}
    print("\n--- EASE (λ sweep) ---")
    for lam in (250, 500, 1000, 2000):
        nd = score(ease_B(G, lam), *args)
        best["EASE"] = max(best.get("EASE", (0, 0)), (nd, lam))
        print(f"  λ={lam:5d}  NDCG@10={nd}")
    el = best["EASE"][1]
    print(f"\n--- EDLAE (denoising; λ={el}, δ sweep) ---")
    for delta in (0.0, 0.05, 0.1, 0.25, 0.5):
        nd = score(edlae_B(G, el, delta), *args)
        best["EDLAE"] = max(best.get("EDLAE", (0, 0)), (nd, delta))
        print(f"  δ={delta:<4}  NDCG@10={nd}")
    print(f"\n--- RLAE (relaxed diag; λ={el}, ρ sweep) ---")
    for rho in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
        nd = score(rlae_B(G, el, rho), *args)
        best["RLAE"] = max(best.get("RLAE", (0, 0)), (nd, rho))
        print(f"  ρ={rho:<6}  NDCG@10={nd}")
    print("\n=== best per variant (valid-masked eval) ===")
    for k, (nd, hp) in best.items():
        print(f"  {k:6s} {nd}  (hp={hp})")
    with open(D / "ease_variants_diy.json", "w") as fh:
        json.dump({k: {"ndcg": v[0], "hp": v[1]} for k, v in best.items()}, fh, indent=2)


if __name__ == "__main__":
    main()
