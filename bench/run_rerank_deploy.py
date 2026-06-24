"""§7.2 re-ranker DEPLOYABILITY test: does the dep-free ridge+crosses win survive
training on an internal (train-only) holdout, or invert like §4.4?

The ceiling probe (run_rerank.py) fit the ridge on a split of EVAL (test-label)
users — not deployable. A deployable re-ranker can only train on TRAIN: for each
train user hold out their LAST item as a pseudo-positive, build features from the
rest, fit ridge there, then rank the REAL test pool. §4.4 found per-fit
calibration on an internal chronological holdout INVERTS the test ranking
(shifted drift structure), so this is the decisive test.

  deploy_ridge_linear / deploy_ridge_cross2  — trained on the internal holdout
  current_blend                              — the shipped reference
  ceiling_ridge_cross2                       — trained on eval-half (oracle, for contrast)

Survives (deploy ≥ current) -> real dep-free win, wire it.
Inverts (deploy ≤ current)  -> §4.4 generalizes, learned re-rank undeployable.

Run: DATASET=movielens-1m .venv/bin/python bench/run_rerank_deploy.py  (| amazon-beauty)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set
from kindling.engine_v2 import EngineV2

K = 10
N_INTERNAL = 3000  # train users sampled for the internal holdout
REPORT_DIR = Path(__file__).parent / "reports"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def zc(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def poly2(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    cols = [X]
    for i in range(d):
        cols.append(X[:, i:] * X[:, i][:, None])
    return np.hstack(cols)


def fit_ridge(Xtr, ytr, lam=10.0):
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-9)
    Z = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ ytr)
    return (W, mu, sd)


def apply_ridge(model, X):
    W, mu, sd = model
    return np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ W


def main() -> None:
    dataset = os.environ.get("DATASET", "movielens-1m")
    split = _load_dataset(dataset, test_fraction=0.1)
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=500, seed=0)
    eng = EngineV2(persona_min_users=10**9, retrieval_budget=500, random_state=0)
    eng.fit(train)
    st = eng._state
    n_items = st.n_items
    catalog = max(n_items, 1)
    ease = st.ease_b.astype(np.float64)
    trend_z = st.trend_z if st.trend_z is not None else np.zeros(n_items)
    pop = np.zeros(n_items)
    col = train["item_id"].map(st.item_to_idx).dropna().astype(int).to_numpy()
    np.add.at(pop, col, 1)
    pop_z = zc(np.log1p(pop))

    def features_for(owned: np.ndarray, pool: np.ndarray) -> np.ndarray:
        return np.stack([
            zc(ease[owned].sum(0))[pool], zc(ease[owned[-3:]].sum(0))[pool],
            zc(ease[owned[-10:]].sum(0))[pool], zc(ease[int(owned[-1])])[pool],
            trend_z[pool], pop_z[pool]], axis=1)

    def pool_for(owned: np.ndarray) -> np.ndarray:
        base = ease[owned].sum(0)
        s = eng._blend_channels(st, owned, base.copy(), user_row=-1)
        s[owned] = -np.inf
        b = min(500, s.size)
        p = np.argpartition(-s, b - 1)[:b]
        return p[np.isfinite(s[p])]

    # ── internal holdout: sample train users, hold out their LAST item.
    rng = np.random.default_rng(0)
    entities = [e for e, o in st.owned_by_entity.items() if o is not None and o.size >= 3]
    rng.shuffle(entities)
    Xint, yint = [], []
    for e in entities[:N_INTERNAL]:
        owned = st.owned_by_entity[e]
        hist, pos = owned[:-1], int(owned[-1])
        pool = pool_for(hist)
        if pool.size == 0:
            continue
        Xint.append(features_for(hist, pool))
        yint.append((pool == pos).astype(np.float64))
    Xint = np.vstack(Xint); yint = np.concatenate(yint)
    log(f"{dataset}: internal holdout {len(yint)} pairs, {int(yint.sum())} pos "
        f"from {min(len(entities), N_INTERNAL)} train users")
    m_lin = fit_ridge(Xint, yint)
    m_cr = fit_ridge(poly2(Xint), yint)

    # ── real test eval rows (full history, real test labels).
    test_rows = []
    for entity, relevant in eval_set.items():
        owned = st.owned_by_entity.get(entity)
        if owned is None or owned.size == 0:
            continue
        pool = pool_for(owned)
        feat = features_for(owned, pool)
        test_rows.append((pool, feat, relevant))

    def ndcg(score_fn):
        per = []
        for pool, feat, relevant in test_rows:
            order = np.argsort(-score_fn(feat))[:K]
            per.append(([st.item_ids[int(pool[o])] for o in order], relevant))
        return round(aggregate(per, catalog_size=catalog, k=K).ndcg_at_k, 4)

    w_cur = np.array([1, 0, 0, 0.25, 0.5, 0.0])
    res = {
        "dataset": dataset, "n_test_users": len(test_rows),
        "internal_pos": int(yint.sum()),
        "current_blend": ndcg(lambda f: f @ w_cur),
        "deploy_ridge_linear": ndcg(lambda f: apply_ridge(m_lin, f)),
        "deploy_ridge_cross2": ndcg(lambda f: apply_ridge(m_cr, poly2(f))),
    }
    # ceiling for contrast: ridge_cross2 trained on eval-half test labels.
    half = len(test_rows) // 2
    Xc = np.vstack([poly2(r[1]) for r in test_rows[:half]])
    yc = np.concatenate([np.array([1.0 if st.item_ids[int(c)] in r[2] else 0.0
                                   for c in r[0]]) for r in test_rows[:half]])
    m_ceil = fit_ridge(Xc, yc)
    per = []
    for pool, feat, relevant in test_rows[half:]:
        order = np.argsort(-apply_ridge(m_ceil, poly2(feat)))[:K]
        per.append(([st.item_ids[int(pool[o])] for o in order], relevant))
    res["ceiling_cross2_evalhalf"] = round(aggregate(per, catalog_size=catalog, k=K).ndcg_at_k, 4)
    res["current_evalhalf"] = round(aggregate(
        [([st.item_ids[int(r[0][o])] for o in np.argsort(-(r[1] @ w_cur))[:K]], r[2])
         for r in test_rows[half:]], catalog_size=catalog, k=K).ndcg_at_k, 4)

    log(f"{dataset} deployability (full test eval, n={len(test_rows)}):")
    for k in ("current_blend", "deploy_ridge_linear", "deploy_ridge_cross2"):
        log(f"    {k:24s} {res[k]}")
    log(f"  contrast (eval-half): current {res['current_evalhalf']} vs "
        f"ceiling_cross2 {res['ceiling_cross2_evalhalf']}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"rerank_deploy_{dataset}.json").write_text(json.dumps(res, indent=2) + "\n")
    log(f"[wrote] rerank_deploy_{dataset}.json")


if __name__ == "__main__":
    main()
