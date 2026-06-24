"""§7.2 learned re-ranker evaluation: is the +7% real, general, and dep-free?

The ml1m probe found a LightGBM re-ranker over the fixed channel features beats
the z-blend +7% while a linear ranker does not — i.e. the gain is non-linear
feature interaction. But LightGBM is a compiled C++ runtime dep, against the
"a wheel that imports works" philosophy (numpy/pandas/scipy only). So the real
questions:

  1. GENERALIZE — does the lift hold on steam / beauty, not just ml1m?
  2. DEP-FREE — can numpy feature-crosses + closed-form ridge (no new dep)
     recover the tree gain, or does it genuinely need trees?

Arms (all on the engine's actual pool, 50/50 user split, score the eval-half):
  current_blend   the shipped z-blend (reference)
  ridge_linear    closed-form ridge on the 6 features (numpy)
  ridge_cross2    closed-form ridge on features + degree-2 crosses (numpy) ← in-philosophy candidate
  logreg          linear logistic (sklearn)
  gbm             LightGBM (the dep-requiring ceiling)

Run: DATASET=movielens-1m .venv/bin/python bench/run_rerank.py   (| steam | amazon-beauty)
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
REPORT_DIR = Path(__file__).parent / "reports"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def zc(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def poly2(X: np.ndarray) -> np.ndarray:
    """Linear features + all degree-2 products (incl. squares). numpy-only."""
    n, d = X.shape
    cols = [X]
    for i in range(d):
        cols.append(X[:, i:] * X[:, i][:, None])
    return np.hstack(cols)


def ridge_fit_predict(Xtr, ytr, Xte, lam=10.0):
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-9)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    Ztr = np.hstack([Ztr, np.ones((len(Ztr), 1))])
    Zte = np.hstack([Zte, np.ones((len(Zte), 1))])
    W = np.linalg.solve(Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1]), Ztr.T @ ytr)
    return Zte @ W


def main() -> None:
    dataset = os.environ.get("DATASET", "movielens-1m")
    split = _load_dataset(dataset, test_fraction=0.1)
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=500, seed=0)
    eng = EngineV2(persona_min_users=10**9, retrieval_budget=500, random_state=0)
    t0 = time.perf_counter()
    eng.fit(train)
    st = eng._state
    if st.ease_b is None:
        raise SystemExit(f"{dataset} is not on the EASE path; re-rank features assume ease_b")
    n_items = st.n_items
    catalog = max(n_items, 1)
    ease = st.ease_b.astype(np.float64)
    trend_z = st.trend_z if st.trend_z is not None else np.zeros(n_items)
    pop = np.zeros(n_items)
    col = train["item_id"].map(st.item_to_idx).dropna().astype(int).to_numpy()
    np.add.at(pop, col, 1)
    pop_z = zc(np.log1p(pop))
    log(f"{dataset}: fit {time.perf_counter()-t0:.0f}s base={st.profile['base_scorer_used']}")

    rows = []
    for entity, relevant in eval_set.items():
        owned = st.owned_by_entity.get(entity)
        if owned is None or owned.size == 0:
            continue
        base_vec = ease[owned].sum(0)
        scores = eng._blend_channels(st, owned, base_vec.copy(),
                                     user_row=st.entity_to_user_idx.get(entity, -1))
        scores[owned] = -np.inf
        b = min(500, scores.size)
        pool = np.argpartition(-scores, b - 1)[:b]
        pool = pool[np.isfinite(scores[pool])]
        feat = np.stack([
            zc(base_vec)[pool], zc(ease[owned[-3:]].sum(0))[pool],
            zc(ease[owned[-10:]].sum(0))[pool], zc(ease[int(owned[-1])])[pool],
            trend_z[pool], pop_z[pool]], axis=1)
        rel_idx = {st.item_to_idx.get(r) for r in relevant}
        label = np.array([1.0 if int(c) in rel_idx else 0.0 for c in pool])
        rows.append((pool, feat, label, relevant))

    def ndcg(score_fn, subset):
        per = []
        for pool, feat, label, relevant in subset:
            order = np.argsort(-score_fn(feat))[:K]
            per.append(([st.item_ids[int(pool[o])] for o in order], relevant))
        return round(aggregate(per, catalog_size=catalog, k=K).ndcg_at_k, 4)

    half = len(rows) // 2
    fit_rows, ev = rows[:half], rows[half:]
    Xtr = np.vstack([r[1] for r in fit_rows]); ytr = np.concatenate([r[2] for r in fit_rows])
    Xtr2 = poly2(Xtr)
    res = {"dataset": dataset, "n_users": len(rows), "fit_pos": int(ytr.sum())}
    w_cur = np.array([1, 0, 0, 0.25, 0.5, 0.0])
    res["current_blend"] = ndcg(lambda f: f @ w_cur, ev)
    res["ridge_linear"] = ndcg(lambda f: ridge_fit_predict(Xtr, ytr, f), ev)
    res["ridge_cross2"] = ndcg(lambda f: ridge_fit_predict(Xtr2, ytr, poly2(f)), ev)
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
        res["logreg"] = ndcg(lambda f: lr.predict_proba(f)[:, 1], ev)
    except Exception as e:  # noqa: BLE001
        res["logreg"] = f"skip ({type(e).__name__})"
    try:
        import lightgbm as lgb
        gbm = lgb.LGBMClassifier(n_estimators=200, num_leaves=31,
                                 learning_rate=0.05, verbose=-1).fit(Xtr, ytr)
        res["gbm"] = ndcg(lambda f: gbm.predict_proba(f)[:, 1], ev)
    except Exception as e:  # noqa: BLE001
        res["gbm"] = f"skip ({type(e).__name__})"

    log(f"{dataset} re-rank (eval-half, n={len(ev)}):")
    for k in ("current_blend", "ridge_linear", "ridge_cross2", "logreg", "gbm"):
        log(f"    {k:16s} {res[k]}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"rerank_{dataset}.json").write_text(json.dumps(res, indent=2) + "\n")
    log(f"[wrote] rerank_{dataset}.json")


if __name__ == "__main__":
    main()
