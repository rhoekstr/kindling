"""§7.2 ml1m ranking probe: is the 0.64 oracle gap closeable in-philosophy?

ml1m is ranking-bound — the EASE pool holds the answers (oracle 0.93) but the
scorer delivers 0.29 (run_gap_decomp). This probes, on the engine's ACTUAL pool,
whether shallow/closed-form re-ranking moves it, and uses a trained re-ranker as
a CEILING diagnostic:

  - reconstruct the current blend over the pool (validate ≈ 0.293)
  - add in-philosophy channels: recent-window EASE (last-K owned), and sweep
    trend / last-item weights (the levers not yet exhausted)
  - LEARNED CEILING: fit logistic + gradient-boosted re-rankers on the SAME
    shallow features (split eval users 50/50, fit on one half, score the other)

Reading: if the learned ceiling ≈ baseline, the signal is missing from the
features → the gap is sequential (out of philosophy). If the learned ceiling
≫ baseline, a better ranker/blend is the lever and the decision is philosophical.

Run: .venv/bin/python bench/run_ml1m_rerank.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set
from kindling.engine_v2 import EngineV2

K = 10
REPORT = Path(__file__).parent / "reports" / "ml1m_rerank.json"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def zc(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def main() -> None:
    split = _load_dataset("movielens-1m", test_fraction=0.1)
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
    log(f"fit done base={st.profile['base_scorer_used']} trend_a={st.trend_alpha} "
        f"lastitem_a={st.last_item_alpha}")

    # Per-user pool + per-candidate features (z-normalized over the catalog, as
    # the engine does), restricted to the engine's actual pool.
    FEATS = ["ease_full", "ease_r3", "ease_r10", "lastitem", "trend", "pop"]
    rows = []  # (entity, pool_idx[np], feat[pool×F], label[pool])
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

        ease_full = zc(base_vec)
        ease_r3 = zc(ease[owned[-3:]].sum(0))
        ease_r10 = zc(ease[owned[-10:]].sum(0))
        lastitem = zc(ease[int(owned[-1])])
        feat = np.stack([ease_full[pool], ease_r3[pool], ease_r10[pool],
                         lastitem[pool], trend_z[pool], pop_z[pool]], axis=1)
        rel_idx = {st.item_to_idx.get(r) for r in relevant}
        label = np.array([1.0 if int(c) in rel_idx else 0.0 for c in pool])
        rows.append((entity, pool, feat, label, relevant))

    log(f"built features for {len(rows)} users")

    def ndcg_for(score_fn, subset):
        per = []
        for entity, pool, feat, label, relevant in subset:
            s = score_fn(feat)
            order = np.argsort(-s)[:K]
            per.append(([st.item_ids[int(pool[o])] for o in order], relevant))
        return aggregate(per, catalog_size=catalog, k=K).ndcg_at_k

    # weight vectors over FEATS = [ease_full, ease_r3, ease_r10, lastitem, trend, pop]
    def wfn(w):
        w = np.array(w, dtype=np.float64)
        return lambda feat: feat @ w

    arms = {
        "current_blend(ease+0.5trend+0.25last)": wfn([1, 0, 0, 0.25, 0.5, 0]),
        "+recent3@0.5":  wfn([1, 0.5, 0, 0.25, 0.5, 0]),
        "+recent10@0.5": wfn([1, 0, 0.5, 0.25, 0.5, 0]),
        "+recent10@1.0": wfn([1, 0, 1.0, 0.25, 0.5, 0]),
        "trend@1.0":     wfn([1, 0, 0, 0.25, 1.0, 0]),
        "trend@2.0":     wfn([1, 0, 0, 0.25, 2.0, 0]),
        "lastitem@1.0":  wfn([1, 0, 0, 1.0, 0.5, 0]),
        "pop@0.5":       wfn([1, 0, 0, 0.25, 0.5, 0.5]),
    }
    results = {}
    for name, fn in arms.items():
        results[name] = round(ndcg_for(fn, rows), 4)
        log(f"  {name:42s} NDCG@10={results[name]}")

    # ── learned ceiling: split eval users 50/50, fit on half, score the other.
    half = len(rows) // 2
    fit_rows, eval_rows = rows[:half], rows[half:]
    Xtr = np.vstack([r[2] for r in fit_rows])
    ytr = np.concatenate([r[3] for r in fit_rows])
    log(f"learned: fit on {len(fit_rows)} users ({len(ytr)} pairs, "
        f"{int(ytr.sum())} pos), eval on {len(eval_rows)}")
    # baseline on the SAME eval-half for apples-to-apples.
    results["__eval_half_current"] = round(
        ndcg_for(arms["current_blend(ease+0.5trend+0.25last)"], eval_rows), 4)
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
        results["learned_logreg(eval_half)"] = round(
            ndcg_for(lambda f: lr.predict_proba(f)[:, 1], eval_rows), 4)
    except Exception as e:  # noqa: BLE001
        results["learned_logreg(eval_half)"] = f"FAILED {e}"
    try:
        import lightgbm as lgb
        gbm = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                                 verbose=-1).fit(Xtr, ytr)
        results["learned_gbm(eval_half)"] = round(
            ndcg_for(lambda f: gbm.predict_proba(f)[:, 1], eval_rows), 4)
    except Exception as e:  # noqa: BLE001
        results["learned_gbm(eval_half)"] = f"FAILED {e}"

    log("learned ceiling (eval-half):")
    for k in ("__eval_half_current", "learned_logreg(eval_half)", "learned_gbm(eval_half)"):
        log(f"  {k:42s} {results[k]}")
    log("references: pop_floor 0.249  current(full) 0.293  oracle_pool 0.932")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2) + "\n")
    log(f"[wrote] {REPORT}")


if __name__ == "__main__":
    main()
