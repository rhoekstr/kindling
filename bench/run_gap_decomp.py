"""Gap decomposition on the CURRENT engine (§7.2 re-grounding).

The library diagnostic (benchmarks/gap_decomposition.py) computes the candidate
pool via raw cooccurrence_retrieve — stale since the pivot: the engine now
retrieves from the EASE-blended scores on the EASE path. This version mirrors
recommend()'s ACTUAL retrieval so oracle_pool / pool_recall reflect the engine
the user runs. Numbers to compare against the pre-pivot 2026-06-10 brackets
(ml1m current 0.256 / oracle 0.885 / pool_recall 0.56).

Decomposition:
  (oracle_pool − current) = lift available from better SCORING (ranking-bound)
  (1 − pool_recall)       = relevance lost to RETRIEVAL misses (retrieval-bound)
  (current − pop_floor)   = value added over trivial

Run: DATASETS=movielens-1m,amazon-beauty .venv/bin/python bench/run_gap_decomp.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import kindling_core
import numpy as np

from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set
from kindling.engine import Engine

K = 10
REPORT = Path(__file__).parent / "reports" / "gap_decomp_current.json"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def engine_pool(engine: Engine, entity, owned: np.ndarray) -> list[int]:
    """The candidate pool recommend() actually retrieves, for this entity."""
    st = engine._state
    budget = engine.retrieval_budget
    if st.ease_b is not None:
        base_vec = st.ease_b[owned].sum(axis=0, dtype=np.float64)
        if base_vec.size < st.n_items:
            base_vec = np.concatenate([base_vec, np.zeros(st.n_items - base_vec.size)])
        scores = engine._blend_channels(
            st, owned, base_vec, user_row=st.entity_to_user_idx.get(entity, -1)
        )
    elif (st.trend_z is not None and st.trend_alpha > 0.0) or (
        st.trans_data is not None and st.transition_alpha > 0.0
    ):
        scores = np.zeros(st.n_items, dtype=np.float64)
        for item in owned.tolist():
            s_, e_ = int(st.cooc_indptr[item]), int(st.cooc_indptr[item + 1])
            if e_ > s_:
                scores[st.cooc_indices[s_:e_]] += st.cooc_data[s_:e_]
        scores = engine._blend_channels(
            st, owned, scores, user_row=st.entity_to_user_idx.get(entity, -1)
        )
    else:
        cand_ids, _ = kindling_core.cooccurrence_retrieve(
            st.cooc_data,
            st.cooc_indices,
            st.cooc_indptr,
            owned_indices=owned.tolist(),
            budget=budget,
            include_owned=False,
        )
        return [int(c) for c in cand_ids]
    scores[owned] = -np.inf
    b = min(budget, scores.size)
    top = np.argpartition(-scores, b - 1)[:b]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [int(c) for c in top if np.isfinite(scores[c])]


def run(loader: str) -> dict:
    split = _load_dataset(loader, test_fraction=0.1)
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=500, seed=0)
    engine = Engine(persona_min_users=10_000_000, retrieval_budget=500, random_state=0)
    t0 = time.perf_counter()
    engine.fit(train)
    st = engine._state
    catalog = max(st.n_items, 1)

    pop_counts = np.zeros(st.n_items, dtype=np.int64)
    col = train["item_id"].map(st.item_to_idx).dropna().astype(np.int64).to_numpy()
    np.add.at(pop_counts, col, 1)
    pop_order = np.argsort(-pop_counts)

    per_pop, per_cur, per_oracle = [], [], []
    pool_recalls = []
    for entity, relevant in eval_set.items():
        owned = st.owned_by_entity.get(entity)
        if owned is None or owned.size == 0:
            continue
        owned_set = set(int(i) for i in owned.tolist())
        pop_top = []
        for i in pop_order:
            if int(i) not in owned_set:
                pop_top.append(st.item_ids[int(i)])
                if len(pop_top) >= K:
                    break
        per_pop.append((pop_top, relevant))
        per_cur.append(([r.item_id for r in engine.recommend(entity, n=K)], relevant))

        pool_ids = [st.item_ids[c] for c in engine_pool(engine, entity, owned)]
        pool_set = set(pool_ids)
        n_rel = len(relevant)
        pool_recalls.append(len(relevant & pool_set) / n_rel if n_rel else 0.0)
        oracle = [iid for iid in pool_ids if iid in relevant][:K]
        for iid in pool_ids:
            if len(oracle) >= K:
                break
            if iid not in relevant:
                oracle.append(iid)
        per_oracle.append((oracle, relevant))

    def _m(per):
        r = aggregate(per, catalog_size=catalog, k=K)
        return {
            "ndcg": round(r.ndcg_at_k, 4),
            "recall": round(r.recall_at_k, 4),
            "mrr": round(r.mrr, 4),
            "hr": round(r.hit_rate, 3),
        }

    out = {
        "loader": loader,
        "n_users": len(per_cur),
        "fit_s": round(time.perf_counter() - t0, 1),
        "base": st.profile.get("base_scorer_used"),
        "pop_floor": _m(per_pop),
        "current": _m(per_cur),
        "oracle_pool": _m(per_oracle),
        "pool_recall_mean": round(float(np.mean(pool_recalls)), 4),
        "pool_recall_median": round(float(np.median(pool_recalls)), 4),
    }
    cur, orc, pop = out["current"]["ndcg"], out["oracle_pool"]["ndcg"], out["pop_floor"]["ndcg"]
    log(f"{loader} (base={out['base']}, n={out['n_users']}):")
    log(
        f"  pop_floor={pop}  current={cur}  oracle_pool={orc}  pool_recall(med)={out['pool_recall_median']}"
    )
    log(
        f"  ranking headroom (oracle−current)={round(orc - cur, 4)}  "
        f"retrieval loss (1−pool_recall)={round(1 - out['pool_recall_median'], 4)}  "
        f"value-add (current−pop)={round(cur - pop, 4)}"
    )
    return out


def main() -> None:
    datasets = os.environ.get("DATASETS", "movielens-1m,amazon-beauty").split(",")
    results = [run(d) for d in datasets]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2) + "\n")
    log(f"[wrote] {REPORT}")


if __name__ == "__main__":
    main()
