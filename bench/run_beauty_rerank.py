"""§7.2 beauty: does multi-source SCORING (not just pooling) lift end-to-end NDCG?

The coverage diagnostic (run_beauty_retrieval.py) showed the missing relevant
items ARE reachable — union recall 0.28→0.38, median 0→0.25 via popularity +
2-hop. But on the EASE path the engine ranks the full catalog by the blend, so
pooling alone changes nothing: the new items must score higher. This tests
whether adding 2-hop-cooc / popularity as FIXED-weight channels (deployable,
unlike the §4.4-undeployable learned re-ranker) lifts NDCG@10 over the union
pool — and whether the oracle over the union pool is even worth chasing.

Arms (fixed weights over the union pool, owned excluded):
  current            z(ease_blend)                     (≈ shipped beauty NDCG)
  +2hop@a            z(ease_blend) + a·z(cooc_2hop)
  +pop@b             z(ease_blend) + b·z(pop)
  +both              z(ease_blend) + a·z(2hop) + b·z(pop)
  oracle_union       relevant-in-union-pool first      (the new ceiling)

Run: .venv/bin/python bench/run_beauty_rerank.py
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
BUDGET = 500
REPORT = Path(__file__).parent / "reports" / "beauty_rerank.json"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def zc(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v * 0.0


def main() -> None:
    split = _load_dataset("amazon-beauty", test_fraction=0.1)
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=500, seed=0)
    eng = EngineV2(persona_min_users=10**9, retrieval_budget=BUDGET, random_state=0)
    eng.fit(train)
    st = eng._state
    n_items = st.n_items
    catalog = max(n_items, 1)
    ease = st.ease_b.astype(np.float64)
    pop = np.zeros(n_items)
    col = train["item_id"].map(st.item_to_idx).dropna().astype(int).to_numpy()
    np.add.at(pop, col, 1)
    pop_top = np.argsort(-pop)
    pop_log = np.log1p(pop)

    def cooc_sum(items):
        v = np.zeros(n_items)
        for it in items:
            s_, e_ = int(st.cooc_indptr[it]), int(st.cooc_indptr[it + 1])
            if e_ > s_:
                v[st.cooc_indices[s_:e_]] += st.cooc_data[s_:e_]
        return v

    rows = []
    for entity, relevant in eval_set.items():
        owned = st.owned_by_entity.get(entity)
        if owned is None or owned.size == 0:
            continue
        owned_set = set(int(i) for i in owned.tolist())
        base = ease[owned].sum(0)
        blend = eng._blend_channels(st, owned, base.copy(),
                                    user_row=st.entity_to_user_idx.get(entity, -1))
        blend[owned] = -np.inf
        onehop = cooc_sum(owned)
        hop1 = [int(i) for i in np.argsort(-onehop)[:200] if onehop[i] > 0]
        twohop = cooc_sum(np.array(hop1)) if hop1 else np.zeros(n_items)
        # union pool: top-BUDGET blend ∪ top-BUDGET pop ∪ top-BUDGET 2hop
        pool = set(int(c) for c in np.argpartition(-blend, BUDGET - 1)[:BUDGET]
                   if np.isfinite(blend[c]))
        pool |= set([int(c) for c in pop_top if int(c) not in owned_set][:BUDGET])
        h2 = onehop + 0.5 * twohop
        pool |= set(int(c) for c in np.argsort(-h2)[:BUDGET]
                    if h2[c] > 0 and int(c) not in owned_set)
        pool -= owned_set
        pool = np.array(sorted(pool))
        rel_idx = {st.item_to_idx.get(r) for r in relevant}
        # z-normalize over the POOL values (blend has -inf at owned, which would
        # poison a catalog-wide mean/std; pool already excludes owned).
        feat = {"blend": zc(blend[pool]), "twohop": zc(h2[pool]), "pop": zc(pop_log[pool])}
        rows.append((pool, feat, relevant, rel_idx))

    def ndcg(score_of):
        per = []
        for pool, feat, relevant, _ in rows:
            order = np.argsort(-score_of(feat))[:K]
            per.append(([st.item_ids[int(pool[o])] for o in order], relevant))
        return round(aggregate(per, catalog_size=catalog, k=K).ndcg_at_k, 4)

    res = {"dataset": "amazon-beauty", "n_users": len(rows), "shipped_ref": 0.0325}
    res["current(blend)"] = ndcg(lambda f: f["blend"])
    for a in (0.25, 0.5, 1.0):
        res[f"+2hop@{a}"] = ndcg(lambda f, a=a: f["blend"] + a * f["twohop"])
    for b in (0.25, 0.5):
        res[f"+pop@{b}"] = ndcg(lambda f, b=b: f["blend"] + b * f["pop"])
    res["+2hop@0.5+pop@0.25"] = ndcg(lambda f: f["blend"] + 0.5 * f["twohop"] + 0.25 * f["pop"])

    # oracle over the union pool (new achievable ceiling)
    per = []
    for pool, feat, relevant, rel_idx in rows:
        oracle = [st.item_ids[int(c)] for c in pool if int(c) in rel_idx][:K]
        for c in pool:
            if len(oracle) >= K:
                break
            if int(c) not in rel_idx:
                oracle.append(st.item_ids[int(c)])
        per.append((oracle, relevant))
    res["oracle_union"] = round(aggregate(per, catalog_size=catalog, k=K).ndcg_at_k, 4)

    log(f"beauty re-rank over union pool (n={len(rows)}, shipped ref 0.0325):")
    for k, v in res.items():
        if k not in ("dataset", "n_users", "shipped_ref"):
            log(f"    {k:22s} {v}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(res, indent=2) + "\n")
    log(f"[wrote] {REPORT}")


if __name__ == "__main__":
    main()
