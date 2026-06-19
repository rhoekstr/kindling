"""§7.2 beauty retrieval diagnostic: is pool_recall=0 fixable by candidate sources?

Beauty is retrieval-bound — median pool recall 0 (run_gap_decomp): half of eval
users' held-out items never reach the EASE-blended pool. Before building a
multi-source retriever, measure whether the missing relevant items are
*reachable at all* by an alternative candidate source, or are fundamentally
unreachable (not in catalog / no signal path). Pure recall@budget, per source:

  catalog_ceiling  fraction of relevant items present in train at all (no
                   retrieval can exceed this)
  current          EASE-blended pool (the shipped retriever; user-CF already
                   folded into the blend on beauty)
  popularity       top-budget most-popular non-owned
  content          item-feature cosine to owned (2014 meta)
  cooc_2hop        neighbors-of-neighbors over the cooc graph
  UNION            reachable by ANY source (upper bound for a multi-source pool)

UNION ≫ current -> retrieval expansion is viable (build a multi-source pool).
UNION ≈ current -> beauty is signal/catalog-bound; retrieval can't help.

Run: .venv/bin/python bench/run_beauty_retrieval.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.parity import _build_eval_set
from kindling.engine_v2 import EngineV2

BUDGET = 500
REPORT = Path(__file__).parent / "reports" / "beauty_retrieval.json"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    dataset = os.environ.get("DATASET", "amazon-beauty")
    split = _load_dataset(dataset, test_fraction=0.1)
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=500, seed=0)
    # Fit WITHOUT item_metadata: match the gap-decomp setting that measured
    # pool_recall=0 (no open-catalog extension). Content is inert on beauty.
    eng = EngineV2(persona_min_users=10**9, retrieval_budget=BUDGET, random_state=0)
    eng.fit(train)
    st = eng._state
    n_items = st.n_items
    ease = st.ease_b.astype(np.float64) if st.ease_b is not None else None

    pop = np.zeros(n_items)
    col = train["item_id"].map(st.item_to_idx).dropna().astype(int).to_numpy()
    np.add.at(pop, col, 1)
    pop_top = np.argsort(-pop)  # global popularity order
    has_content = st.content_features is not None and st.content_features.n_features > 0
    log(f"{dataset}: items={n_items} base={st.profile['base_scorer_used']} "
        f"content={'yes' if has_content else 'no'} usercf={st.profile.get('user_cf_channel_active')}")

    def topset(scores, owned_set, budget=BUDGET):
        scores = scores.copy()
        for o in owned_set:
            if o < scores.size:
                scores[o] = -np.inf
        b = min(budget, scores.size)
        top = np.argpartition(-scores, b - 1)[:b]
        return {int(c) for c in top if np.isfinite(scores[c])}

    def cooc_row_sum(owned):
        v = np.zeros(n_items)
        for it in owned:
            s_, e_ = int(st.cooc_indptr[it]), int(st.cooc_indptr[it + 1])
            if e_ > s_:
                v[st.cooc_indices[s_:e_]] += st.cooc_data[s_:e_]
        return v

    srcs = ["current", "popularity", "content", "cooc_2hop", "UNION"]
    rec = {s: [] for s in srcs}
    cat_ceiling = []
    catalog_set = set(range(n_items))
    for entity, relevant in eval_set.items():
        owned = st.owned_by_entity.get(entity)
        if owned is None or owned.size == 0:
            continue
        owned_set = set(int(i) for i in owned.tolist())
        rel_idx = {st.item_to_idx.get(r) for r in relevant}
        rel_idx = {r for r in rel_idx if r is not None}
        n_rel = len(rel_idx)
        if n_rel == 0:
            continue
        cat_ceiling.append(len(rel_idx & catalog_set) / n_rel)

        pools = {}
        # current: EASE-blended pool (mirrors recommend on the EASE path)
        base = ease[owned].sum(0)
        blended = eng._blend_channels(st, owned, base.copy(),
                                      user_row=st.entity_to_user_idx.get(entity, -1))
        pools["current"] = topset(blended, owned_set)
        # popularity backfill
        pools["popularity"] = set(
            [int(c) for c in pop_top if int(c) not in owned_set][:BUDGET]
        )
        # content cosine to owned
        if has_content:
            from kindling.item_features import content_scores
            pools["content"] = topset(content_scores(st.content_features, owned), owned_set)
        else:
            pools["content"] = set()
        # 2-hop cooc
        onehop = cooc_row_sum(owned)
        hop1_items = [int(i) for i in np.argsort(-onehop)[:200] if onehop[i] > 0]
        twohop = cooc_row_sum(np.array(hop1_items)) if hop1_items else np.zeros(n_items)
        pools["cooc_2hop"] = topset(onehop + 0.5 * twohop, owned_set)
        pools["UNION"] = set().union(*(pools[s] for s in
                                       ["current", "popularity", "content", "cooc_2hop"]))

        for s in srcs:
            rec[s].append(len(rel_idx & pools[s]) / n_rel)

    out = {"dataset": dataset, "budget": BUDGET, "n_users": len(cat_ceiling),
           "catalog_ceiling_mean": round(float(np.mean(cat_ceiling)), 4)}
    log(f"recall@{BUDGET} (catalog ceiling {out['catalog_ceiling_mean']}):")
    for s in srcs:
        out[f"{s}_recall_mean"] = round(float(np.mean(rec[s])), 4)
        out[f"{s}_recall_median"] = round(float(np.median(rec[s])), 4)
        log(f"    {s:12s} mean={out[f'{s}_recall_mean']:.4f}  median={out[f'{s}_recall_median']:.4f}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2) + "\n")
    log(f"[wrote] {REPORT}")


if __name__ == "__main__":
    main()
