"""Gap decomposition: where is NDCG being lost — retrieval or scoring?

For each loader, computes four numbers that bracket the current system:

  1. **pop_floor**      — global-popularity top-K (owned excluded).
                          Any system must beat this to be adding value.
  2. **current**        — the engine's cooc base (no personas), as shipped.
  3. **oracle_pool**    — perfect scorer given the SAME candidate pool:
                          relevant items in the pool ranked first. This is
                          the ceiling achievable WITHOUT touching retrieval.
  4. **pool_recall**    — fraction of held-out items present in the
                          retrieved pool at all. The retrieval ceiling:
                          recall@K can never exceed this.

Decomposition:
  (oracle_pool − current)  = lift available from better SCORING
  (1 − pool_recall)        = relevance lost to RETRIEVAL misses
  (current − pop_floor)    = how much value the system adds over trivial
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import kindling_core
import numpy as np

from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set
from kindling.engine import Engine


def run(loader: str, max_eval_users: int = 500, k: int = 10, seed: int = 0) -> dict[str, object]:
    from kindling.benchmarks.comparison import _load_dataset

    split = _load_dataset(loader, test_fraction=0.1)
    train, test = split.train, split.test
    eval_set = _build_eval_set(train, test, max_users=max_eval_users, seed=seed)
    if not eval_set:
        raise RuntimeError("eval set empty")

    engine = Engine(
        persona_min_users=10_000_000,  # personas off — isolate cooc base
        retrieval_budget=500,
        random_state=seed,
    )
    t0 = time.perf_counter()
    engine.fit(train)
    fit_s = time.perf_counter() - t0
    st = engine._state
    assert st is not None
    catalog = max(st.n_items, 1)

    # Popularity ranking from train (interaction counts per item index).
    pop_counts = np.zeros(st.n_items, dtype=np.int64)
    item_idx_col = train["item_id"].map(st.item_to_idx).dropna().astype(np.int64)
    np.add.at(pop_counts, item_idx_col.to_numpy(), 1)
    pop_order = np.argsort(-pop_counts)  # most popular first

    per_pop: list[tuple[list[object], set[object]]] = []
    per_cur: list[tuple[list[object], set[object]]] = []
    per_oracle: list[tuple[list[object], set[object]]] = []
    pool_recalls: list[float] = []
    pool_sizes: list[int] = []

    for entity, relevant in eval_set.items():
        owned = st.owned_by_entity.get(entity)
        if owned is None or owned.size == 0:
            continue
        owned_set = set(int(i) for i in owned.tolist())

        # ── pop floor: most popular non-owned items.
        pop_top: list[object] = []
        for i in pop_order:
            ii = int(i)
            if ii in owned_set:
                continue
            pop_top.append(st.item_ids[ii])
            if len(pop_top) >= k:
                break
        per_pop.append((pop_top, relevant))

        # ── current system.
        recs = engine.recommend(entity_id=entity, n=k)
        per_cur.append(([r.item_id for r in recs], relevant))

        # ── candidate pool (same retrieval the engine used).
        cand_ids, _ = kindling_core.cooccurrence_retrieve(
            st.cooc_data,
            st.cooc_indices,
            st.cooc_indptr,
            owned_indices=owned.tolist(),
            budget=engine.retrieval_budget,
            include_owned=False,
        )
        cand_ids = list(cand_ids)
        pool_sizes.append(len(cand_ids))
        pool_item_ids = [st.item_ids[int(c)] for c in cand_ids]
        pool_set = set(pool_item_ids)
        n_rel = len(relevant)
        n_rel_in_pool = len(relevant & pool_set)
        pool_recalls.append(n_rel_in_pool / n_rel if n_rel else 0.0)

        # ── oracle given pool: relevant-in-pool first, pad with rest.
        oracle_top = [iid for iid in pool_item_ids if iid in relevant][:k]
        if len(oracle_top) < k:
            for iid in pool_item_ids:
                if iid not in relevant:
                    oracle_top.append(iid)
                    if len(oracle_top) >= k:
                        break
        per_oracle.append((oracle_top, relevant))

    def _m(per: list[tuple[list[object], set[object]]]) -> dict[str, float]:
        rep = aggregate(per, catalog_size=catalog, k=k)
        return {
            "ndcg_at_k": rep.ndcg_at_k,
            "recall_at_k": rep.recall_at_k,
            "mrr": rep.mrr,
            "hit_rate": rep.hit_rate,
        }

    out = {
        "loader": loader,
        "n_users": len(per_cur),
        "k": k,
        "fit_seconds": fit_s,
        "retrieval_budget": engine.retrieval_budget,
        "pop_floor": _m(per_pop),
        "current_cooc": _m(per_cur),
        "oracle_pool": _m(per_oracle),
        "pool_recall_mean": float(np.mean(pool_recalls)),
        "pool_recall_median": float(np.median(pool_recalls)),
        "pool_size_mean": float(np.mean(pool_sizes)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", default="movielens-1m")
    parser.add_argument("--max-eval-users", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run(args.loader, max_eval_users=args.max_eval_users, k=args.k, seed=args.seed)
    payload = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
        print(f"wrote {args.output}")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
