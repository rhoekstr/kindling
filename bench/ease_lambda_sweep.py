"""Diagnostic: how far is the auto-λ heuristic from optimal, on kindling's OWN
chronological splits (the eval that drives the reference numbers)?

The heuristic is eff_lambda = 20 · n_interactions / n_items (≈ 20× the mean Gram
diagonal). This sweeps ease_lambda around it and reports NDCG@10 per value, so we
can see the headroom and where the optimum actually sits.

Run: python bench/ease_lambda_sweep.py [dataset ...]
"""

from __future__ import annotations

import sys

from kindling import Engine
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set

_CONFIG = {
    "movielens-1m": {},
    "amazon-beauty": {"ease_lambda": None},  # override removed below
    "steam": {"cold_slots": 1},
}
# multipliers of the heuristic λ to probe
MULTS = [0.1, 0.15, 0.25, 0.4, 0.6, 1.0, 1.6]


def evaluate_lambda(dataset: str, train, test, items, has_meta, lam, eval_set, cat):
    cfg = {k: v for k, v in _CONFIG.get(dataset, {}).items() if v is not None}
    eng = Engine(retrieval_budget=500, random_state=0, ease_lambda=lam, **cfg).fit(
        train, item_metadata=items if has_meta else None
    )
    per = []
    for ent, rel in eval_set.items():
        recs = eng.recommend(ent, 10)
        per.append(([r.item_id for r in recs], rel))
    return float(aggregate(per, catalog_size=cat, k=10).ndcg_at_k), eng._state.n_items


def main(argv: list[str]) -> int:
    datasets = argv[1:] or ["movielens-1m", "amazon-beauty"]
    for ds in datasets:
        split = _load_dataset(ds, 0.1)
        has_meta = getattr(split, "items", None) is not None
        items = split.items if has_meta else None
        eval_set = _build_eval_set(split.train, split.test, max_users=500, seed=0)
        # heuristic λ uses the *preprocessed* interaction/item counts; approximate
        # with the raw train here, then report the engine's own n_items.
        n_int = len(split.train)
        # fit once at auto to read the engine's resolved λ + n_items
        eng0 = Engine(retrieval_budget=500, random_state=0,
                      **{k: v for k, v in _CONFIG.get(ds, {}).items() if v is not None}).fit(
            split.train, item_metadata=items)
        heur = float(eng0._state.profile.get("ease_lambda", 20.0 * n_int / max(eng0._state.n_items, 1)))
        cat = max(eng0._state.n_items, 1)
        base = eng0._state.base_scorer_used
        print(f"\n=== {ds}  base={base}  heuristic_λ={heur:.0f}  n_items={eng0._state.n_items} ===")
        results = []
        for m in MULTS:
            lam = heur * m
            ndcg, _ = evaluate_lambda(ds, split.train, split.test, items, has_meta, lam, eval_set, cat)
            results.append((lam, m, ndcg))
            print(f"  λ={lam:8.0f} ({m:>4.2f}× heuristic)  NDCG@10={ndcg:.4f}")
        best = max(results, key=lambda r: r[2])
        heur_ndcg = next(r[2] for r in results if abs(r[1] - 1.0) < 1e-9)
        print(f"  → best λ={best[0]:.0f} ({best[1]:.2f}×) NDCG={best[2]:.4f}  vs heuristic NDCG={heur_ndcg:.4f}  "
              f"headroom=+{best[2]-heur_ndcg:.4f} ({100*(best[2]-heur_ndcg)/max(heur_ndcg,1e-9):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
