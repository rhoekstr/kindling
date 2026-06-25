"""Real-world generalization check: kindling vs published GNN baselines on
the standard yelp2018 benchmark (the exact LightGCN/NGCF academic split).

yelp2018 is a *new domain* (local-business recommendation) versus the four
datasets kindling was tuned on, and it has widely-published full-ranking
numbers — so it's an external, apples-to-apples test of whether the
value-add generalizes. No timestamps in the academic split, so kindling
runs its weakest config (wilson-cooc base, channels no-op), exactly like
the book-academic comparison (REFERENCE §3.4).

Protocol: full-catalog ranking, k=20, train items excluded, evaluated over
a 5000-user sample. Run: python bench/validate_yelp2018.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from kindling import Engine
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate

# Published full-ranking yelp2018 results (He et al., LightGCN, SIGIR 2020).
PUBLISHED = {
    "BPR-MF": (0.0549, 0.0445),
    "Mult-VAE": (0.0584, 0.0450),
    "NGCF": (0.0579, 0.0477),
    "LightGCN": (0.0649, 0.0530),
}
N_EVAL = 5000
K = 20


def main() -> None:
    split = _load_dataset("yelp2018", 0.1)
    train, test = split.train, split.test
    test_by_u = test.groupby("entity_id")["item_id"].apply(lambda x: set(x.tolist()))
    train_users = set(train.entity_id.unique())
    eval_users = [u for u in test_by_u.index if u in train_users][:N_EVAL]

    t0 = time.perf_counter()
    eng = Engine(persona_min_users=10**9, random_state=0, retrieval_budget=1000)
    eng.fit(train)
    fit_s = time.perf_counter() - t0

    per = [([r.item_id for r in eng.recommend(entity_id=u, n=K)], test_by_u[u]) for u in eval_users]
    rep = aggregate(per, catalog_size=eng._state.n_items, k=K)
    recall, ndcg = round(float(rep.recall_at_k), 4), round(float(rep.ndcg_at_k), 4)

    print(
        f"\nkindling on yelp2018  (n_items={eng._state.n_items:,}, fit {fit_s:.0f}s, zero training)"
    )
    print(
        f"  base={eng.activation_plan.base_scorer}  channels={eng.activation_plan.active_channels}"
    )
    print(f"\n{'model':<12}{'Recall@20':>11}{'NDCG@20':>10}")
    rows = [("kindling", recall, ndcg), *[(m, r, n) for m, (r, n) in PUBLISHED.items()]]
    for m, r, n in sorted(rows, key=lambda x: x[2]):
        mark = "  <-- no training, CPU" if m == "kindling" else ""
        print(f"{m:<12}{r:>11.4f}{n:>10.4f}{mark}")

    beats = [m for m, (r, n) in PUBLISHED.items() if ndcg > n]
    out = {
        "dataset": "yelp2018",
        "kindling": {
            "recall@20": recall,
            "ndcg@20": ndcg,
            "fit_seconds": round(fit_s, 1),
            "base": eng.activation_plan.base_scorer,
            "n_eval": len(eval_users),
        },
        "published": {m: {"recall@20": r, "ndcg@20": n} for m, (r, n) in PUBLISHED.items()},
        "kindling_beats_on_ndcg": beats,
    }
    Path("bench/reports/validate_yelp2018.json").write_text(json.dumps(out, indent=2))
    print(
        f"\nkindling NDCG@20 beats: {beats or 'none'}; "
        f"{ndcg / PUBLISHED['LightGCN'][1]:.0%} of LightGCN. Wrote bench/reports/validate_yelp2018.json"
    )


if __name__ == "__main__":
    main()
