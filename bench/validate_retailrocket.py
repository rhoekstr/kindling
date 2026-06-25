"""Real-world validation: kindling vs popularity on RetailRocket.

RetailRocket is a production e-commerce clickstream (view / add-to-cart /
transaction) with extreme churn: ~1.27M users, ~224k items, ~2 events/user,
86% of eval users with <=4 interactions. This is the data-starved cold regime
the academic 5-core splits delete entirely — and the regime where the warming
benchmark (REFERENCE 3.5) found the popularity prior is the bar to beat. So
the meaningful question isn't "what's kindling's NDCG" but "does
personalization add value over popularity here, and for which users?"

Protocol: realistic tier — no k-core, chronological 90/10 split (loader),
full-catalog ranking, k=20, sliced by user history length. kindling vs a
popularity baseline (top items by train frequency).

Run: python bench/validate_retailrocket.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from kindling import Engine
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate

K = 20
N_EVAL = 8000


def _ndcg_recall(per: list[tuple[list, set]], catalog: int) -> tuple[float, float]:
    rep = aggregate(per, catalog_size=catalog, k=K)
    return round(float(rep.ndcg_at_k), 4), round(float(rep.recall_at_k), 4)


def main() -> None:
    split = _load_dataset("retailrocket", 0.1)
    train, test = split.train, split.test
    train_owned = train.groupby("entity_id")["item_id"].apply(lambda x: set(x.tolist()))
    test_by = test.groupby("entity_id")["item_id"].apply(lambda x: set(x.tolist()))
    train_users = set(train_owned.index)
    eval_users = [u for u in test_by.index if u in train_users]
    random.seed(0)
    eval_users = random.sample(eval_users, min(N_EVAL, len(eval_users)))

    # Popularity baseline: top items by train interaction count.
    pop_order = train["item_id"].value_counts().index.tolist()

    t0 = time.perf_counter()
    eng = Engine(persona_min_users=10**9, random_state=0, open_catalog=False)
    eng.fit(train)
    fit_s = time.perf_counter() - t0
    n_items = eng._state.n_items

    # Per-user recs for both models, bucketed by history length.
    buckets = {"1-4": [], "5-19": [], "20+": [], "all": []}  # type: ignore[var-annotated]
    pop_buckets = {"1-4": [], "5-19": [], "20+": [], "all": []}  # type: ignore[var-annotated]
    for u in eval_users:
        owned = train_owned.get(u, set())
        held = test_by[u] - owned
        if not held:
            continue
        h = len(owned)
        b = "1-4" if h <= 4 else ("5-19" if h < 20 else "20+")
        k_recs = [r.item_id for r in eng.recommend(entity_id=u, n=K)]
        p_recs = [i for i in pop_order if i not in owned][:K]
        for key in (b, "all"):
            buckets[key].append((k_recs, held))
            pop_buckets[key].append((p_recs, held))

    print(
        f"\nRetailRocket  (n_items={n_items:,}, fit {fit_s:.0f}s, "
        f"base={eng.activation_plan.base_scorer}, channels={eng.activation_plan.active_channels})"
    )
    print(
        f"{'bucket':<8}{'n':>7}{'kindling NDCG':>15}{'pop NDCG':>11}{'kindling Rec':>14}{'pop Rec':>10}"
    )
    out: dict = {
        "dataset": "retailrocket",
        "fit_seconds": round(fit_s, 1),
        "base": eng.activation_plan.base_scorer,
        "channels": eng.activation_plan.active_channels,
        "buckets": {},
    }
    for b in ("1-4", "5-19", "20+", "all"):
        if not buckets[b]:
            continue
        kn, kr = _ndcg_recall(buckets[b], n_items)
        pn, pr = _ndcg_recall(pop_buckets[b], n_items)
        win = "kindling" if kn > pn else "popularity"
        print(
            f"{b:<8}{len(buckets[b]):>7}{kn:>15.4f}{pn:>11.4f}{kr:>14.4f}{pr:>10.4f}   {win} wins NDCG"
        )
        out["buckets"][b] = {
            "n": len(buckets[b]),
            "kindling": {"ndcg": kn, "recall": kr},
            "popularity": {"ndcg": pn, "recall": pr},
            "ndcg_winner": win,
        }
    Path("bench/reports/validate_retailrocket.json").write_text(json.dumps(out, indent=2))
    print("\nWrote bench/reports/validate_retailrocket.json")


if __name__ == "__main__":
    main()
