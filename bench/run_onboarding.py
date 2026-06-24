"""New-user onboarding curve: how fast does kindling personalize a brand-new
user as they provide their first k seed interactions?

Simulates a cold-start arrival: the user is treated as anonymous
(recommend_for_items — no stored identity), given the first k items of their
history as seeds, and evaluated on their held-out test items. Sweeps k = 0..10.

Baseline = popularity (what you serve a new user without personalization, and
what trained MF/ALS reduce to — they cannot serve a user absent from training
without retraining). The point: kindling converts a couple of seed clicks into
real personalization with NO per-user training, while popularity is flat.

Run: DATASET=amazon-beauty .venv/bin/python bench/run_onboarding.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from kindling.benchmarks.metrics import aggregate
from kindling.engine_v2 import EngineV2
from run_warming_curve import load_split

K = 10
SEEDS = [0, 1, 2, 3, 5, 10]
REPORT_DIR = Path(__file__).parent / "reports"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    dataset = os.environ.get("DATASET", "amazon-beauty")
    max_eval = int(os.environ.get("MAX_EVAL", "3000"))
    split = load_split(dataset)
    train, test = split.train, split.test
    # chronological per-user train sequence (loader is time-ordered) + test set
    train_seq = train.groupby("entity_id", sort=False)["item_id"].apply(list)
    test_by = test.groupby("entity_id", sort=False)["item_id"].apply(lambda s: set(s))
    catalog = int(train["item_id"].nunique())

    pop_prior_c = float(os.environ.get("COLD_POP_PRIOR", "8.0"))
    eng = EngineV2(persona_min_users=10**9, retrieval_budget=500, random_state=0).fit(train)
    st = eng._state
    pop = st.item_popularity if st.item_popularity is not None else np.zeros(st.n_items)
    pop_order = [st.item_ids[int(i)] for i in np.argsort(-pop)[: 4 * K + max(SEEDS)]]

    # eval users: have test items AND enough train history to draw max(SEEDS) seeds
    eval_users = [e for e in test_by.index
                  if e in train_seq.index and len(train_seq[e]) >= max(SEEDS)]
    rng = np.random.default_rng(0)
    rng.shuffle(eval_users)
    eval_users = eval_users[:max_eval]
    log(f"{dataset}: eval_users {len(eval_users)} catalog {catalog:,} seeds {SEEDS} "
        f"pop_prior_c={pop_prior_c}")

    rows = []
    for k in SEEDS:
        per_raw, per_shrunk, per_pop = [], [], []
        for e in eval_users:
            seeds = train_seq[e][:k]
            rel = test_by[e] - set(seeds)
            if not rel:
                continue
            eng.cold_user_pop_prior = 0.0  # raw (no shrinkage)
            per_raw.append(([r.item_id for r in eng.recommend_for_items(seeds, n=K)], rel))
            eng.cold_user_pop_prior = pop_prior_c  # empirical-Bayes shrinkage
            per_shrunk.append(([r.item_id for r in eng.recommend_for_items(seeds, n=K)], rel))
            seed_set = set(seeds)
            per_pop.append(([it for it in pop_order if it not in seed_set][:K], rel))
        mr = aggregate(per_raw, catalog_size=catalog, k=K)
        ms = aggregate(per_shrunk, catalog_size=catalog, k=K)
        mp = aggregate(per_pop, catalog_size=catalog, k=K)
        rows.append({"seeds": k, "n": len(per_raw),
                     "kindling_raw_ndcg": round(mr.ndcg_at_k, 4),
                     "kindling_ndcg": round(ms.ndcg_at_k, 4),
                     "kindling_recall": round(ms.recall_at_k, 4),
                     "popularity_ndcg": round(mp.ndcg_at_k, 4)})
        log(f"  seeds={k:<3} raw={rows[-1]['kindling_raw_ndcg']:.4f} "
            f"shrunk={rows[-1]['kindling_ndcg']:.4f} pop={rows[-1]['popularity_ndcg']:.4f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"onboarding_{dataset}.json").write_text(
        json.dumps({"dataset": dataset, "k": K, "rows": rows}, indent=2) + "\n")
    log(f"[wrote] onboarding_{dataset}.json")


if __name__ == "__main__":
    main()
