"""Fair repeat-aware comparison (Stage 5).

The earlier repeat-module numbers compared kindling (allowed to recommend
reorders) against baselines that still masked seen items — not a fair head-to-head.
This gives the baselines a repeat path too, and scores everyone on the same
repeat-aware eval (the full next basket, reorders included):

  kindling         repeat module on (timing-aware reorders + new items)
  personal_freq    the "buy it again" gold standard: the user's own most-bought
                   items, by train count (unmasked)
  global_pop       top globally-popular items, unmasked

So the question is fair: does kindling's timing-aware repeat beat naive
buy-it-again? Run on the repeat-regime datasets.

Run: PYTHONPATH=src .venv/bin/python bench/repeat_aware_compare.py <dataset...>
"""

from __future__ import annotations

import sys
from math import log2

import numpy as np
import pandas as pd

sys.path.insert(0, "bench")
from run_warming_curve import load_split

from kindling import Engine

DATASETS = ["dunnhumby", "tafeng", "instacart", "gowalla"]


def _ndcg(per, k=10):
    nd = []
    for recs, rel in per:
        dcg = sum(1 / log2(i + 2) for i, p in enumerate(recs[:k]) if p in rel)
        idcg = sum(1 / log2(i + 2) for i in range(min(len(rel), k)))
        nd.append(dcg / idcg if idcg else 0.0)
    return round(float(np.mean(nd)), 4)


def _eval_users(train, test, max_users=2000, seed=0):
    test_by = {e: set(g) for e, g in test.groupby("entity_id")["item_id"]}
    train_by = {e: list(g) for e, g in train.groupby("entity_id")["item_id"]}
    ents = [e for e in test_by if e in train_by]
    rng = np.random.default_rng(seed)
    if len(ents) > max_users:
        ents = [ents[i] for i in rng.choice(len(ents), max_users, replace=False)]
    return test_by, train_by, ents


def run(ds: str):
    sp = load_split(ds, 0.1)
    test_by, train_by, ents = _eval_users(sp.train, sp.test)
    # personal-frequency: user's items by descending train count
    pf = {}
    for e in ents:
        vc = pd.Series(train_by[e]).value_counts()
        pf[e] = list(vc.index[:10])
    # global popularity (unmasked)
    gp = list(pd.Series(sp.train["item_id"]).value_counts().index[:10])
    # kindling with repeat module
    eng = Engine(random_state=0, repeat_recommend=True).fit(sp.train)
    out = {}
    out["personal_freq"] = _ndcg([(pf[e], test_by[e]) for e in ents])
    out["global_pop"] = _ndcg([(gp, test_by[e]) for e in ents])
    out["kindling_repeat"] = _ndcg(
        [([r.item_id for r in eng.recommend(e, 10)], test_by[e]) for e in ents]
    )
    return out


def main(argv):
    datasets = argv[1:] or DATASETS
    print(f"{'dataset':12s} {'kindling_repeat':>16} {'personal_freq':>14} {'global_pop':>11}")
    for ds in datasets:
        o = run(ds)
        print(f"{ds:12s} {o['kindling_repeat']:>16} {o['personal_freq']:>14} {o['global_pop']:>11}")


if __name__ == "__main__":
    main(sys.argv)
