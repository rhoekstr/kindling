"""Sweep the repeat personal-frequency layer (repeat_freq_alpha) vs the
personal-frequency baseline, repeat-aware eval. Finds the alpha that makes
kindling's reorders frequency-driven enough to match/beat "buy it again".

Run: PYTHONPATH=src .venv/bin/python bench/repeat_freq_sweep.py <dataset...>
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
ALPHAS = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def _ndcg(per, k=10):
    nd = []
    for recs, rel in per:
        dcg = sum(1 / log2(i + 2) for i, p in enumerate(recs[:k]) if p in rel)
        idcg = sum(1 / log2(i + 2) for i in range(min(len(rel), k)))
        nd.append(dcg / idcg if idcg else 0.0)
    return round(float(np.mean(nd)), 4)


def run(ds: str, max_users: int = 1500):
    sp = load_split(ds, 0.1)
    test_by = {e: set(g) for e, g in sp.test.groupby("entity_id")["item_id"]}
    train_by = {e: list(g) for e, g in sp.train.groupby("entity_id")["item_id"]}
    ents = [e for e in test_by if e in train_by]
    rng = np.random.default_rng(0)
    if len(ents) > max_users:
        ents = [ents[i] for i in rng.choice(len(ents), max_users, replace=False)]
    pf = {e: list(pd.Series(train_by[e]).value_counts().index[:10]) for e in ents}
    pf_ndcg = _ndcg([(pf[e], test_by[e]) for e in ents])
    out = {"personal_freq": pf_ndcg}
    for a in ALPHAS:
        eng = Engine(random_state=0, repeat_recommend=True, repeat_freq_alpha=a).fit(sp.train)
        out[a] = _ndcg([([r.item_id for r in eng.recommend(e, 10)], test_by[e]) for e in ents])
    return out


def main(argv):
    datasets = argv[1:] or DATASETS
    head = "  ".join(f"a={a:g}" for a in ALPHAS)
    print(f"{'dataset':12s} {'pers_freq':>9}   {head}")
    for ds in datasets:
        o = run(ds)
        cells = "  ".join(f"{o[a]:.4f}" for a in ALPHAS)
        best = max(ALPHAS, key=lambda a: o[a])
        print(f"{ds:12s} {o['personal_freq']:>9}   {cells}   best_a={best:g} "
              f"({'>=pf' if o[best] >= o['personal_freq'] else '<pf'})")


if __name__ == "__main__":
    main(sys.argv)
