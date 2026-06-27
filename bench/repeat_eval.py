"""Repeat-module eval: does re-surfacing reorders help on repeat-regime data,
and does it leave non-repeat data untouched?

For repeat datasets we score two ways:
  - exclude-seen (the standard full-ranking eval; hides reorders)
  - include-seen  (repeat-aware: the real next-basket, reorders count)
each with repeat_recommend ON vs OFF. For non-repeat datasets repeat is auto-off,
so the standard eval must be unchanged (regression gate).

Run: PYTHONPATH=src .venv/bin/python bench/repeat_eval.py <dataset...>
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "bench")
from run_warming_curve import load_split

from kindling import Engine
from kindling.benchmarks.metrics import aggregate

REPEAT_DS = {"dunnhumby", "instacart", "tafeng", "gowalla", "retailrocket"}


def _eval_set(st, test_df, max_users=1000, seed=0):
    test_by = {e: set(g) for e, g in test_df.groupby("entity_id")["item_id"]}
    ents = [e for e in test_by if e in st.owned_by_entity]
    rng = np.random.default_rng(seed)
    if len(ents) > max_users:
        ents = [ents[i] for i in rng.choice(len(ents), max_users, replace=False)]
    return test_by, ents


def run(ds: str, max_users=1000):
    sp = load_split(ds, 0.1)
    repeat_aware = ds in REPEAT_DS
    # Repeat datasets need no content metadata (and some carry malformed release
    # dates); the regression datasets keep theirs to match reference numbers.
    items = None if repeat_aware else getattr(sp, "items", None)
    out = {}
    for rep in ([False, True] if repeat_aware else [False]):
        eng = Engine(random_state=0, repeat_recommend=rep).fit(sp.train, item_metadata=items)
        st = eng._state
        test_by, ents = _eval_set(st, sp.test, max_users)
        cat = max(st.n_items, 1)
        per_excl, per_incl = [], []
        for ent in ents:
            owned_ids = {st.item_ids[i] for i in st.owned_by_entity[ent].tolist()}
            recs = [r.item_id for r in eng.recommend(ent, 10)]
            target = test_by[ent]
            per_incl.append((recs, target))
            per_excl.append((recs, target - owned_ids))
        nd_excl = float(aggregate(per_excl, catalog_size=cat, k=10).ndcg_at_k)
        nd_incl = float(aggregate(per_incl, catalog_size=cat, k=10).ndcg_at_k)
        out[rep] = (nd_excl, nd_incl, st.repeat_active, st.repeat_rate)
    return out, repeat_aware


def main(argv):
    datasets = argv[1:] or ["dunnhumby", "tafeng"]
    print(f"{'dataset':14s} {'repeat':>7} {'active':>6} | {'excl-seen':>9} {'incl-seen(repeat-aware)':>23}")
    for ds in datasets:
        out, aware = run(ds)
        for rep in sorted(out):
            ex, inc, act, rate = out[rep]
            tag = "repeat-aware" if aware else "standard"
            print(f"{ds:14s} {rep!s:>7} {act!s:>6} | {ex:>9.4f} {inc:>23.4f}   ({tag}, rate={rate:.2f})")
        print()


if __name__ == "__main__":
    main(sys.argv)
