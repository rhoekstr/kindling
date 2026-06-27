"""Gowalla super-consumer trimming probe (Stage 6).

Gowalla's NDCG peaks before full data and dips at 100% — and unlike dunnhumby/
instacart, popularity is flat there, so the dip looks model-side, not an eval
artifact. Gowalla also has the highest interaction concentration (gini 0.70,
top-1% share 18%). Hypothesis: super-consumers (users with huge check-in counts)
inject spurious co-occurrence edges (quadratic in their basket size) that pollute
the cooc base at full data.

Clean isolation: cap each user's training interactions at K (trim the heavy
users), refit, measure. If a cap recovers/beats the full-data NDCG, super-consumer
pollution is real and per-user capping is the fix. Eval is fixed (exclude-seen,
repeat off) so we isolate the base, not the repeat path.

Run: PYTHONPATH=src .venv/bin/python bench/gowalla_trim.py
"""

from __future__ import annotations

import sys
from math import log2

import numpy as np
import pandas as pd

sys.path.insert(0, "bench")
from run_warming_curve import load_split

from kindling import Engine


def _trim(train: pd.DataFrame, cap: int | None, seed: int = 0) -> pd.DataFrame:
    if cap is None:
        return train
    rng = np.random.default_rng(seed)
    keep = []
    for _, idx in train.groupby("entity_id").indices.items():
        sub = rng.choice(idx, cap, replace=False) if len(idx) > cap else idx
        keep.append(sub)
    sel = np.concatenate(keep)
    return train.iloc[sel].reset_index(drop=True)


def main():
    sp = load_split("gowalla", 0.1)
    train, test = sp.train, sp.test
    counts = train.groupby("entity_id").size().to_numpy()
    print(f"gowalla: users={len(counts)} items={train.item_id.nunique()} "
          f"median/user={int(np.median(counts))} p99={int(np.quantile(counts, 0.99))} "
          f"max={int(counts.max())}")
    test_by = {e: set(g) for e, g in test.groupby("entity_id")["item_id"]}
    full_owned = {e: set(g) for e, g in train.groupby("entity_id")["item_id"]}
    ents = [e for e in test_by if e in full_owned]
    rng = np.random.default_rng(0)
    if len(ents) > 1500:
        ents = [ents[i] for i in rng.choice(len(ents), 1500, replace=False)]

    print(f"\n{'cap':>6} {'train_rows':>11} {'NDCG@10':>8}")
    for cap in (None, 200, 100, 50, 25):
        tr = _trim(train, cap)
        eng = Engine(random_state=0, repeat_recommend=False).fit(tr)
        nd = []
        for e in ents:
            recs = [r.item_id for r in eng.recommend(e, 30)]
            rel = test_by[e] - full_owned[e]
            recs = [r for r in recs if r not in full_owned[e]][:10]
            dcg = sum(1 / log2(i + 2) for i, p in enumerate(recs) if p in rel)
            idcg = sum(1 / log2(i + 2) for i in range(min(len(rel), 10)))
            nd.append(dcg / idcg if idcg else 0.0)
        print(f"{cap!s:>6} {len(tr):>11} {round(float(np.mean(nd)), 4):>8}", flush=True)


if __name__ == "__main__":
    main()
