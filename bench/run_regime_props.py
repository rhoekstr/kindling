"""Can we auto-detect the best cooc transform/aggregation from data properties?

Hypothesis from the 4-dataset smoothing sweep:
  - TRANSFORM is predicted by POPULARITY DRIFT. Strong drift (chronological,
    new-release-driven) => popularity IS signal => raw/light-cosine win,
    normalizing HURTS. Weak/no drift (stationary) => popularity is a static
    confound => normalize hard (wilson/jaccard/cosine) wins big.
  - AGGREGATION is predicted by GRAPH DENSITY. Dense => sum leaks popularity
    => candidate-L2 normalization wins. Sparse => sum wins.

This computes the candidate predictors cheaply (no cooc build) so we can check
whether they actually separate the observed winners.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from kindling.benchmarks.comparison import _load_academic_split, _load_dataset

# density measured in the smoothing sweep ([graph] lines) — carried here.
KNOWN = {
    "movielens-1m": dict(density=0.79, winner="ppmi_cds/cand_l2 (raw/cos≈ on sum)", agg="cand_l2"),
    "amazon-beauty": dict(density=0.013, winner="llr/sum ≈ raw ≈ EASE", agg="sum"),
    "steam": dict(density=0.14, winner="raw/sum (pop) > llr", agg="sum"),
    "amazon-book-academic": dict(density=0.039, winner="wilson/jaccard/cosine sum", agg="sum"),
}


def load(name):
    if name == "amazon-book-academic":
        book = Path("~/.cache/kindling/amazon-book").expanduser()
        return _load_academic_split(book / "train.txt", book / "test.txt",
                                    name=name, action_type="purchase")
    return _load_dataset(name, test_fraction=0.1)


def props(name):
    s = load(name)
    train, test = s.train, s.test
    n_items = train["item_id"].nunique()
    n_users = train["entity_id"].nunique()
    d_train = train["item_id"].value_counts()

    # popularity drift within train: first vs second chronological half.
    drift = float("nan")
    if "timestamp" in train.columns:
        t = train.sort_values("timestamp", kind="mergesort")
        half = len(t) // 2
        c1 = t.iloc[:half]["item_id"].value_counts()
        c2 = t.iloc[half:]["item_id"].value_counts()
        common = c1.index.intersection(c2.index)
        if len(common) > 10:
            drift = float(spearmanr(c1[common], c2[common]).statistic)

    # train->test popularity correlation (1 = stationary, low = drift/turnover)
    d_test = test["item_id"].value_counts()
    common = d_train.index.intersection(d_test.index)
    tt = float(spearmanr(d_train[common], d_test[common]).statistic) if len(common) > 10 else float("nan")

    # head-share: fraction of test interactions on the top-10% train-popular items.
    head = set(d_train.index[: max(1, int(0.10 * len(d_train)))])
    head_share = float(test["item_id"].isin(head).mean())

    # Gini of train popularity
    x = np.sort(d_train.to_numpy().astype(float))
    gini = float((2 * np.arange(1, len(x) + 1) - len(x) - 1).dot(x) / (len(x) * x.sum()))

    return dict(n_items=n_items, n_users=n_users, median_pop=int(d_train.median()),
                pop_gini=round(gini, 3), within_train_drift=round(drift, 3) if drift == drift else None,
                train_test_pop_corr=round(tt, 3), head10_share=round(head_share, 3))


def main():
    datasets = os.environ.get("FPR_DATASETS",
                              "movielens-1m,amazon-beauty,steam,amazon-book-academic").split(",")
    print(f"{'dataset':22s} {'n_items':>8s} {'dens':>6s} {'pop_gini':>8s} "
          f"{'drift':>6s} {'tt_corr':>7s} {'head10':>6s}  winner")
    for ds in datasets:
        try:
            p = props(ds)
            k = KNOWN.get(ds, {})
            print(f"{ds:22s} {p['n_items']:>8d} {k.get('density','?'):>6} {p['pop_gini']:>8.3f} "
                  f"{str(p['within_train_drift']):>6} {p['train_test_pop_corr']:>7.3f} "
                  f"{p['head10_share']:>6.3f}  agg={k.get('agg','?')} | {k.get('winner','?')}")
        except Exception as e:  # noqa: BLE001
            print(f"{ds:22s} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
