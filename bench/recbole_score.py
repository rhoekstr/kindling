"""Unified scorer for the RecBole-vs-kindling comparison.

Fits kindling on the *same* train split RecBole used (recbole_data/split_train.csv),
predicts top-10 for the same test users, then grades EVERY model — the RecBole
baselines (from recbole_<model>.json) and kindling — with one NDCG@10 /
Recall@10 / MRR function against the shared ground truth (split_truth.json).
Cross-checks the scorer against RecBole's own reported metric so the numbers are
on one calibrated scale. Honest time columns: fit and predict wall-clock.

Run (main venv, kindling): python bench/recbole_score.py
Out: bench/reports/recbole_comparison.json
"""

from __future__ import annotations

import json
import time
from math import log2
from pathlib import Path

import numpy as np
import pandas as pd

from kindling import Engine

OUT = Path("recbole_data")
REPORTS = Path("bench/reports")
K = 10


def score(preds: dict[str, list[str]], truth: dict[str, set[str]]) -> tuple[float, float, float]:
    ndcgs, recalls, mrrs = [], [], []
    for u, T in truth.items():
        if not T:
            continue
        P = preds.get(u, [])[:K]
        dcg = sum(1.0 / log2(i + 2) for i, p in enumerate(P) if p in T)
        idcg = sum(1.0 / log2(i + 2) for i in range(min(len(T), K)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        recalls.append(len(set(P) & T) / len(T))
        rr = next((1.0 / (i + 1) for i, p in enumerate(P) if p in T), 0.0)
        mrrs.append(rr)
    return float(np.mean(ndcgs)), float(np.mean(recalls)), float(np.mean(mrrs))


def main() -> int:
    truth = {u: set(v) for u, v in json.loads((OUT / "split_truth.json").read_text()).items()}
    rows = []

    # ── RecBole baselines (their predictions + their reported metric + time).
    for path in sorted(OUT.glob("recbole_*.json")):
        d = json.loads(path.read_text())
        ndcg, recall, mrr = score({u: v for u, v in d["topk"].items()}, truth)
        rows.append({
            "model": d["model"],
            "framework": "RecBole",
            "ndcg@10": round(ndcg, 4),
            "recall@10": round(recall, 4),
            "mrr": round(mrr, 4),
            "ndcg@10_recbole": round(d["metrics"].get("ndcg@10", float("nan")), 4),
            "fit_seconds": d["fit_seconds"],
            "predict_seconds": d["predict_seconds"],
        })

    # ── kindling on the identical train split.
    # IMPORTANT: RecBole masks BOTH train and validation items from the test
    # ranking. kindling is fit on (and only knows) the 80% train, so we must
    # mask the user's validation items too — else kindling is charged misses
    # for recommending items it was never told the user had seen. valid =
    # full − train − test, reconstructed from the full atomic file.
    train = pd.read_csv(OUT / "split_train.csv")
    full = pd.read_csv(OUT / "ml-1m" / "ml-1m.inter", sep="\t")
    full.columns = ["user", "item", "rating", "ts"]
    full_by = {str(u): set(map(str, g)) for u, g in full.groupby("user")["item"]}
    train_by = {str(u): set(map(str, g)) for u, g in train.groupby("entity_id")["item_id"]}
    valid_by = {u: (full_by.get(u, set()) - train_by.get(u, set()) - truth.get(u, set())) for u in truth}

    t0 = time.perf_counter()
    eng = Engine(random_state=0).fit(train)
    fit_s = time.perf_counter() - t0
    st = eng._state
    test_users = list(truth.keys())
    ent_of = {u: int(u) for u in test_users}
    known = [u for u in test_users if ent_of[u] in st.owned_by_entity]
    t0 = time.perf_counter()
    batch = eng.recommend_batch([ent_of[u] for u in known], n=K + 40)  # over-fetch to mask valid
    predict_s = time.perf_counter() - t0
    kpreds = {
        u: [str(r.item_id) for r in recs if str(r.item_id) not in valid_by.get(u, set())][:K]
        for u, recs in zip(known, batch)
    }
    ndcg, recall, mrr = score(kpreds, truth)
    rows.append({
        "model": "kindling",
        "framework": "kindling",
        "ndcg@10": round(ndcg, 4),
        "recall@10": round(recall, 4),
        "mrr": round(mrr, 4),
        "ndcg@10_recbole": None,
        "fit_seconds": round(fit_s, 3),
        "predict_seconds": round(predict_s, 3),
        "base": st.base_scorer_used,
    })

    rows.sort(key=lambda r: -r["ndcg@10"])
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "recbole_comparison.json").write_text(json.dumps({"dataset": "ml-1m", "k": K, "rows": rows}, indent=2))

    hdr = f"{'model':10s} {'fw':9s} {'NDCG@10':>8s} {'Recall@10':>9s} {'MRR':>7s} {'fit(s)':>8s} {'pred(s)':>8s} {'xcheck':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        xc = "" if r["ndcg@10_recbole"] is None else f"{r['ndcg@10_recbole']:.4f}"
        print(
            f"{r['model']:10s} {r['framework']:9s} {r['ndcg@10']:8.4f} {r['recall@10']:9.4f} "
            f"{r['mrr']:7.4f} {r['fit_seconds']:8.1f} {r['predict_seconds']:8.2f} {xc:>7s}"
        )
    print("\n(xcheck = RecBole's own NDCG@10 for that model — should match this scorer's column)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
