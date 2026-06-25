"""Freeze the channel-progression numbers (addendum U1/U2) to JSON.

The REFERENCE §2/§3.4 progression (raw cooc → EASE → +trend → +last-item →
+rating-weight → +user_cf) and the per-channel deltas existed only as prose /
runner stdout. This re-captures them as a retained artifact so they survive
the deletion of the experiment runners. Cumulative arms; auto-gates left on,
so each dataset shows what it actually activates (ml1m no-ops transitions /
user_cf by design).

Run: DATASET=movielens-1m .venv/bin/python bench/capture_channel_ablation.py
Out: bench/reports/channel_ablation_<dataset>.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set
from kindling.engine import Engine

_BASE = dict(retrieval_budget=500, random_state=0)

# (label, config-delta applied cumulatively over the previous arm)
ARMS = [
    (
        "raw_cooc_base",
        dict(
            base_scorer="cooc",
            trend_alpha=0.0,
            transition_alpha=0.0,
            last_item_alpha=0.0,
            user_cf_alpha=0.0,
            ease_use_weights="off",
        ),
    ),
    ("ease_base", dict(base_scorer="ease")),
    ("+trend", dict(trend_alpha=0.5)),
    ("+last_item", dict(last_item_alpha=0.25)),
    ("+transitions", dict(transition_alpha=0.25)),
    ("+rating_weight", dict(ease_use_weights="auto")),
    ("+user_cf", dict(user_cf_alpha=1.0)),
]


def main() -> None:
    dataset = os.environ.get("DATASET", "movielens-1m")
    lam = {"amazon-beauty": 250.0}.get(dataset)
    split = _load_dataset(dataset, test_fraction=0.1)
    train = split.train
    eval_set = _build_eval_set(train, split.test, max_users=500, seed=0)
    has_meta = getattr(split, "items", None) is not None

    cfg: dict = dict(_BASE)
    if lam is not None:
        cfg["ease_lambda"] = lam
    rows = []
    for label, delta in ARMS:
        cfg = {**cfg, **delta}
        t0 = time.perf_counter()
        eng = Engine(**cfg)
        eng.fit(train, item_metadata=split.items if has_meta else None)
        st = eng._state
        per = [
            ([r.item_id for r in eng.recommend(entity_id=e, n=10)], rel)
            for e, rel in eval_set.items()
        ]
        rep = aggregate(per, catalog_size=max(st.n_items, 1), k=10)
        p = st.profile
        row = {
            "arm": label,
            "ndcg@10": round(float(rep.ndcg_at_k), 4),
            "recall@10": round(float(rep.recall_at_k), 4),
            "mrr": round(float(rep.mrr), 4),
            "base_used": p.get("base_scorer_used"),
            "transition_active": p.get("transition_channel_active"),
            "fit_s": round(time.perf_counter() - t0, 1),
        }
        rows.append(row)
        print(
            f"[{dataset}] {label:16s} ndcg@10={row['ndcg@10']:.4f} "
            f"base={row['base_used']} trans={row['transition_active']}",
            flush=True,
        )

    out_path = Path("bench/reports") / f"channel_ablation_{dataset}.json"
    out_path.write_text(json.dumps({"dataset": dataset, "arms": rows}, indent=2))
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
