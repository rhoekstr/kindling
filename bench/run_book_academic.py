"""kindling vs published baselines on the LightGCN academic amazon-book split.

The single most-cited RecSys benchmark: LightGCN-PyTorch's amazon-book
train.txt/test.txt (52,643 users / 91,599 items / 2.38M train). Metric:
full-catalog ranking, Recall@20 + NDCG@20 over all test users (we sample
for speed). Published numbers on this exact split:

    NGCF       Recall@20 0.0344  NDCG@20 0.0263
    Mult-VAE   Recall@20 0.0407  NDCG@20 0.0315
    LightGCN   Recall@20 0.0411  NDCG@20 0.0315

This is timestamp-less, non-chronological — so kindling runs cooc base
(91k items > EASE gate) with NO trend/transition channels (they gate
off). It is a pure item-item-CF comparison: how does the kindling cooc
base stack up against graph-neural and VAE models on their home turf?
"""
import time

import numpy as np

from kindling.benchmarks.comparison import _load_academic_split
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set
from kindling.engine_v2 import EngineV2
from pathlib import Path


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


K = 20
book = Path("~/.cache/kindling/amazon-book").expanduser()
split = _load_academic_split(
    book / "train.txt", book / "test.txt", name="amazon-book-academic",
    action_type="rate",
)
train, test = split.train, split.test
log(f"academic split: train {len(train):,}  test {len(test):,}  "
    f"users {train.entity_id.nunique():,}  items {train.item_id.nunique():,}")
eval_set = _build_eval_set(train, test, max_users=5000, seed=0)
log(f"eval users: {len(eval_set)}")

t0 = time.perf_counter()
e = EngineV2(persona_min_users=10**9, retrieval_budget=max(500, K * 25), random_state=0)
e.fit(train)
log(f"fit {time.perf_counter()-t0:.0f}s  base={e._state.profile.get('base_scorer_used')}")

per = []
for n, (entity, rel) in enumerate(eval_set.items()):
    recs = e.recommend(entity_id=entity, n=K)
    per.append(([r.item_id for r in recs], rel))
    if (n + 1) % 1000 == 0:
        log(f"eval {n + 1}/{len(eval_set)}")

rep = aggregate(per, catalog_size=max(e._state.n_items, 1), k=K)
log(f"KINDLING  Recall@{K}={rep.recall_at_k:.4f}  NDCG@{K}={rep.ndcg_at_k:.4f}  "
    f"MRR={rep.mrr:.4f}  HR={rep.hit_rate:.3f}")
log("PUBLISHED NGCF R@20 0.0344 N@20 0.0263 | Mult-VAE 0.0407/0.0315 | "
    "LightGCN 0.0411/0.0315")
