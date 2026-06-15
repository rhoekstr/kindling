"""Standalone LightGCN on the LightGCN academic amazon-book split.

Settles the question: can OUR hand-rolled LightGCN reproduce the
published 0.0411/0.0315 when run the way the published number is
produced — standalone, full-catalog ranking, k=20, properly trained?

Same split, same eval set (seed=0, 5000 users), same k as
bench/run_book_academic.py, so LightGCN, our cooc base (0.0369/0.0285),
and the published rows are all directly comparable. Sweeps epoch counts
to expose the convergence trajectory: climbing toward 0.041 (impl fine,
just needs training) vs stalled low (impl bug or protocol mismatch).
"""
import os
import sys
import time

import numpy as np

import kindling_core
from kindling.benchmarks.comparison import _load_academic_split
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set
from pathlib import Path


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


K = 20
EPOCHS = [int(x) for x in (sys.argv[1:] or [50, 200, 500])]
# Fit config overridable via env (defaults = published LightGCN config).
DIM = int(os.environ.get("LGCN_DIM", 64))
LAYERS = int(os.environ.get("LGCN_LAYERS", 3))
BATCH = int(os.environ.get("LGCN_BATCH", 8192))
LR = float(os.environ.get("LGCN_LR", 0.005))
book = Path("~/.cache/kindling/amazon-book").expanduser()
split = _load_academic_split(
    book / "train.txt", book / "test.txt", name="amazon-book-academic",
    action_type="rate",
)
train, test = split.train, split.test
entity_ids = list(dict.fromkeys(train["entity_id"].tolist()))  # first-appearance
e2u = {e: i for i, e in enumerate(entity_ids)}
item_ids = list(dict.fromkeys(train["item_id"].tolist()))
i2c = {it: j for j, it in enumerate(item_ids)}
item_ids_arr = np.array(item_ids, dtype=object)
n_users, n_items = len(entity_ids), len(item_ids)
uidx = train["entity_id"].map(e2u).to_numpy(np.int64)
iidx = train["item_id"].map(i2c).to_numpy(np.int64)
w = np.ones(len(train), dtype=np.float32)
log(f"academic split: {n_users:,} users  {n_items:,} items  {len(train):,} train")

# Per-user train items (to mask at scoring) + eval set identical to cooc run.
owned = {}
for u, i in zip(uidx, iidx):
    owned.setdefault(int(u), []).append(int(i))
owned = {u: np.array(v, dtype=np.int64) for u, v in owned.items()}
eval_set = _build_eval_set(train, test, max_users=5000, seed=0)
log(f"eval users: {len(eval_set)}")

for ep in EPOCHS:
    t0 = time.perf_counter()
    U, I, trained = kindling_core.fit_lightgcn_py(
        uidx, iidx, w, n_users=n_users, n_items=n_items,
        dim=DIM, n_layers=LAYERS, learning_rate=LR, weight_decay=1e-4,
        n_epochs=ep, batch_size=BATCH, seed=0, min_users=1, min_items=1,
    )
    U = np.asarray(U); I = np.asarray(I)
    fit_s = time.perf_counter() - t0
    per = []
    for entity, rel in eval_set.items():
        u = e2u.get(entity)
        if u is None:
            per.append(([], rel)); continue
        s = I @ U[u]
        own = owned.get(u)
        if own is not None:
            s[own] = -np.inf
        top = np.argpartition(-s, K)[:K]
        top = top[np.argsort(-s[top])]
        per.append(([item_ids_arr[j] for j in top], rel))
    rep = aggregate(per, catalog_size=n_items, k=K)
    log(f"LightGCN d={DIM} L={LAYERS} ep={ep:<4} (fit {fit_s:.0f}s)  "
        f"Recall@{K}={rep.recall_at_k:.4f}  NDCG@{K}={rep.ndcg_at_k:.4f}  "
        f"HR={rep.hit_rate:.3f}")

log("REFERENCE  cooc-base 0.0369/0.0285 | NGCF 0.0344/0.0263 | "
    "Mult-VAE 0.0407/0.0315 | LightGCN(pub) 0.0411/0.0315")
