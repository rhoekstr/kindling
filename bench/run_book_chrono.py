"""Book-chrono benchmark runner with stage logging (memory-diagnosable)."""
import time

import numpy as np

from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set
from kindling.engine_v2 import EngineV2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


t0 = time.perf_counter()
split = _load_dataset("amazon-book-chrono", test_fraction=0.1)
train, test = split.train, split.test
log(f"loaded {time.perf_counter()-t0:.0f}s: train {len(train):,}  "
    f"users {train.entity_id.nunique():,}  items {train.item_id.nunique():,}")
eval_set = _build_eval_set(train, test, max_users=500, seed=0)
tr_counts = train.groupby("item_id").size()
log(f"eval set: {len(eval_set)} users")

t0 = time.perf_counter()
e = EngineV2(persona_min_users=10_000_000, retrieval_budget=500, random_state=0)
e.fit(train)
st = e._state
p = st.profile
log(f"fit {time.perf_counter()-t0:.0f}s  base={p.get('base_scorer_used')}  "
    f"trans={p.get('transition_channel_active')}  "
    f"boost_skip={p.get('boost_layers_skipped', 'no')}")

per = []
hb = {}
for n_done, (entity, rel) in enumerate(eval_set.items()):
    recs = e.recommend(entity_id=entity, n=10)
    top = [r.item_id for r in recs]
    per.append((top, rel))
    ts = set(top)
    for item in rel:
        c = tr_counts.get(item, 0)
        b = "0" if c == 0 else ("1-4" if c < 5 else ("5-19" if c < 20 else "20+"))
        d = hb.setdefault(b, [0, 0])
        d[1] += 1
        if item in ts:
            d[0] += 1
    if (n_done + 1) % 100 == 0:
        log(f"eval {n_done + 1}/{len(eval_set)}")

rep = aggregate(per, catalog_size=max(st.n_items, 1), k=10)
log(f"BOOK-CHRONO: NDCG={rep.ndcg_at_k:.4f}  MRR={rep.mrr:.4f}  "
    f"recall={rep.recall_at_k:.4f}  HR={rep.hit_rate:.3f}")
for b in ["0", "1-4", "5-19", "20+"]:
    h, t = hb.get(b, [0, 0])
    log(f"  item-warmth {b:>4}: {h}/{t} ({h/max(t,1):.1%})")
