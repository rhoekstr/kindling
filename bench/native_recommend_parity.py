"""Native-engine end-to-end recommend parity (Rust `EngineState`).

`Engine.recommend_batch` serves known entities through the native Rust engine
(`kindling._native_engine.build_native_engine` → `kindling_core.EngineState`),
which reproduces `engine.py::_recommend_core` (EASE base + channel blend +
temporal-cooc boost layer + cold-slots). This harness checks, over the
canonical 500-user eval set, that the batch path matches the per-user Python
`recommend` (byte-exact modulo rare FP-tie orderings — NDCG-neutral) and
reproduces the frozen `gates.toml` NDCG@10 baseline.

Run:  python bench/native_recommend_parity.py            # ml1m (fast)
      python bench/native_recommend_parity.py steam      # + cold-slots
      python bench/native_recommend_parity.py movielens-1m amazon-beauty steam
"""

from __future__ import annotations

import sys

from kindling import Engine
from kindling._native_engine import native_supported
from kindling.benchmarks.comparison import _load_dataset
from kindling.benchmarks.metrics import aggregate
from kindling.benchmarks.parity import _build_eval_set

# Documented per-dataset config (mirrors bench/verify.py).
_CONFIG = {
    "movielens-1m": {},
    "amazon-beauty": {"ease_lambda": 250.0},
    "steam": {"cold_slots": 1},
    "amazon-book-chrono": {"cold_slots": 1},
}


def check(dataset: str) -> tuple[int, int, float, float, bool]:
    """Returns (identical, n_users, ndcg_python, ndcg_native, native_used)."""
    cfg = _CONFIG.get(dataset, {})
    split = _load_dataset(dataset, 0.1)
    has_meta = getattr(split, "items", None) is not None
    eng = Engine(retrieval_budget=500, random_state=0, **cfg).fit(
        split.train, item_metadata=split.items if has_meta else None
    )
    st = eng._state
    used = native_supported(eng)
    eval_set = _build_eval_set(split.train, split.test, max_users=500, seed=0)
    ents = [e for e in eval_set if (ow := st.owned_by_entity.get(e)) is not None and ow.size > 0]

    py_lists = [[r.item_id for r in eng.recommend(e, 10)] for e in ents]
    rs_lists = [[r.item_id for r in recs] for recs in eng.recommend_batch(ents, 10)]
    identical = sum(int(a == b) for a, b in zip(py_lists, rs_lists))

    cat = max(st.n_items, 1)
    per_py = [(py_lists[i], eval_set[ents[i]]) for i in range(len(ents))]
    per_rs = [(rs_lists[i], eval_set[ents[i]]) for i in range(len(ents))]
    nd_py = float(aggregate(per_py, catalog_size=cat, k=10).ndcg_at_k)
    nd_rs = float(aggregate(per_rs, catalog_size=cat, k=10).ndcg_at_k)
    return identical, len(ents), nd_py, nd_rs, used


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["movielens-1m"]
    ok = True
    for ds in datasets:
        ident, nuser, nd_py, nd_rs, used = check(ds)
        match = abs(nd_py - nd_rs) < 5e-4
        ok = ok and match
        print(
            f"[{'PASS' if match else 'FAIL'}] {ds:18s} native={used!s:5s} "
            f"reclist {ident}/{nuser}  NDCG@10 python={nd_py:.4f} batch={nd_rs:.4f}"
        )
    raise SystemExit(0 if ok else 1)
