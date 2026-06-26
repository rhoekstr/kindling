"""Native-engine end-to-end recommend parity (Rust `EngineState`).

The Rust `kindling_core.build_engine(arrays, config)` owns the fit arrays and
its `.recommend(owned, user_row, n)` reproduces `engine.py::_recommend_core`
(EASE base + channel blend + temporal-cooc boost layer + cold-slots). This
harness builds the native engine from a Python-fitted `EngineState` and checks,
over the canonical 500-user eval set, that:

  * the rec lists match Python's `engine.recommend` (byte-exact modulo
    tie-equivalent neighbors / final-argsort ties — NDCG-neutral), and
  * the native NDCG@10 reproduces the frozen `gates.toml` baseline.

Run:  python bench/native_recommend_parity.py            # ml1m (fast)
      python bench/native_recommend_parity.py steam      # + cold-slots
      python bench/native_recommend_parity.py amazon-beauty steam
"""

from __future__ import annotations

import sys

import numpy as np

from kindling import Engine
from kindling._native import kindling_core
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


def _uri_csr(st, n_users: int):
    """user_row_items dict → CSR (data, indptr) for native neighbor voting."""
    indptr = np.zeros(n_users + 1, np.int64)
    parts = []
    for u in range(n_users):
        items = st.user_row_items.get(u)
        if items is not None and len(items) > 0:
            parts.append(np.asarray(items, np.int64))
            indptr[u + 1] = indptr[u] + len(items)
        else:
            indptr[u + 1] = indptr[u]
    data = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    return data, indptr


def build_native_engine(eng: Engine):
    """Construct the Rust `EngineState` from a Python-fitted engine."""
    st = eng._state
    n_users = st.uu_user_deg.shape[0] if st.uu_user_deg is not None else 0
    arrays: dict = {"ease_b": np.ascontiguousarray(st.ease_b, np.float32)}
    if st.trend_z is not None:
        arrays["trend_z"] = st.trend_z.astype(np.float64)
    if st.trans_data is not None:
        arrays["trans_data"] = st.trans_data.astype(np.float64)
        arrays["trans_indices"] = st.trans_indices.astype(np.int32)
        arrays["trans_indptr"] = st.trans_indptr.astype(np.int64)
    if st.uu_users_data is not None:
        arrays["uu_data"] = st.uu_users_data.astype(np.int64)
        arrays["uu_indptr"] = st.uu_users_indptr.astype(np.int64)
        arrays["uu_deg"] = st.uu_user_deg.astype(np.float64)
        ud, uip = _uri_csr(st, n_users)
        arrays["uri_data"], arrays["uri_indptr"] = ud, uip
    if st.item_popularity is not None:
        arrays["item_pop"] = st.item_popularity.astype(np.float64)
    boost = []
    for name in st.enabled_boost_layers:
        adj = st.boost_layer_adjacencies.get(name)
        if adj is not None:
            d, i, p = adj
            boost.append((
                np.ascontiguousarray(d, np.float32),
                np.ascontiguousarray(i, np.int32),
                np.ascontiguousarray(p, np.int32),
            ))
    arrays["boost"] = boost
    cf = st.content_features
    content_nfeat = 0
    if cf is not None and cf.n_features > 0 and eng.cold_slots > 0:
        content_nfeat = int(cf.n_features)
        arrays["content_data"] = np.ascontiguousarray(cf.data, np.float32)
        arrays["content_indices"] = np.ascontiguousarray(cf.indices, np.int32)
        arrays["content_indptr"] = np.ascontiguousarray(cf.indptr, np.int32)
        if st.content_coldness is not None:
            arrays["content_coldness"] = st.content_coldness.astype(np.float64)
        if st.cold_recency is not None:
            arrays["cold_recency"] = st.cold_recency.astype(np.float64)
    config = dict(
        n_items=int(st.n_items),
        trend_alpha=float(st.trend_alpha),
        last_item_alpha=float(st.last_item_alpha),
        transition_alpha=float(st.transition_alpha),
        transition_last_k=int(st.transition_last_k),
        transition_decay=float(st.transition_decay),
        user_cf_alpha=float(st.user_cf_alpha),
        user_cf_k=int(st.user_cf_k),
        n_users=int(n_users),
        z_threshold=float(st.z_threshold),
        boost_multiplier=float(st.boost_multiplier),
        retrieval_budget=int(eng.retrieval_budget),
        cold_slots=int(eng.cold_slots),
        content_nfeat=content_nfeat,
        cold_recency_beta=float(st.cold_recency_beta or 0.0),
    )
    return kindling_core.build_engine(arrays, config)


def check(dataset: str) -> tuple[int, int, float, float]:
    """Returns (identical, n_users, ndcg_python, ndcg_native)."""
    cfg = _CONFIG.get(dataset, {})
    split = _load_dataset(dataset, 0.1)
    has_meta = getattr(split, "items", None) is not None
    eng = Engine(retrieval_budget=500, random_state=0, **cfg).fit(
        split.train, item_metadata=split.items if has_meta else None
    )
    st = eng._state
    ne = build_native_engine(eng)
    eval_set = _build_eval_set(split.train, split.test, max_users=500, seed=0)
    per_py, per_rs, identical, nuser = [], [], 0, 0
    for ent, rel in eval_set.items():
        owned = st.owned_by_entity.get(ent)
        if owned is None or owned.size == 0:
            continue
        py = [r.item_id for r in eng.recommend(ent, 10)]
        ur = int(st.entity_to_user_idx.get(ent, -1))
        items, _sc, _kind = ne.recommend(owned.astype(np.int64).tolist(), ur, 10, 0.0)
        rs = [st.item_ids[i] for i in items]
        identical += int(py == rs)
        nuser += 1
        per_py.append((py, rel))
        per_rs.append((rs, rel))
    cat = max(st.n_items, 1)
    nd_py = float(aggregate(per_py, catalog_size=cat, k=10).ndcg_at_k)
    nd_rs = float(aggregate(per_rs, catalog_size=cat, k=10).ndcg_at_k)
    return identical, nuser, nd_py, nd_rs


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["movielens-1m"]
    ok = True
    for ds in datasets:
        ident, nuser, nd_py, nd_rs = check(ds)
        match = abs(nd_py - nd_rs) < 5e-4
        ok = ok and match
        print(
            f"[{'PASS' if match else 'FAIL'}] {ds:18s} reclist {ident}/{nuser}  "
            f"NDCG@10 python={nd_py:.4f} native={nd_rs:.4f}"
        )
    raise SystemExit(0 if ok else 1)
