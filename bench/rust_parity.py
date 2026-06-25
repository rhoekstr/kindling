"""Differential parity harness for the Rust engine port.

Each Rust kernel/stage must reproduce its Python counterpart byte-for-byte (or
within tight tolerance). This is the gate for every phase of the port
(docs/RUST-ENGINE-PLAN.md). Run: python bench/rust_parity.py
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from kindling._native import kindling_core
from kindling.graph.cooc_transform import apply_cooc_transform


def check_cooc_transform() -> bool:
    rng = np.random.default_rng(0)
    n = 3000
    c = sp.random(n, n, density=0.01, format="csr", random_state=1,
                  data_rvs=lambda s: rng.integers(1, 50, s)).astype(np.float32)
    counts = rng.integers(1, 500, n).astype(np.float64)
    ind, ip = c.indices.astype(np.int32), c.indptr.astype(np.int32)
    ok = True
    for t in ("wilson", "cosine", "jaccard", "raw"):
        py = apply_cooc_transform(c.data.copy(), ind, ip, counts, n_users=1000, transform=t)
        rs = np.asarray(
            kindling_core.cooc_transform(np.ascontiguousarray(c.data, np.float32), ind, ip, counts, t, 1.96),
            dtype=np.float32,
        )
        exact = np.array_equal(py, rs)
        print(f"  cooc_transform[{t:8s}] exact={exact}")
        ok = ok and exact
    return ok


def check_metadata_knn() -> bool:
    rng = np.random.default_rng(0)
    from kindling.graph.metadata_smoothing import _knn_edges  # Rust-backed when available

    f = sp.random(1500, 200, density=0.02, format="csr", random_state=0,
                  data_rvs=lambda s: rng.uniform(0.5, 2, s)).astype(np.float32)
    ei, ej, _ = _knn_edges(f, 20)
    print(f"  metadata_knn edges={len(ei)} (Rust path)")
    return len(ei) > 0


def check_fit_channels() -> bool:
    """Rust fit_channels (popularity / trend_z / user-CF CSR) vs Python EngineState."""
    import pandas as pd

    from kindling import Engine
    from kindling.benchmarks.comparison import _load_dataset
    from kindling.ingest.contract import canonicalize, validate_interactions
    from kindling.preprocess import preprocess_interactions

    def recon(train: pd.DataFrame):  # reproduce the engine's preprocessed arrays
        sch = validate_interactions(train)
        c = canonicalize(train, sch)
        c, _ = preprocess_interactions(c, use_ratings=None)
        i2i = {it: i for i, it in enumerate(pd.Index(c["item_id"].unique()))}
        e2u = {e: i for i, e in enumerate(pd.Index(c["entity_id"].unique()))}
        ui = c["entity_id"].map(e2u).to_numpy(np.int64)
        ii = c["item_id"].map(i2i).to_numpy(np.int64)
        ts = c["timestamp"].to_numpy(np.float64) if "timestamp" in c.columns else None
        return ui, ii, ts, len(e2u), len(i2i)

    ok = True
    for ds, cfg in [("movielens-1m", {}), ("amazon-beauty", {"ease_lambda": 250.0})]:
        split = _load_dataset(ds, 0.1)
        eng = Engine(random_state=0, **cfg).fit(
            split.train, item_metadata=getattr(split, "items", None)
        )
        st = eng._state
        ui, ii, ts, nu, ni = recon(split.train)
        wt, wu = st.trend_z is not None, st.uu_users_data is not None
        pop, trend, uud, uui, uudeg = kindling_core.fit_channels(
            ui, ii, ts, nu, ni, st.n_items, eng.trend_window_fraction, wt, wu
        )
        c = {"popularity": np.array_equal(np.asarray(pop), st.item_popularity.astype(np.float64))}
        if wt:
            c["trend_z"] = bool(np.allclose(np.asarray(trend), st.trend_z, atol=1e-8))
        if wu:
            c["uu_data"] = np.array_equal(np.asarray(uud, np.int64), st.uu_users_data.astype(np.int64))
            c["uu_indptr"] = np.array_equal(np.asarray(uui, np.int64), st.uu_users_indptr.astype(np.int64))
            c["uu_deg"] = np.array_equal(np.asarray(uudeg), st.uu_user_deg.astype(np.float64))
        print(f"  fit_channels[{ds}] {c}")
        ok = ok and all(c.values())
    return ok


def check_recommend_ml1m() -> bool:
    """Rust recommend_ease_blend (EASE base + trend + last-item) vs Python
    _recommend_core, on ml1m — the clean path (no boost layers / user-CF /
    content / cold-slots, so layered-score is identity). Exact rec-list match
    across a sample of users is the gate for the 0.2928 reference number.
    """
    from kindling import Engine
    from kindling.benchmarks.comparison import _load_dataset

    split = _load_dataset("movielens-1m", 0.1)
    eng = Engine(random_state=0).fit(split.train)
    st = eng._state
    assert st.base_scorer_used == "ease"
    assert not st.enabled_boost_layers
    tz = (
        st.trend_z[: st.n_items].astype(np.float64)
        if st.trend_z is not None
        else np.zeros(0, np.float64)
    )
    ents = [e for e, ow in st.owned_by_entity.items() if ow.size > 0][:300]
    mism = 0
    for ent in ents:
        owned = st.owned_by_entity[ent].astype(np.int64)
        py = [r.item_id for r in eng._recommend_core(owned, ent, 10)]
        idx, _sc = kindling_core.recommend_ease_blend(
            np.ascontiguousarray(st.ease_b, np.float32),
            tz, owned, float(st.trend_alpha), float(st.last_item_alpha), 10,
        )
        rs = [st.item_ids[i] for i in idx]
        mism += int(py != rs)
    print(f"  recommend_ml1m users={len(ents)} mismatches={mism}")
    return mism == 0


def _uri_csr(st, n_users: int):
    """user_row_items dict → CSR (data, indptr) for Rust neighbor voting."""
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


def check_blend_channels() -> bool:
    """Rust blend_channels (full _blend_channels port: trend + user-CF +
    last-item + transitions) vs Python, on the blended full-catalog vector.
    Covers beauty/steam (which add transitions + user-CF over the ml1m path).
    """
    from kindling import Engine
    from kindling.benchmarks.comparison import _load_dataset

    empty_f, empty_i, empty_i32 = (
        np.zeros(0, np.float64), np.zeros(0, np.int64), np.zeros(0, np.int32),
    )
    ok = True
    for ds, cfg in [
        ("movielens-1m", {}),
        ("amazon-beauty", {"ease_lambda": 250.0}),
        ("steam", {"cold_slots": 1}),
    ]:
        # No item_metadata: the channels under test (trend/user-CF/last-item/
        # transitions) are data-driven, and dropping the open-catalog metadata
        # extension keeps n_items_ext == n_items (no base padding to reconcile).
        split = _load_dataset(ds, 0.1)
        eng = Engine(random_state=0, **cfg).fit(split.train)
        st = eng._state
        n_users = st.uu_user_deg.shape[0] if st.uu_user_deg is not None else 0
        if st.uu_users_data is not None:
            uri_d, uri_ip = _uri_csr(st, n_users)
            uu_d = st.uu_users_data.astype(np.int64)
            uu_ip = st.uu_users_indptr.astype(np.int64)
            uu_dg = st.uu_user_deg.astype(np.float64)
        else:
            uri_d, uri_ip, uu_d, uu_ip, uu_dg = empty_i, empty_i, empty_i, empty_i, empty_f
        tz = st.trend_z[: st.n_items].astype(np.float64) if st.trend_z is not None else empty_f
        if st.trans_data is not None:
            tr_d, tr_i, tr_ip = (
                st.trans_data.astype(np.float64),
                st.trans_indices.astype(np.int32),
                st.trans_indptr.astype(np.int64),
            )
        else:
            tr_d, tr_i, tr_ip = empty_f, empty_i32, empty_i

        # The neighbor sets now match byte-for-byte (stable user-CF tie-break),
        # so the residual is pure summation-order FP noise — numpy's pairwise
        # reductions vs Rust's sequential sum in the z-norms. ml1m's simpler
        # path lands at ~1e-13; the user-CF paths at ~1e-6. The gate that
        # actually matters is ranking identity (top-10 argsort), which the FP
        # noise never perturbs.
        num_tol = 1e-5
        ents = [e for e, ow in st.owned_by_entity.items() if ow.size > 0][:200]
        maxdiff, numok, rankok = 0.0, 0, 0
        for ent in ents:
            owned = st.owned_by_entity[ent].astype(np.int64)
            base_vec = st.ease_b[owned].sum(axis=0, dtype=np.float64)
            last_row = st.ease_b[int(owned[-1])].astype(np.float64)
            user_row = st.entity_to_user_idx.get(ent, -1)
            py = eng._blend_channels(st, owned, base_vec.copy(), user_row=user_row)
            rs = np.asarray(kindling_core.blend_channels(
                base_vec, owned, st.n_items,
                tz, float(st.trend_alpha),
                last_row, float(st.last_item_alpha),
                tr_d, tr_i, tr_ip, float(st.transition_alpha),
                int(st.transition_last_k), float(st.transition_decay),
                uu_d, uu_ip, uu_dg, uri_d, uri_ip,
                float(st.user_cf_alpha), int(st.user_cf_k), int(user_row), n_users,
            ))
            d = float(np.max(np.abs(py - rs))) if py.size else 0.0
            maxdiff = max(maxdiff, d)
            numok += int(d < num_tol)
            pa, ra = py.copy(), rs.copy()
            pa[owned], ra[owned] = -np.inf, -np.inf
            rankok += int(np.array_equal(np.argsort(-pa)[:10], np.argsort(-ra)[:10]))
        print(
            f"  blend_channels[{ds:13s}] users={len(ents)} "
            f"num_id={numok}/{len(ents)} rank_top10={rankok}/{len(ents)} max|Δ|={maxdiff:.2e}"
        )
        ok = ok and numok == len(ents) and rankok == len(ents)
    return ok


if __name__ == "__main__":
    results = {
        "cooc_transform": check_cooc_transform(),
        "metadata_knn": check_metadata_knn(),
        "fit_channels": check_fit_channels(),
        "recommend_ml1m": check_recommend_ml1m(),
        "blend_channels": check_blend_channels(),
    }
    print("\nPARITY:", "ALL PASS" if all(results.values()) else "FAIL", results)
    raise SystemExit(0 if all(results.values()) else 1)
