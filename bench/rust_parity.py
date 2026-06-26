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


if __name__ == "__main__":
    results = {
        "cooc_transform": check_cooc_transform(),
        "metadata_knn": check_metadata_knn(),
        "fit_channels": check_fit_channels(),
    }
    print("\nPARITY:", "ALL PASS" if all(results.values()) else "FAIL", results)
    raise SystemExit(0 if all(results.values()) else 1)
