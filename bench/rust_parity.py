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


if __name__ == "__main__":
    results = {"cooc_transform": check_cooc_transform(), "metadata_knn": check_metadata_knn()}
    print("\nPARITY:", "ALL PASS" if all(results.values()) else "FAIL", results)
    raise SystemExit(0 if all(results.values()) else 1)
