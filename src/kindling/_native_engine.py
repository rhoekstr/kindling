"""Native Rust recommend engine — build a ``kindling_core.EngineState`` from a
fitted :class:`~kindling.engine.Engine` and serve recommendations from it.

Parity-first: the native engine is *built from* the Python ``EngineState``
(Python keeps fit orchestration) and reproduces ``_recommend_core`` for the
supported path — EASE base + channel blend + temporal-cooc boost layer +
cold-slots. It is the fast batch path (``Engine.recommend_batch``); single
``recommend`` stays on the Python reference. See ``docs/RUST-ENGINE-PLAN.md``
and ``bench/native_recommend_parity.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from kindling._native import CORE_AVAILABLE, kindling_core

if TYPE_CHECKING:
    from kindling.engine import Engine


def native_supported(engine: Engine) -> bool:
    """Whether the native engine can serve this fit.

    Requires the Rust extension, an EASE base, no active content channel in the
    blend (not ported), and no path-basket boost layer (served by basket_index,
    not ported). The cooc base (book) is handled by the Python path.
    """
    if not CORE_AVAILABLE or not hasattr(kindling_core, "build_engine"):
        return False
    st = engine._state
    if st is None or st.ease_b is None:
        return False
    cf = st.content_features
    if st.content_alpha > 0.0 and cf is not None and cf.n_features > 0:
        return False
    basket = st.basket_index
    return not (basket is not None and getattr(basket, "observations", None))


def _uri_csr(st: Any, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    """user_row_items dict → CSR (data, indptr) for native neighbor voting."""
    indptr = np.zeros(n_users + 1, np.int64)
    parts: list[np.ndarray] = []
    for u in range(n_users):
        items = st.user_row_items.get(u)
        if items is not None and len(items) > 0:
            parts.append(np.asarray(items, np.int64))
            indptr[u + 1] = indptr[u] + len(items)
        else:
            indptr[u + 1] = indptr[u]
    data = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    return data, indptr


def build_native_engine(engine: Engine) -> Any | None:
    """Construct the Rust ``EngineState`` from a fitted engine, or ``None`` if
    the native path doesn't support this fit (see :func:`native_supported`)."""
    if not native_supported(engine):
        return None
    st = engine._state
    assert st is not None
    n_users = st.uu_user_deg.shape[0] if st.uu_user_deg is not None else 0
    arrays: dict[str, Any] = {"ease_b": np.ascontiguousarray(st.ease_b, np.float32)}
    if st.trend_z is not None:
        arrays["trend_z"] = st.trend_z.astype(np.float64)
    if st.trans_data is not None:
        assert st.trans_indices is not None
        assert st.trans_indptr is not None
        arrays["trans_data"] = st.trans_data.astype(np.float64)
        arrays["trans_indices"] = st.trans_indices.astype(np.int32)
        arrays["trans_indptr"] = st.trans_indptr.astype(np.int64)
    if st.uu_users_data is not None:
        assert st.uu_users_indptr is not None
        assert st.uu_user_deg is not None
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
    if cf is not None and cf.n_features > 0 and engine.cold_slots > 0:
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
        retrieval_budget=int(engine.retrieval_budget),
        cold_slots=int(engine.cold_slots),
        content_nfeat=content_nfeat,
        cold_recency_beta=float(st.cold_recency_beta or 0.0),
    )
    return kindling_core.build_engine(arrays, config)
