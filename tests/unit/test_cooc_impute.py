"""Unit + wiring tests for embedding-imputation cold-start placement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from kindling.engine_v2 import EngineV2
from kindling.graph.cooc_impute import (
    ImputeModel,
    _neighbor_recovery,
    cold_scores,
    fit_impute,
    ppmi,
)


# ── synthetic clustered dataset: genres by clusters; each user consumes within
# ONE (genre, cluster), so co-occurrence is cluster-block-diagonal and the
# cluster tag predicts cooc structure — the regime imputation is for. Many
# blocks give the transfer R^2 real (content-predictable) dimensions to fit, so
# R^2 is reliably positive. User u's (genre, cluster) is deterministic so
# entity 0 always sits in cluster (0, 0) for the placement assertions. ──
def _clustered_dataset(seed: int = 0, genres: int = 2, clusters: int = 8,
                       per_cluster: int = 10, n_users: int = 2500):
    rng = np.random.default_rng(seed)
    warm = [
        f"w{g}_{c}_{i}"
        for g in range(genres) for c in range(clusters) for i in range(per_cluster)
    ]
    rows = []
    for u in range(n_users):
        g, c = u % genres, (u // genres) % clusters
        pool = [it for it in warm if it.startswith(f"w{g}_{c}_")]
        k = int(rng.integers(5, per_cluster + 1))
        for it in rng.choice(pool, size=k, replace=False):
            rows.append((u, it))
    inter = pd.DataFrame(rows, columns=["entity_id", "item_id"])

    def tag(g: int, c: int) -> str:
        return f"cl{g}_{c}"

    meta_rows = [
        (it, tag(int(it.split("_")[0][1:]), int(it.split("_")[1]))) for it in warm
    ]
    # cold, metadata-only items (never in train) — one per (genre, cluster).
    cold_ids = [f"cold{g}_{c}" for g in range(genres) for c in range(clusters)]
    meta_rows += [
        (f"cold{g}_{c}", tag(g, c)) for g in range(genres) for c in range(clusters)
    ]
    meta = pd.DataFrame(meta_rows, columns=["item_id", "tags"])
    return inter, meta, cold_ids


# ── module-level math ───────────────────────────────────────────────────────
def test_ppmi_is_positive_and_shape_preserving():
    co = sp.coo_matrix(
        (np.array([4.0, 4.0, 1.0]), (np.array([0, 1, 0]), np.array([1, 0, 2]))),
        shape=(3, 3),
    )
    out = ppmi(co, item_counts=np.array([10.0, 8.0, 4.0]), n_users=20)
    assert out.shape == (3, 3)
    assert np.all(out.data > 0.0)  # PPMI drops non-positive cells


def test_neighbor_recovery_perfect_when_prediction_equals_truth():
    y = np.random.default_rng(0).standard_normal((50, 8))
    assert _neighbor_recovery(y, y.copy()) == 1.0


def test_fit_impute_thin_data_returns_inert_model():
    # 3 items, all below warm_min → no usable map.
    cooc = sp.csr_matrix(np.zeros((3, 3)))
    content = sp.csr_matrix(np.eye(3))
    m = fit_impute(
        cooc.data, cooc.indices, cooc.indptr, content,
        item_counts=np.array([1.0, 1.0, 1.0]), n_users=3, n_items=3,
    )
    assert m.r2 == 0.0
    assert m.n_warm < 10
    assert not m.positions.any()  # zero positions → score 0


def test_fit_impute_shapes_and_warm_mask():
    inter, meta, _ = _clustered_dataset()
    # Drive the model through the engine to reuse its cooc/content build.
    eng = EngineV2(cold_slots=1, cold_impute="impute").fit(inter, item_metadata=meta)
    m = eng._state.impute_model
    assert isinstance(m, ImputeModel)
    n_ext = eng._state.n_items
    assert m.positions.shape == (n_ext, m.dim)
    # warm items carry a true (nonzero) cooc embedding.
    assert m.warm.sum() >= 10
    assert m.positions[m.warm].any()


# ── cold_scores semantics: a cold item is placed near the user's cooc cluster ─
def test_cold_scores_places_cold_item_in_users_cluster():
    inter, meta, cold_ids = _clustered_dataset()
    eng = EngineV2(cold_slots=1, cold_impute="impute").fit(inter, item_metadata=meta)
    st = eng._state
    cs = cold_scores(st.impute_model, st.owned_by_entity[0])  # entity 0 → (0, 0)
    ranked = sorted(cold_ids, key=lambda c: cs[st.item_to_idx[c]], reverse=True)
    assert ranked[0] == "cold0_0"  # the user's own-cluster cold item ranks first


# ── engine wiring ───────────────────────────────────────────────────────────
def test_cold_slots_zero_builds_no_impute_model():
    inter, meta, _ = _clustered_dataset()
    eng = EngineV2().fit(inter, item_metadata=meta)  # cold_slots=0 default
    assert eng._state.impute_model is None
    assert "cold_impute_active" not in eng._state.profile


def test_cold_impute_forced_activates_and_records_profile():
    inter, meta, _ = _clustered_dataset()
    eng = EngineV2(cold_slots=1, cold_impute="impute").fit(inter, item_metadata=meta)
    prof = eng._state.profile
    assert prof["cold_impute_active"] is True
    assert eng._state.impute_model is not None
    assert "cold_impute_r2" in prof
    assert "cold_impute_neighbor_recovery" in prof


def test_cold_impute_defaults_to_content_ranker():
    # Default is "content" (impute did not transfer through the engine), so
    # cold_slots without an explicit cold_impute builds no imputation model.
    inter, meta, _ = _clustered_dataset()
    eng = EngineV2(cold_slots=1).fit(inter, item_metadata=meta)
    assert eng._state.impute_model is None
    assert "cold_impute_active" not in eng._state.profile
    cold = [r for r in eng.recommend(0, n=10) if r.base_kind == "cold_content"]
    assert len(cold) == 1


def test_cold_impute_content_mode_uses_content_ranker():
    inter, meta, _ = _clustered_dataset()
    eng = EngineV2(cold_slots=1, cold_impute="content").fit(inter, item_metadata=meta)
    # content ranker → no imputation model, but cold slots still fill.
    assert eng._state.impute_model is None
    cold = [r for r in eng.recommend(0, n=10) if r.base_kind == "cold_content"]
    assert len(cold) == 1


def test_cold_impute_auto_respects_r2_floor():
    inter, meta, _ = _clustered_dataset()
    high = EngineV2(
        cold_slots=1, cold_impute="auto", cold_impute_min_r2=0.99
    ).fit(inter, item_metadata=meta)
    assert high._state.profile["cold_impute_active"] is False
    assert high._state.impute_model is None  # falls back to content
    low = EngineV2(
        cold_slots=1, cold_impute="auto", cold_impute_min_r2=0.0
    ).fit(inter, item_metadata=meta)
    assert low._state.profile["cold_impute_active"] is True
    assert low._state.impute_model is not None


def test_recommend_reserves_a_cold_slot_for_users_cluster_item():
    inter, meta, _ = _clustered_dataset()
    eng = EngineV2(cold_slots=1, cold_impute="impute").fit(inter, item_metadata=meta)
    recs = eng.recommend(0, n=10)
    cold = [r for r in recs if r.base_kind == "cold_content"]
    assert len(cold) == 1
    assert cold[0].item_id == "cold0_0"  # entity 0's own-cluster cold item


def test_invalid_cold_impute_rejected():
    with pytest.raises(ValueError, match="cold_impute"):
        EngineV2(cold_impute="bogus")
    with pytest.raises(ValueError, match="cold_impute_min_r2"):
        EngineV2(cold_impute_min_r2=-0.5)
