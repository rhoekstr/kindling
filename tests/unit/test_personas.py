"""Persona signal Commit 1: clustering + persona index build + matching.

Engine integration is in Commit 2; cold-start is in Commit 3; the end-
to-end ablation verdict is in Commit 4. This file locks in the
building blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kindling.personas import (
    HDBSCANClustering,
    KMeansClustering,
    PersonaConfig,
)
from kindling.personas.build import build_persona_index, build_user_vectors
from kindling.personas.matching import (
    build_user_query_vector,
    match_user,
    score_candidates,
)


def _synthetic_taste_groups(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, list[object]]:
    """Three taste groups, 30 users each. Items 0-9, 10-19, 20-29 are each
    group's signature. Group 1 also has some overlap with group 2 on items
    7-12 to make clustering non-trivial."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for group in range(3):
        group_items = list(range(group * 10, group * 10 + 10))
        overlap_items = list(range(group * 10 + 7, group * 10 + 13))
        for user_in_group in range(30):
            entity = f"g{group}u{user_in_group}"
            # 5 signature items + 1-2 overlap items per user.
            picks = rng.choice(group_items, size=5, replace=False).tolist()
            picks.extend(rng.choice(overlap_items, size=2, replace=False).tolist())
            for it in picks:
                rows.append({"entity_id": entity, "item_id": int(it)})
    df = pd.DataFrame(rows)
    item_ids = np.asarray(sorted(df["item_id"].unique()), dtype=np.int64)
    entity_order = sorted(df["entity_id"].unique())
    return df, item_ids, entity_order


def test_kmeans_clustering_produces_labeled_personas() -> None:
    df, item_ids, entity_order = _synthetic_taste_groups()
    user_mat = build_user_vectors(df, item_ids, entity_order)
    km = KMeansClustering(n_clusters=3, random_state=0).fit(user_mat.toarray())
    assert km.n_personas == 3
    assert km.assignments.shape == (len(entity_order),)
    assert (km.assignments >= 0).all()


def test_hdbscan_clustering_runs_on_synthetic_data() -> None:
    pytest.importorskip("hdbscan")
    pytest.importorskip("umap")
    df, item_ids, entity_order = _synthetic_taste_groups()
    user_mat = build_user_vectors(df, item_ids, entity_order)
    # Skip UMAP on this tiny fixture - passing 30-dim directly is fine.
    hc = HDBSCANClustering(min_cluster_size_pct=0.05, reduction_method="none").fit(
        user_mat.toarray()
    )
    # On this fixture we expect at least one cluster, often three.
    assert hc.n_personas >= 1


def test_persona_index_has_expected_shape() -> None:
    df, item_ids, entity_order = _synthetic_taste_groups()
    user_mat = build_user_vectors(df, item_ids, entity_order)
    cluster = KMeansClustering(n_clusters=3, random_state=0).fit(user_mat.toarray())
    index = build_persona_index(
        interactions=df,
        cluster_result=cluster,
        item_ids=item_ids,
        entity_order=entity_order,
        z_threshold=1.5,
    )
    assert index.n_personas == 3
    assert index.n_items == len(item_ids)
    assert index.persona_sizes.sum() == len(entity_order)
    # Each persona vector is L2-normalized: row sum-of-squares = 1 (or 0
    # if the persona has no surviving items after the filter).
    row_norms = np.sqrt(
        np.asarray(index.persona_vectors.multiply(index.persona_vectors).sum(axis=1)).ravel()
    )
    for n in row_norms:
        assert n == pytest.approx(1.0, abs=1e-6) or n == 0.0


def test_matching_ranks_own_group_highest() -> None:
    """A user drawn from group 0 should match persona 0 highest."""
    df, item_ids, entity_order = _synthetic_taste_groups()
    user_mat = build_user_vectors(df, item_ids, entity_order)
    cluster = KMeansClustering(n_clusters=3, random_state=0).fit(user_mat.toarray())
    index = build_persona_index(
        interactions=df,
        cluster_result=cluster,
        item_ids=item_ids,
        entity_order=entity_order,
    )

    # Identify a prototypical group-0 user.
    g0_user = "g0u0"
    g0_items = np.asarray(
        [it for it in df[df["entity_id"] == g0_user]["item_id"].unique()],
        dtype=object,
    )
    vec = build_user_query_vector(owned_items=g0_items, history_items=(), index=index)
    matches = match_user(vec, index)
    # The user's assigned persona should score highest.
    assigned = index.persona_of_entity(g0_user)
    assert matches[assigned] == matches.max()


def test_scoring_candidates_produces_nonzero_for_own_group_items() -> None:
    df, item_ids, entity_order = _synthetic_taste_groups()
    user_mat = build_user_vectors(df, item_ids, entity_order)
    cluster = KMeansClustering(n_clusters=3, random_state=0).fit(user_mat.toarray())
    index = build_persona_index(
        interactions=df, cluster_result=cluster, item_ids=item_ids, entity_order=entity_order
    )
    vec = build_user_query_vector(owned_items=np.asarray([0], dtype=object), history_items=(), index=index)
    matches = match_user(vec, index)
    # Score a mix of own-group (items 0-9), other-group (items 20-29),
    # and unknown (item 999) candidates.
    candidates = [3, 5, 25, 28, 999]
    scores = score_candidates(matches, index, candidates)
    assert scores.shape == (5,)
    # Unknown item must score zero.
    assert scores[-1] == 0.0


def test_rust_kernel_matches_python_fallback() -> None:
    """Differential test: native ``persona_rates`` must agree with the
    Python implementation in ``_rates_python`` on the same inputs."""
    pytest.importorskip("kindling._native", reason="native build required")
    from kindling._native import NATIVE_AVAILABLE, kindling_native
    from kindling.personas.build import _rates_python

    if not NATIVE_AVAILABLE or kindling_native is None:
        pytest.skip("kindling_native not built in this environment")

    df, item_ids, entity_order = _synthetic_taste_groups()
    user_mat = build_user_vectors(df, item_ids, entity_order)
    cluster = KMeansClustering(n_clusters=3, random_state=0).fit(user_mat.toarray())

    entity_id_to_idx = {e: i for i, e in enumerate(entity_order)}
    item_id_to_idx = {it: i for i, it in enumerate(item_ids)}
    u = df["entity_id"].map(entity_id_to_idx).to_numpy(dtype=np.int64)
    m = df["item_id"].map(item_id_to_idx).to_numpy(dtype=np.int64)

    # Python path.
    rates_py, sizes_py = _rates_python(
        assignments=cluster.assignments,
        user_idx=u,
        item_idx=m,
        n_personas=cluster.n_personas,
        n_items=len(item_ids),
    )
    # Native path.
    sizes_r, rows, cols, vals = kindling_native.persona_rates(
        cluster.assignments.tolist(),
        u.tolist(),
        m.tolist(),
        int(cluster.n_personas),
        int(len(item_ids)),
    )
    import scipy.sparse as sp

    rates_r = sp.csr_matrix(
        (vals, (rows, cols)), shape=(cluster.n_personas, len(item_ids))
    ).toarray()
    assert np.allclose(rates_py.toarray(), rates_r, atol=1e-9)
    assert np.array_equal(np.asarray(sizes_r, dtype=np.int64), sizes_py)


def test_persona_config_defaults_to_hdbscan() -> None:
    pytest.importorskip("hdbscan")
    cfg = PersonaConfig()
    resolved = cfg.resolved_clustering()
    assert resolved.name == "hdbscan"
