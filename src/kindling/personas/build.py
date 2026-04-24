"""Persona construction: from interactions + cluster assignments to a
fitted ``PersonaIndex`` (PRD supplement §2.3).

Stages:
1. Rate computation - Rust kernel ``persona_rates`` aggregates per-persona
   item rates.
2. Z-score filter - drop items with rate below mean - z_threshold*std.
3. TF-IDF weighting - rate-weighted denominator; log(1+rate) * idf.
4. L2 normalization - each persona vector becomes unit-length so cosine
   similarity reduces to a dot product at query time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from kindling._native import NATIVE_AVAILABLE, kindling_native
from kindling.personas.clustering import ClusterResult
from kindling.personas.index import PersonaIndex


def build_persona_index(
    interactions: pd.DataFrame,
    cluster_result: ClusterResult,
    item_ids: np.ndarray,
    entity_order: list[object],
    z_threshold: float = 1.5,
) -> PersonaIndex:
    """Build a ``PersonaIndex`` from interactions + cluster assignments.

    Parameters
    ----------
    interactions:
        DataFrame with ``entity_id`` and ``item_id`` columns.
    cluster_result:
        Output of a ``ClusteringProtocol.fit`` call. Assignments are
        aligned to ``entity_order``.
    item_ids:
        Catalog in the same order as the engine's item graph. Column
        indices in the resulting ``persona_vectors`` follow this order.
    entity_order:
        The entity ids in the same order as ``cluster_result.assignments``.
    z_threshold:
        Single-tailed z-score threshold for the noise filter.
    """
    item_id_to_idx = {item: i for i, item in enumerate(item_ids)}
    entity_id_to_user_idx = {e: i for i, e in enumerate(entity_order)}
    n_items = len(item_ids)
    n_personas = cluster_result.n_personas

    if n_personas == 0:
        return PersonaIndex(
            persona_vectors=sp.csr_matrix((0, n_items), dtype=np.float64),
            idf=np.ones(n_items, dtype=np.float64),
            persona_sizes=np.zeros(0, dtype=np.int64),
            item_id_to_idx=item_id_to_idx,
            user_to_persona=cluster_result.assignments,
            user_membership=cluster_result.probabilities,
            entity_id_to_user_idx=entity_id_to_user_idx,
        )

    # Map interactions to internal indices.
    ent_col = interactions["entity_id"]
    item_col = interactions["item_id"]
    user_idx = ent_col.map(entity_id_to_user_idx).to_numpy()
    item_idx = item_col.map(item_id_to_idx).to_numpy()
    mask = ~(pd.isna(user_idx) | pd.isna(item_idx))
    user_idx = user_idx[mask].astype(np.int64)
    item_idx = item_idx[mask].astype(np.int64)

    assignments = np.asarray(cluster_result.assignments, dtype=np.int64)

    # Stage 1: per-persona rates. Rust kernel if available, else scipy.
    if NATIVE_AVAILABLE and kindling_native is not None:
        persona_sizes, rows, cols, vals = kindling_native.persona_rates(
            assignments.tolist(),
            user_idx.tolist(),
            item_idx.tolist(),
            int(n_personas),
            int(n_items),
        )
        persona_sizes = np.asarray(persona_sizes, dtype=np.int64)
        rates = sp.csr_matrix(
            (np.asarray(vals, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(n_personas, n_items),
        )
    else:
        rates, persona_sizes = _rates_python(
            assignments=assignments,
            user_idx=user_idx,
            item_idx=item_idx,
            n_personas=n_personas,
            n_items=n_items,
        )

    # Stage 2: z-score filter. Per-persona, drop items whose rate is
    # more than z_threshold std-devs below the persona's mean rate.
    filtered = _zscore_filter(rates, z_threshold=z_threshold)

    # Stage 3: IDF with rate-weighted denominator (§2.3 step 3).
    # idf(i) = log(n_personas / (1 + sum_P rate(i, P)))
    rate_sum_per_item = np.asarray(filtered.sum(axis=0)).ravel()
    idf = np.log(n_personas / (1.0 + rate_sum_per_item))
    idf = np.clip(idf, 0.0, None)

    # Apply TF-IDF: weight = log(1 + rate) * idf
    tf = filtered.copy()
    tf.data = np.log1p(tf.data)
    weighted = tf.multiply(sp.csr_matrix(idf.reshape(1, -1))).tocsr()

    # Stage 4: L2 normalize each row (persona).
    norms = np.sqrt(np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel())
    inv = np.where(norms > 0.0, 1.0 / norms, 0.0)
    weighted = sp.diags(inv) @ weighted

    return PersonaIndex(
        persona_vectors=weighted.tocsr(),
        idf=idf,
        persona_sizes=persona_sizes,
        item_id_to_idx=item_id_to_idx,
        user_to_persona=assignments,
        user_membership=np.asarray(cluster_result.probabilities, dtype=np.float64),
        entity_id_to_user_idx=entity_id_to_user_idx,
    )


def _rates_python(
    assignments: np.ndarray,
    user_idx: np.ndarray,
    item_idx: np.ndarray,
    n_personas: int,
    n_items: int,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Python fallback for the native ``persona_rates`` kernel."""
    # Unique users per persona.
    persona_sizes = np.zeros(n_personas, dtype=np.int64)
    for p in range(n_personas):
        persona_sizes[p] = int(np.sum(assignments == p))

    # De-dup (user, item) pairs per persona.
    valid = (assignments[user_idx] >= 0) & (assignments[user_idx] < n_personas)
    u_valid = user_idx[valid]
    i_valid = item_idx[valid]
    p_valid = assignments[u_valid]

    # Use a DataFrame to dedupe (p, u, i) combos → unique (p, i) count.
    key = pd.DataFrame({"p": p_valid, "i": i_valid, "u": u_valid}).drop_duplicates()
    counts = key.groupby(["p", "i"]).size().reset_index(name="n")
    rows = counts["p"].to_numpy()
    cols = counts["i"].to_numpy()
    vals = counts["n"].to_numpy(dtype=np.float64)
    # Normalize by persona size.
    size_row = persona_sizes[rows].astype(np.float64)
    np.divide(vals, size_row, out=vals, where=size_row > 0)
    rates = sp.csr_matrix((vals, (rows, cols)), shape=(n_personas, n_items))
    return rates, persona_sizes


def _zscore_filter(rates: sp.csr_matrix, z_threshold: float) -> sp.csr_matrix:
    """Zero out entries where rate < mean - z_threshold * std (per persona)."""
    rates = rates.tocsr()
    out_rows: list[int] = []
    out_cols: list[int] = []
    out_vals: list[float] = []
    for p in range(rates.shape[0]):
        start, end = rates.indptr[p], rates.indptr[p + 1]
        if end == start:
            continue
        row_vals = rates.data[start:end]
        row_cols = rates.indices[start:end]
        mean = row_vals.mean()
        std = row_vals.std()
        if std <= 0.0:
            cutoff = -np.inf
        else:
            cutoff = mean - z_threshold * std
        keep_mask = row_vals > cutoff
        if not keep_mask.any():
            continue
        out_rows.extend([p] * int(keep_mask.sum()))
        out_cols.extend(row_cols[keep_mask].tolist())
        out_vals.extend(row_vals[keep_mask].tolist())
    return sp.csr_matrix(
        (out_vals, (out_rows, out_cols)),
        shape=rates.shape,
        dtype=np.float64,
    )


def build_user_vectors(
    interactions: pd.DataFrame,
    item_ids: np.ndarray,
    entity_order: list[object],
) -> sp.csr_matrix:
    """Build the (n_users, n_items) binary interaction matrix.

    Used as the input to dimensionality reduction + clustering. The
    caller may choose to reduce this (UMAP, ALS) before clustering;
    HDBSCANClustering handles reduction internally, KMeansClustering
    expects already-reduced input.
    """
    item_id_to_idx = {item: i for i, item in enumerate(item_ids)}
    entity_id_to_user_idx = {e: i for i, e in enumerate(entity_order)}
    n_users = len(entity_order)
    n_items = len(item_ids)

    from kindling.preprocess import weights_of

    user_idx = interactions["entity_id"].map(entity_id_to_user_idx).to_numpy()
    item_idx = interactions["item_id"].map(item_id_to_idx).to_numpy()
    weights = weights_of(interactions)
    mask = ~(pd.isna(user_idx) | pd.isna(item_idx))
    user_idx = user_idx[mask].astype(np.int64)
    item_idx = item_idx[mask].astype(np.int64)
    data = weights[mask].astype(np.float32)
    mat = sp.csr_matrix((data, (user_idx, item_idx)), shape=(n_users, n_items))
    mat.sum_duplicates()
    # Cap at 1.0 per (user, item) so duplicate rating upgrades don't
    # inflate the matrix unboundedly, and binary datasets (data=1)
    # recover the old behavior exactly.
    mat.data = np.minimum(mat.data, 1.0)
    return mat
