"""Baseline recommenders for apples-to-apples comparison against kindling.

All baselines expose the same two-method surface:

    baseline.fit(interactions)             # pd.DataFrame with entity_id, item_id
    baseline.recommend(entity_id, n=10)    # list of item_ids

Baselines are intentionally minimal - no reranking, no diversity, no
constraints - so the comparison isolates the retriever/ranker quality.
Each excludes items the entity already interacted with in the train set
(matching kindling's default behavior).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class PopularityBaseline:
    """Most-interacted items, with the entity's own history masked out.

    Trivial but often hard to beat on sparse public datasets.
    """

    name: str = "popularity"
    _ranked_items: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    _owned: dict[object, set[object]] = field(default_factory=dict)

    def fit(self, interactions: pd.DataFrame) -> "PopularityBaseline":
        counts = interactions["item_id"].value_counts()
        self._ranked_items = counts.index.to_numpy()
        self._owned = {
            entity: set(group["item_id"].tolist())
            for entity, group in interactions.groupby("entity_id", sort=False)
        }
        return self

    def recommend(self, entity_id: object, n: int = 10) -> list[object]:
        owned = self._owned.get(entity_id, set())
        out: list[object] = []
        for item in self._ranked_items:
            if item in owned:
                continue
            out.append(item)
            if len(out) >= n:
                break
        return out


@dataclass
class ItemItemKNN:
    """Item-item cosine kNN on the binarized user-item matrix.

    Score(entity, item) = sum over owned j of cos(i, j). Standard collaborative
    filtering baseline - no learning, just linear algebra. Industry-tested.
    """

    name: str = "item_item_knn"
    k_neighbors: int = 200
    _item_sim: sp.csr_matrix = field(default_factory=lambda: sp.csr_matrix((0, 0)))
    _user_items: sp.csr_matrix = field(default_factory=lambda: sp.csr_matrix((0, 0)))
    _entity_ix: dict[object, int] = field(default_factory=dict)
    _item_ix: dict[object, int] = field(default_factory=dict)
    _ix_item: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))

    def fit(self, interactions: pd.DataFrame) -> "ItemItemKNN":
        entities = interactions["entity_id"].unique()
        items = interactions["item_id"].unique()
        self._entity_ix = {e: i for i, e in enumerate(entities)}
        self._item_ix = {m: i for i, m in enumerate(items)}
        self._ix_item = np.asarray(items, dtype=object)

        rows = interactions["entity_id"].map(self._entity_ix).to_numpy()
        cols = interactions["item_id"].map(self._item_ix).to_numpy()
        data = np.ones(len(interactions), dtype=np.float32)
        ui = sp.csr_matrix(
            (data, (rows, cols)), shape=(len(entities), len(items)), dtype=np.float32
        )
        # Deduplicate co-interactions (collapse repeat visits).
        ui.sum_duplicates()
        ui.data = np.minimum(ui.data, 1.0)
        self._user_items = ui.tocsr()

        # L2-normalize columns for cosine.
        norms = np.sqrt(np.asarray(ui.multiply(ui).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        inv = sp.diags(1.0 / norms)
        normed = ui @ inv  # (users, items) with unit-norm columns
        sim = normed.T @ normed  # (items, items)
        sim.setdiag(0.0)
        sim.eliminate_zeros()
        # Keep only top-k per row for tractable scoring.
        self._item_sim = _keep_top_k_per_row(sim.tocsr(), self.k_neighbors)
        return self

    def recommend(self, entity_id: object, n: int = 10) -> list[object]:
        if entity_id not in self._entity_ix:
            return []
        uid = self._entity_ix[entity_id]
        owned_row = self._user_items.getrow(uid)
        scores = np.asarray((owned_row @ self._item_sim).todense()).ravel()
        owned_cols = owned_row.indices
        scores[owned_cols] = -np.inf
        if n >= len(scores):
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, n)[:n]
            order = part[np.argsort(-scores[part])]
        return [self._ix_item[i] for i in order[:n] if np.isfinite(scores[i])]


@dataclass
class ImplicitALSBaseline:
    """Wrapper around ``implicit.als.AlternatingLeastSquares``.

    Weighted matrix factorization from Hu, Koren, Volinsky (2008). The
    canonical implicit-feedback baseline; `implicit` is the industry-standard
    implementation.
    """

    name: str = "implicit_als"
    factors: int = 64
    regularization: float = 0.01
    iterations: int = 15
    random_state: int = 0
    _model: object = None
    _user_items: sp.csr_matrix = field(default_factory=lambda: sp.csr_matrix((0, 0)))
    _entity_ix: dict[object, int] = field(default_factory=dict)
    _ix_item: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))

    def fit(self, interactions: pd.DataFrame) -> "ImplicitALSBaseline":
        import os

        # Respect the single-threaded bench environment.
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        from implicit.als import AlternatingLeastSquares

        entities = interactions["entity_id"].unique()
        items = interactions["item_id"].unique()
        self._entity_ix = {e: i for i, e in enumerate(entities)}
        item_ix = {m: i for i, m in enumerate(items)}
        self._ix_item = np.asarray(items, dtype=object)

        rows = interactions["entity_id"].map(self._entity_ix).to_numpy()
        cols = interactions["item_id"].map(item_ix).to_numpy()
        data = np.ones(len(interactions), dtype=np.float32)
        ui = sp.csr_matrix(
            (data, (rows, cols)), shape=(len(entities), len(items)), dtype=np.float32
        )
        ui.sum_duplicates()
        ui.data = np.minimum(ui.data, 1.0)
        self._user_items = ui.tocsr()

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            use_gpu=False,
        )
        self._model.fit(self._user_items, show_progress=False)  # type: ignore[attr-defined]
        return self

    def recommend(self, entity_id: object, n: int = 10) -> list[object]:
        if entity_id not in self._entity_ix or self._model is None:
            return []
        uid = self._entity_ix[entity_id]
        ids, _ = self._model.recommend(  # type: ignore[attr-defined]
            uid,
            self._user_items[uid],
            N=n,
            filter_already_liked_items=True,
        )
        return [self._ix_item[int(i)] for i in ids]


def _keep_top_k_per_row(mat: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Zero out all but the top-k entries per row. In-place-friendly."""
    if k <= 0 or mat.nnz == 0:
        return mat
    rows, cols, data = [], [], []
    for i in range(mat.shape[0]):
        start, end = mat.indptr[i], mat.indptr[i + 1]
        if end - start <= k:
            rows.extend([i] * (end - start))
            cols.extend(mat.indices[start:end].tolist())
            data.extend(mat.data[start:end].tolist())
            continue
        row_data = mat.data[start:end]
        idx = np.argpartition(-row_data, k)[:k]
        rows.extend([i] * k)
        cols.extend(mat.indices[start:end][idx].tolist())
        data.extend(row_data[idx].tolist())
    return sp.csr_matrix((data, (rows, cols)), shape=mat.shape, dtype=mat.dtype)
