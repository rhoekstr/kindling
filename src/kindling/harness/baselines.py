"""Baseline recommenders for the eval harness.

``popularity`` has no dependencies and is always available — it is also the
single most important baseline, because a recommender that cannot beat
"most-purchased, minus what you own" is not personalizing. The trained
baselines (item-kNN, ALS, BPR) come from the optional ``implicit`` library;
they are skipped with a logged note when it is not installed, so the harness
degrades gracefully to a popularity-only comparison.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd


class Baseline(Protocol):
    """A fitted baseline that can rank a held-out catalog for one user."""

    name: str

    def recommend(self, entity_id: object, owned: set[object], k: int) -> list[object]:
        """Top-``k`` item ids for ``entity_id``, excluding already-owned items."""
        ...


class PopularityBaseline:
    """Global most-frequent items, minus what the user already has."""

    name = "popularity"

    def __init__(self, train: pd.DataFrame) -> None:
        self._order: list[object] = train["item_id"].value_counts().index.tolist()

    def recommend(self, entity_id: object, owned: set[object], k: int) -> list[object]:
        out: list[object] = []
        for item in self._order:
            if item in owned:
                continue
            out.append(item)
            if len(out) >= k:
                break
        return out


class _ImplicitBaseline:
    """Adapter over an ``implicit`` model (ALS / BPR / CosineRecommender)."""

    def __init__(
        self,
        name: str,
        model: Any,  # an implicit model (untyped third-party)
        ui: Any,  # scipy.sparse.csr_matrix
        u_row: dict[object, int],
        col_item: np.ndarray,
    ) -> None:
        self.name = name
        self._model = model
        self._ui = ui
        self._u_row = u_row
        self._col_item = col_item

    def recommend(self, entity_id: object, owned: set[object], k: int) -> list[object]:
        row = self._u_row.get(entity_id)
        if row is None:
            return []
        ids, _ = self._model.recommend(row, self._ui[row], N=k, filter_already_liked_items=True)
        return [self._col_item[c] for c in ids]


def available_baselines() -> list[str]:
    """Names the harness can build right now (``implicit`` ones gated on import)."""
    names = ["popularity"]
    try:
        import implicit  # noqa: F401

        names += ["item-knn", "als", "bpr"]
    except ImportError:
        pass
    return names


def build_baselines(
    train: pd.DataFrame, requested: list[str], *, seed: int = 0
) -> tuple[list[Baseline], list[str]]:
    """Build the requested baselines; return ``(built, skipped_with_reason)``.

    ``popularity`` is always built. The trained baselines share one fitted
    user×item matrix. Unknown or unavailable names are returned in the second
    list (as ``"name: reason"``) rather than raising, so a partial comparison
    still runs.
    """
    requested = [r.lower() for r in requested]
    built: list[Baseline] = []
    skipped: list[str] = []
    if "popularity" in requested or not requested:
        built.append(PopularityBaseline(train))

    trained = [r for r in requested if r in {"item-knn", "als", "bpr"}]
    unknown = [r for r in requested if r not in {"popularity", "item-knn", "als", "bpr"}]
    skipped += [f"{u}: unknown baseline" for u in unknown]
    if not trained:
        return built, skipped

    try:
        import scipy.sparse as sp
        from implicit.als import AlternatingLeastSquares
        from implicit.bpr import BayesianPersonalizedRanking
        from implicit.nearest_neighbours import CosineRecommender
    except ImportError:
        skipped += [
            f"{t}: requires the 'baselines' extra (pip install kindling[baselines])"
            for t in trained
        ]
        return built, skipped

    users = sorted(train["entity_id"].unique(), key=str)
    items = sorted(train["item_id"].unique(), key=str)
    u_row = {u: i for i, u in enumerate(users)}
    i_col = {it: j for j, it in enumerate(items)}
    col_item = np.array(items, dtype=object)
    ui = sp.csr_matrix(
        (
            np.ones(len(train), dtype=np.float32),
            (
                train["entity_id"].map(u_row).to_numpy(),
                train["item_id"].map(i_col).to_numpy(),
            ),
        ),
        shape=(len(users), len(items)),
    )
    ctors = {
        "item-knn": lambda: CosineRecommender(K=200),
        "als": lambda: AlternatingLeastSquares(factors=64, iterations=15, random_state=seed),
        "bpr": lambda: BayesianPersonalizedRanking(factors=64, iterations=80, random_state=seed),
    }
    for name in trained:
        model = ctors[name]()
        model.fit(ui, show_progress=False)
        built.append(_ImplicitBaseline(name, model, ui, u_row, col_item))
    return built, skipped
