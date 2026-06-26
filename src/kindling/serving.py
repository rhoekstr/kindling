"""Serving harness — a self-contained, framework-agnostic recommender server.

``KindlingServer`` wraps a fitted engine's native Rust recommend core plus the
catalog mappings it needs to answer requests, and persists the pair as a
**serving artifact** (a directory: ``engine.bin`` bincode blob + ``catalog.pkl``
mappings). Load the artifact in a serving process and call ``recommend`` /
``recommend_batch`` / ``recommend_for_items`` — no re-fit, no training-time
dependencies (pandas, the eval harness) required.

The class has no web-framework dependency; ``kindling.serving_app`` is a thin
FastAPI example built on top (``pip install 'kindling[serve]'``).

    from kindling import Engine
    from kindling.serving import KindlingServer

    server = KindlingServer.from_engine(Engine().fit(interactions))
    server.save("artifact/")
    # ... in the serving process ...
    server = KindlingServer.load("artifact/")
    server.recommend("user-42", n=10)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from kindling._native import kindling_core
from kindling.engine import Recommendation

if TYPE_CHECKING:
    from kindling.engine import Engine

_ARTIFACT_VERSION = 1
_ENGINE_BIN = "engine.bin"
_CATALOG_PKL = "catalog.pkl"


class KindlingServer:
    """Serves recommendations from a persisted native engine + catalog."""

    def __init__(
        self,
        native: Any,
        *,
        item_ids: np.ndarray,
        item_to_idx: dict[object, int],
        owned_by_entity: dict[object, np.ndarray],
        entity_to_user_idx: dict[object, int],
        item_popularity: np.ndarray | None,
        cold_user_pop_prior: float = 5.0,
    ) -> None:
        self._native = native
        self._item_ids = item_ids
        self._item_to_idx = item_to_idx
        self._owned_by_entity = owned_by_entity
        self._entity_to_user_idx = entity_to_user_idx
        self._item_popularity = item_popularity
        self._cold_user_pop_prior = float(cold_user_pop_prior)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_engine(cls, engine: Engine) -> KindlingServer:
        """Build a server from a fitted :class:`~kindling.engine.Engine`."""
        st = engine._state
        if st is None:
            raise RuntimeError("Engine is not fitted; nothing to serve.")
        native = engine._require_native()  # builds + validates the native core
        return cls(
            native,
            item_ids=st.item_ids,
            item_to_idx=st.item_to_idx,
            owned_by_entity=st.owned_by_entity,
            entity_to_user_idx=st.entity_to_user_idx,
            item_popularity=st.item_popularity,
            cold_user_pop_prior=engine.cold_user_pop_prior,
        )

    # ------------------------------------------------------------------
    # serving
    # ------------------------------------------------------------------
    def recommend(self, entity_id: object, n: int = 10) -> list[Recommendation]:
        """Recommend for a known entity; zero-history entities get popularity."""
        owned = self._owned_by_entity.get(entity_id)
        if owned is None or len(owned) == 0:
            return self._cold(n)
        user_row = int(self._entity_to_user_idx.get(entity_id, -1))
        return self._wrap(self._native.recommend([int(x) for x in owned], user_row, n, 0.0))

    def recommend_batch(self, entity_ids: list[object], n: int = 10) -> list[list[Recommendation]]:
        """Recommend for many known entities in parallel (native batch path)."""
        ids = list(entity_ids)
        results: list[list[Recommendation]] = [[] for _ in ids]
        owneds: list[list[int]] = []
        user_rows: list[int] = []
        positions: list[int] = []
        for k, ent in enumerate(ids):
            owned = self._owned_by_entity.get(ent)
            if owned is None or len(owned) == 0:
                results[k] = self._cold(n)
            else:
                owneds.append([int(x) for x in owned])
                user_rows.append(int(self._entity_to_user_idx.get(ent, -1)))
                positions.append(k)
        if owneds:
            for j, k in enumerate(positions):
                results[k] = self._wrap(self._native.recommend_batch(owneds, user_rows, n, 0.0)[j])
        return results

    def recommend_for_items(
        self, seed_item_ids: list[object], n: int = 10
    ) -> list[Recommendation]:
        """Serve a new / anonymous user from ad-hoc seed items (stateless)."""
        seen: set[int] = set()
        owned: list[int] = []
        for it in seed_item_ids:
            ix = self._item_to_idx.get(it)
            if ix is not None and ix not in seen:
                seen.add(ix)
                owned.append(ix)
        if not owned:
            return self._cold(n)
        pop_prior = self._cold_user_pop_prior / len(owned)
        return self._wrap(self._native.recommend(owned, -1, n, pop_prior))

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Write the serving artifact (a directory) to ``path``."""
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        self._native.save(str(d / _ENGINE_BIN))
        catalog = {
            "version": _ARTIFACT_VERSION,
            "item_ids": self._item_ids,
            "item_to_idx": self._item_to_idx,
            "owned_by_entity": self._owned_by_entity,
            "entity_to_user_idx": self._entity_to_user_idx,
            "item_popularity": self._item_popularity,
            "cold_user_pop_prior": self._cold_user_pop_prior,
        }
        with open(d / _CATALOG_PKL, "wb") as fh:
            pickle.dump(catalog, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> KindlingServer:
        """Load a serving artifact written by :meth:`save`."""
        d = Path(path)
        native = kindling_core.EngineState.load(str(d / _ENGINE_BIN))
        with open(d / _CATALOG_PKL, "rb") as fh:
            c = pickle.load(fh)
        if c.get("version") != _ARTIFACT_VERSION:
            raise ValueError(f"unsupported artifact version {c.get('version')!r}")
        return cls(
            native,
            item_ids=c["item_ids"],
            item_to_idx=c["item_to_idx"],
            owned_by_entity=c["owned_by_entity"],
            entity_to_user_idx=c["entity_to_user_idx"],
            item_popularity=c["item_popularity"],
            cold_user_pop_prior=c["cold_user_pop_prior"],
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _wrap(self, result: tuple[list[int], list[float], list[str]]) -> list[Recommendation]:
        items, scores, kinds = result
        return [
            Recommendation(item_id=self._item_ids[i], score=float(s), base_kind=k)
            for i, s, k in zip(items, scores, kinds)
        ]

    def _cold(self, n: int) -> list[Recommendation]:
        """Zero-history fallback: top-n by all-time popularity."""
        pop = self._item_popularity
        if pop is None or pop.size == 0:
            return []
        n_eff = min(n, pop.size)
        top = np.argpartition(-pop, n_eff - 1)[:n_eff]
        top = top[np.argsort(-pop[top], kind="stable")]
        return [
            Recommendation(item_id=self._item_ids[int(c)], score=float(pop[c]), base_kind="cold_popularity")
            for c in top
        ]
