"""FastAPI application factory for serving a fitted engine.

Endpoints
---------
* ``GET  /health``               liveness + catalog size + base scorer
* ``GET  /``                     service info + the activation plan
* ``POST /recommend``            known user: ``{entity_id, n}`` → recommendations
* ``POST /recommend_for_items``  new user: ``{item_ids, n}`` → recommendations
* ``POST /recommend/batch``      a list of mixed requests, in one call

All FastAPI/pydantic imports are deferred into :func:`create_app` so that
``import kindling.serving`` does not require the ``serve`` extra until you
actually build the app.

This module intentionally does **not** use ``from __future__ import
annotations``: FastAPI resolves an endpoint's body model from its real
annotation object, and the request models are defined *locally* inside
:func:`create_app` (so pydantic stays an optional import). Stringized
annotations would make FastAPI look the models up in the module globals,
where they do not exist, and silently demote the body to a query param.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from kindling.engine import Engine

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

_MISSING = (
    "kindling serving needs FastAPI + uvicorn. Install the extra:\n"
    "    pip install 'kindling[serve]'"
)


def _load_engine(engine_or_path: "Engine | str | Path") -> Engine:
    if isinstance(engine_or_path, Engine):
        if engine_or_path._state is None:
            raise ValueError("Engine is not fitted — call .fit(...) before serving.")
        return engine_or_path
    return Engine.load(engine_or_path)


def _resolve_entity(engine: Engine, entity_id: object) -> "object | None":
    """Match a JSON-decoded id against the trained keyspace.

    JSON has no int/str distinction at the schema level, so a catalog trained
    on integer ids still resolves a ``"42"`` request (and vice versa). Returns
    the matching stored key, or ``None`` if the user is unknown.
    """
    state = engine._state
    assert state is not None
    owned = state.owned_by_entity
    if entity_id in owned:
        return entity_id
    s = str(entity_id)
    if s in owned:
        return s
    if s.lstrip("-").isdigit() and int(s) in owned:
        return int(s)
    return None


def _rec_payload(recs: "list[Any]") -> "list[dict[str, Any]]":
    return [
        {"item_id": r.item_id, "score": round(float(r.score), 6), "base_kind": r.base_kind}
        for r in recs
    ]


def create_app(engine_or_path: "Engine | str | Path", *, title: str = "kindling") -> "FastAPI":
    """Build a FastAPI app serving recommendations from a fitted engine.

    ``engine_or_path`` is a fitted :class:`~kindling.Engine` or a path to one
    saved with :meth:`Engine.save`. Raises a clear :class:`ImportError` if the
    ``serve`` extra is not installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - exercised via the extra
        raise ImportError(_MISSING) from exc

    engine = _load_engine(engine_or_path)
    state = engine._state
    assert state is not None
    plan = engine.activation_plan

    class RecommendRequest(BaseModel):
        entity_id: int | str = Field(..., description="A user id present in training.")
        n: int = Field(10, ge=1, le=500)

    class ItemsRequest(BaseModel):
        item_ids: list[int | str] = Field(
            default_factory=list, description="Seed items for a new/anonymous user."
        )
        n: int = Field(10, ge=1, le=500)

    class BatchItem(BaseModel):
        entity_id: int | str | None = Field(None, description="Known user id.")
        item_ids: list[int | str] | None = Field(None, description="Or seed items.")
        n: int = Field(10, ge=1, le=500)

    class BatchRequest(BaseModel):
        requests: list[BatchItem] = Field(default_factory=list)

    app = FastAPI(title=title, version=_engine_version())

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "n_items": state.n_items,
            "n_users": len(state.owned_by_entity),
            "base_scorer": plan.base_scorer,
        }

    @app.get("/")
    def info() -> dict[str, Any]:
        return {
            "service": title,
            "n_items": state.n_items,
            "n_users": len(state.owned_by_entity),
            "activation": {
                "base_scorer": plan.base_scorer,
                "active_channels": plan.active_channels,
                "rating_weighted": plan.rating_weighted,
            },
            "endpoints": ["/health", "/recommend", "/recommend_for_items", "/recommend/batch"],
        }

    @app.post("/recommend")
    def recommend(req: RecommendRequest) -> dict[str, Any]:
        resolved = _resolve_entity(engine, req.entity_id)
        if resolved is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown entity_id {req.entity_id!r}; "
                "use /recommend_for_items for new users",
            )
        recs = engine.recommend(entity_id=resolved, n=req.n)
        return {"entity_id": req.entity_id, "recommendations": _rec_payload(recs)}

    @app.post("/recommend_for_items")
    def recommend_for_items(req: ItemsRequest) -> dict[str, Any]:
        recs = engine.recommend_for_items(seed_item_ids=req.item_ids, n=req.n)
        return {
            "seed_items": req.item_ids,
            "recommendations": _rec_payload(recs),
            "fallback": bool(recs) and recs[0].base_kind.startswith("cold"),
        }

    def _serve_one(item: "BatchItem") -> dict[str, Any]:
        if item.entity_id is not None:
            resolved = _resolve_entity(engine, item.entity_id)
            recs = engine.recommend(entity_id=resolved, n=item.n) if resolved is not None else []
            return {"entity_id": item.entity_id, "recommendations": _rec_payload(recs)}
        recs = engine.recommend_for_items(seed_item_ids=item.item_ids or [], n=item.n)
        return {"seed_items": item.item_ids or [], "recommendations": _rec_payload(recs)}

    @app.post("/recommend/batch")
    def recommend_batch(req: BatchRequest) -> dict[str, Any]:
        return {"results": [_serve_one(item) for item in req.requests]}

    return app


def _engine_version() -> str:
    try:
        from kindling import __version__

        return __version__
    except Exception:  # pragma: no cover - defensive
        return "0"


def serve(
    engine_or_path: "Engine | str | Path",
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Build the app and run it with uvicorn (blocking). Used by ``kindling serve``."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised via the extra
        raise ImportError(_MISSING) from exc
    app = create_app(engine_or_path)
    uvicorn.run(app, host=host, port=port)
