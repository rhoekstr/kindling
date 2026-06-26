"""FastAPI example for serving a kindling artifact — thin glue over
:class:`kindling.serving.KindlingServer`.

Install the optional deps and point at a saved artifact directory::

    pip install 'kindling[serve]'
    KINDLING_ARTIFACT=artifact/ uvicorn kindling.serving_app:app

Endpoints:
    GET  /healthz                          → {"status": "ok", "items": N}
    POST /recommend       {entity_id, n}   → [{item_id, score, base_kind}, ...]
    POST /recommend_batch {entity_ids, n}  → [[...], ...]
    POST /recommend_for_items {seeds, n}   → [...]  (new / anonymous user)

The class underneath has no web dependency; swap FastAPI for any framework.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "kindling.serving_app needs FastAPI + pydantic — `pip install 'kindling[serve]'`."
    ) from exc

from kindling.serving import KindlingServer


def _rec_to_dict(r: Any) -> dict[str, Any]:
    return {"item_id": r.item_id, "score": r.score, "base_kind": r.base_kind}


def create_app(artifact_path: str | None = None) -> FastAPI:
    """Build the FastAPI app, loading the artifact from ``artifact_path`` or
    the ``KINDLING_ARTIFACT`` environment variable."""
    path = artifact_path or os.environ.get("KINDLING_ARTIFACT")
    if not path:
        raise RuntimeError("set KINDLING_ARTIFACT (or pass artifact_path) to a saved artifact dir.")
    server = KindlingServer.load(path)
    app = FastAPI(title="kindling", version="0.2.0")

    class RecommendReq(BaseModel):
        entity_id: Any
        n: int = 10

    class BatchReq(BaseModel):
        entity_ids: list[Any]
        n: int = 10

    class SeedReq(BaseModel):
        seeds: list[Any]
        n: int = 10

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "items": len(server._item_ids)}

    @app.post("/recommend")
    def recommend(req: RecommendReq) -> list[dict[str, Any]]:
        return [_rec_to_dict(r) for r in server.recommend(req.entity_id, req.n)]

    @app.post("/recommend_batch")
    def recommend_batch(req: BatchReq) -> list[list[dict[str, Any]]]:
        return [[_rec_to_dict(r) for r in recs] for recs in server.recommend_batch(req.entity_ids, req.n)]

    @app.post("/recommend_for_items")
    def recommend_for_items(req: SeedReq) -> list[dict[str, Any]]:
        return [_rec_to_dict(r) for r in server.recommend_for_items(req.seeds, req.n)]

    return app


# Module-level app for `uvicorn kindling.serving_app:app` (needs KINDLING_ARTIFACT).
app = create_app() if os.environ.get("KINDLING_ARTIFACT") else None
