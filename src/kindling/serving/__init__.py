"""kindling serving — a small HTTP surface over a fitted engine.

``create_app`` turns a fitted (or saved-and-reloaded) :class:`~kindling.Engine`
into a FastAPI application exposing ``recommend`` for known users,
``recommend_for_items`` for new / anonymous users, and a batch endpoint. The
FastAPI/uvicorn dependency is optional and imported lazily, so importing
``kindling`` never requires them — install with ``pip install kindling[serve]``.
"""

from __future__ import annotations

from kindling.serving.app import create_app, serve

__all__ = ["create_app", "serve"]
