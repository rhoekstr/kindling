"""HTTP serving surface — exercised with FastAPI's TestClient.

Skipped entirely unless the ``serve`` extra (FastAPI) and ``httpx`` (which
the TestClient needs) are installed, so the core test run has no new deps.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from kindling import Engine

pytest.importorskip("fastapi", reason="serving needs the [serve] extra")
pytest.importorskip("httpx", reason="FastAPI TestClient needs httpx")

# Importing starlette's TestClient can emit a third-party deprecation warning
# (some starlette versions prefer `httpx2`); the suite runs under -W error, and
# this fires at import time before any pytest filter applies — suppress it here.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from kindling.serving import create_app


@pytest.fixture
def client() -> TestClient:
    df = pd.DataFrame(
        {
            "entity_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
            "item_id": [10, 11, 12, 10, 11, 11, 12, 13, 10, 13],
        }
    )
    engine = Engine(random_state=0).fit(df)
    return TestClient(create_app(engine))


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_items"] == 4
    assert body["base_scorer"]


def test_info_lists_endpoints(client: TestClient) -> None:
    body = client.get("/").json()
    assert "/recommend" in body["endpoints"]
    assert "base_scorer" in body["activation"]


def test_recommend_known_user(client: TestClient) -> None:
    body = client.post("/recommend", json={"entity_id": 1, "n": 3}).json()
    recs = body["recommendations"]
    assert all("item_id" in r and "score" in r for r in recs)
    assert 10 not in [r["item_id"] for r in recs]  # user 1 already owns 10


def test_recommend_resolves_string_id(client: TestClient) -> None:
    # Catalog trained on int ids still resolves a string id from JSON.
    resp = client.post("/recommend", json={"entity_id": "2", "n": 3})
    assert resp.status_code == 200
    assert resp.json()["recommendations"]


def test_recommend_unknown_user_404(client: TestClient) -> None:
    assert client.post("/recommend", json={"entity_id": 999, "n": 3}).status_code == 404


def test_recommend_for_items_new_user(client: TestClient) -> None:
    body = client.post("/recommend_for_items", json={"item_ids": [10, 11], "n": 3}).json()
    assert body["recommendations"]
    assert body["fallback"] is False


def test_recommend_for_items_empty_falls_back(client: TestClient) -> None:
    body = client.post("/recommend_for_items", json={"item_ids": [], "n": 3}).json()
    assert body["fallback"] is True


def test_batch_mixes_request_shapes(client: TestClient) -> None:
    body = client.post(
        "/recommend/batch",
        json={"requests": [{"entity_id": 1, "n": 2}, {"item_ids": [10], "n": 2}]},
    ).json()
    results = body["results"]
    assert len(results) == 2
    assert "entity_id" in results[0]
    assert "seed_items" in results[1]


def test_recommend_validates_n_bounds(client: TestClient) -> None:
    assert client.post("/recommend", json={"entity_id": 1, "n": 0}).status_code == 422


def test_create_app_rejects_unfitted_engine() -> None:
    with pytest.raises(ValueError, match="not fitted"):
        create_app(Engine())
