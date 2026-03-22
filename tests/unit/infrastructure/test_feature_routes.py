"""Tests for feature_routes API router."""

from unittest.mock import AsyncMock, patch

import pytest

# Use a minimal FastAPI app with just the feature_router
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from phoenix_ml.infrastructure.http.feature_routes import feature_router

app = FastAPI()
app.include_router(feature_router)


@pytest.fixture
def mock_feature_store() -> AsyncMock:
    store = AsyncMock()
    return store


async def test_get_features_requires_keys() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/features/entity1")
        assert resp.status_code == 400
        assert "keys" in resp.json()["detail"].lower()


async def test_get_features_not_found() -> None:
    with patch("src.infrastructure.http.feature_routes.feature_store") as mock_store:
        mock_store.get_online_features = AsyncMock(return_value=None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/features/entity1?keys=f1,f2")
            assert resp.status_code == 404


async def test_get_features_success() -> None:
    with patch("src.infrastructure.http.feature_routes.feature_store") as mock_store:
        mock_store.get_online_features = AsyncMock(return_value=[1.0, 2.0])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/features/entity1?keys=f1,f2")
            assert resp.status_code == 200
            data = resp.json()
            assert data["entity_id"] == "entity1"
            assert data["features"] == {"f1": 1.0, "f2": 2.0}


async def test_ingest_features() -> None:
    with patch("src.infrastructure.http.feature_routes.feature_store") as mock_store:
        mock_store.add_features = AsyncMock()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/features/ingest",
                json={"entity_id": "e1", "features": {"f1": 1.0, "f2": 2.0}},
            )
            assert resp.status_code == 200
            assert "success" in resp.json()["status"]


async def test_get_feature_metadata() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/features/metadata/income")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "income"
        assert data["dtype"] == "float"
