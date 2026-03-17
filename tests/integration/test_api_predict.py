import asyncio

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from src.infrastructure.http.fastapi_server import app

# Constants for testing
SUCCESS_STATUS = 200
CONCURRENT_REQUESTS = 3
TEST_TIMEOUT = 30.0  # seconds


@pytest.mark.asyncio
async def test_api_predict_dynamic_batching() -> None:
    """
    Integration test for the FastAPI /predict endpoint.
    Uses LifespanManager to ensure DB seeding and component startup happens.
    """
    async with asyncio.timeout(TEST_TIMEOUT):
        async with LifespanManager(app) as manager:
            transport = ASGITransport(app=manager.app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # 1. Send multiple concurrent prediction requests
                payload = {
                    "model_id": "credit-risk",
                    "model_version": "v1",
                    "entity_id": "customer-good",
                }

                # Simulating concurrent requests to trigger BatchManager
                tasks = [
                    client.post("/predict", json=payload)
                    for _ in range(CONCURRENT_REQUESTS)
                ]
                responses = await asyncio.gather(*tasks)

                # 2. Verify all responses are successful
                assert len(responses) == CONCURRENT_REQUESTS
                for response in responses:
                    if response.status_code != SUCCESS_STATUS:
                        print(f"❌ Error: {response.json()}")
                    assert response.status_code == SUCCESS_STATUS
                    data = response.json()
                    assert data["model_id"] == "credit-risk"
                    assert data["version"] == "v1"
                    assert "result" in data


@pytest.mark.asyncio
async def test_api_health() -> None:
    """
    Basic health check integration test.
    """
    async with asyncio.timeout(TEST_TIMEOUT):
        async with LifespanManager(app) as manager:
            transport = ASGITransport(app=manager.app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.get("/health")
                assert response.status_code == SUCCESS_STATUS
                assert response.json()["status"] == "healthy"
