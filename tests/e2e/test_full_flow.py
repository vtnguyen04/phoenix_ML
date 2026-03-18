import asyncio

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from src.infrastructure.http.fastapi_server import app

SUCCESS_STATUS = 200
TEST_TIMEOUT = 30.0


@pytest.mark.asyncio
async def test_e2e_predict_and_feedback_flow() -> None:
    """
    End-to-End test: Health → Predict → Feedback.
    Validates the full user-facing flow works as a single pipeline.
    """
    async with asyncio.timeout(TEST_TIMEOUT):
        async with LifespanManager(app) as manager:
            transport = ASGITransport(app=manager.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                health_resp = await client.get("/health")
                assert health_resp.status_code == SUCCESS_STATUS
                assert health_resp.json()["status"] == "healthy"

                predict_resp = await client.post(
                    "/predict",
                    json={
                        "model_id": "credit-risk",
                        "model_version": "v1",
                        "entity_id": "customer-good",
                    },
                )
                assert predict_resp.status_code == SUCCESS_STATUS
                data = predict_resp.json()
                assert "prediction_id" in data
                assert data["model_id"] == "credit-risk"

                await asyncio.sleep(1)

                feedback_resp = await client.post(
                    "/feedback",
                    json={
                        "prediction_id": data["prediction_id"],
                        "ground_truth": 1,
                    },
                )
                assert feedback_resp.status_code == SUCCESS_STATUS
                assert feedback_resp.json()["status"] == "feedback_received"
