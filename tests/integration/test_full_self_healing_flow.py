import asyncio

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from src.infrastructure.http.fastapi_server import app, find_project_root
from src.infrastructure.persistence.database import get_db
from src.infrastructure.persistence.postgres_drift_repo import (
    PostgresDriftReportRepository,
)

# Test Constants
SUCCESS_STATUS = 200
DRIFT_REQUEST_COUNT = 5
WAIT_FOR_SUBPROCESS = 5


@pytest.mark.asyncio
async def test_full_self_healing_flow() -> None:
    """
    End-to-End Integration Test:
    Predict (with drift) -> Monitoring Detects ->
    Retrain Handler Executes -> Model File Updated.
    """
    # 1. Start App
    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # --- PHASE 1: Generate Drifted Data ---
            payload = {
                "model_id": "credit-risk",
                "features": [100.0, 100.0, 100.0, 100.0],
            }

            for _ in range(DRIFT_REQUEST_COUNT):
                resp = await client.post("/predict", json=payload)
                assert resp.status_code == SUCCESS_STATUS

            # --- PHASE 2: Trigger Drift Detection ---
            # Triggers check_drift -> _trigger_retrain -> RetrainHandler
            drift_resp = await client.get("/monitoring/drift/credit-risk")
            assert drift_resp.status_code == SUCCESS_STATUS
            report = drift_resp.json()

            assert report["drift_detected"] is True
            rec = report["recommendation"]
            assert "CRITICAL" in rec or "WARNING" in rec

            # --- PHASE 3: Verify Self-Healing ---
            # Give the subprocess a few seconds to complete training
            await asyncio.sleep(WAIT_FOR_SUBPROCESS)

            # 3.1 Check Database for saved report
            async for db in get_db():
                repo = PostgresDriftReportRepository(db)
                history = await repo.get_history("credit-risk", limit=1)
                assert len(history) > 0
                assert history[0].drift_detected is True
                break

            # 3.2 Check if model file exists
            root = find_project_root()
            model_path = root / "models" / "credit_risk" / "v1" / "model.onnx"

            assert model_path.exists()
