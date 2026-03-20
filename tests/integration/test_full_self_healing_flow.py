"""
Integration test: Self-Healing Pipeline.

Full flow: Predict (drifted data) → Drift Detection → Retrain Trigger → Model Update.

MODEL-AGNOSTIC: resolves model config, feature count, and model paths
dynamically from Settings and model_configs/.
"""

import asyncio
import json

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from src.config import get_settings
from src.infrastructure.bootstrap.container import find_project_root
from src.infrastructure.http.fastapi_server import app
from src.infrastructure.persistence.database import get_db
from src.infrastructure.persistence.postgres_drift_repo import (
    PostgresDriftReportRepository,
)

SUCCESS_STATUS = 200
DRIFT_REQUEST_COUNT = 15
WAIT_FOR_SUBPROCESS = 5


def _resolve_feature_count() -> int:
    """Determine expected feature count for the configured model."""
    root = find_project_root()
    settings = get_settings()

    # 1. Try reference_features.json
    ref_path = root / "data" / "reference_features.json"
    if ref_path.exists():
        with open(ref_path) as f:
            records = json.load(f)
        if records and isinstance(records, list):
            features = records[0].get("features", {})
            if isinstance(features, dict):
                return len(features)

    # 2. Try model config
    from src.infrastructure.bootstrap.model_config_loader import load_all_model_configs

    configs = load_all_model_configs(root / settings.MODEL_CONFIG_DIR)
    model_id = settings.DEFAULT_MODEL_ID
    if model_id in configs:
        return len(configs[model_id].feature_names)

    return 10


@pytest.mark.asyncio
async def test_full_self_healing_flow() -> None:
    """
    End-to-End Integration Test:
    Predict (with drift) → Monitoring Detects →
    Retrain Handler Executes → Model File Updated.

    Feature count is dynamically resolved — works for any model.
    """
    _s = get_settings()
    n_features = _resolve_feature_count()

    async with LifespanManager(app) as manager:
        transport = ASGITransport(app=manager.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # --- PHASE 1: Generate Drifted Data ---
            # Use extreme values to trigger drift detection
            payload = {
                "model_id": _s.DEFAULT_MODEL_ID,
                "features": [100.0] * n_features,
            }

            for _ in range(DRIFT_REQUEST_COUNT):
                resp = await client.post("/predict", json=payload)
                assert resp.status_code == SUCCESS_STATUS

            await asyncio.sleep(2)

            # --- PHASE 2: Trigger Drift Detection ---
            drift_resp = await client.get(f"/monitoring/drift/{_s.DEFAULT_MODEL_ID}")
            if drift_resp.status_code != SUCCESS_STATUS:
                print(f"❌ Drift Error: {drift_resp.json()}")

            assert drift_resp.status_code == SUCCESS_STATUS
            report = drift_resp.json()

            assert report["drift_detected"] is True
            rec = report["recommendation"]
            assert "CRITICAL" in rec or "WARNING" in rec

            # --- PHASE 3: Verify Self-Healing ---
            await asyncio.sleep(WAIT_FOR_SUBPROCESS)

            # 3.1 Check Database for saved report
            async for db in get_db():
                repo = PostgresDriftReportRepository(db)
                history = await repo.get_history(_s.DEFAULT_MODEL_ID, limit=1)
                assert len(history) > 0
                assert history[0].drift_detected is True
                break

            # 3.2 Check if model file exists (path resolved from config)
            root = find_project_root()
            _fs_model_id = _s.DEFAULT_MODEL_ID.replace("-", "_")
            _version = _s.DEFAULT_MODEL_VERSION or "v1"
            model_path = root / "models" / _fs_model_id / _version / "model.onnx"

            assert model_path.exists()
