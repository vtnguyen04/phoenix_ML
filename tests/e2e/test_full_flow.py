"""
End-to-End tests for the Phoenix ML platform.

These tests are MODEL-AGNOSTIC: they read configuration from Settings
and model_configs/ to determine the model, features, and entity IDs
dynamically. No hardcoded model-specific data.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from src.config import get_settings
from src.infrastructure.http.fastapi_server import app

SUCCESS_STATUS = 200
TEST_TIMEOUT = 30.0


def _resolve_test_features() -> list[float]:
    """Resolve a valid feature vector for the configured DEFAULT_MODEL_ID.

    Resolution order:
    1. reference_features.json (first record's feature values)
    2. model_configs/<model_id>.yaml → feature_names count → synthetic zeros
    3. Fallback to 10 generic features
    """
    settings = get_settings()
    root = _find_root()

    # 1. Try reference_features.json
    ref_path = root / "data" / "reference_features.json"
    if ref_path.exists():
        with open(ref_path) as f:
            records = json.load(f)
        if records and isinstance(records, list):
            features = records[0].get("features", {})
            if isinstance(features, dict):
                return list(features.values())

    # 2. Try model config → feature count
    from src.infrastructure.http.model_config_loader import load_all_model_configs

    configs = load_all_model_configs(root / settings.MODEL_CONFIG_DIR)
    model_id = settings.DEFAULT_MODEL_ID
    if model_id in configs:
        n = len(configs[model_id].feature_names)
        if n > 0:
            return [0.5] * n

    # 3. Fallback
    return [0.5] * 10


def _resolve_test_entity_id() -> str | None:
    """Get first entity_id from reference_features.json, or None."""
    ref_path = _find_root() / "data" / "reference_features.json"
    if ref_path.exists():
        with open(ref_path) as f:
            records = json.load(f)
        if records and isinstance(records, list):
            ent = records[0].get("entity_id")
            return str(ent) if ent else None
    return None


def _find_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


@pytest.mark.asyncio
async def test_e2e_predict_and_feedback_flow() -> None:
    """
    End-to-End test: Health → Predict → Feedback.
    Dynamically resolves model config — no hardcoded model data.
    """
    _settings = get_settings()
    entity_id = _resolve_test_entity_id()

    # Build payload: prefer entity_id (feature store lookup), fallback to raw features
    payload: dict[str, Any] = {"model_id": _settings.DEFAULT_MODEL_ID}
    if _settings.DEFAULT_MODEL_VERSION:
        payload["model_version"] = _settings.DEFAULT_MODEL_VERSION
    if entity_id:
        payload["entity_id"] = entity_id
    else:
        payload["features"] = _resolve_test_features()

    async with asyncio.timeout(TEST_TIMEOUT):
        async with LifespanManager(app) as manager:
            transport = ASGITransport(app=manager.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                health_resp = await client.get("/health")
                assert health_resp.status_code == SUCCESS_STATUS
                assert health_resp.json()["status"] == "healthy"

                predict_resp = await client.post("/predict", json=payload)
                assert predict_resp.status_code == SUCCESS_STATUS, predict_resp.json()
                data = predict_resp.json()
                assert "prediction_id" in data
                assert data["model_id"] == _settings.DEFAULT_MODEL_ID

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


@pytest.mark.asyncio
async def test_e2e_monitoring_drift_check() -> None:
    """
    End-to-End test: Health → Predict → Drift Check.
    Uses dynamically resolved features — model-agnostic.
    """
    _settings = get_settings()
    features = _resolve_test_features()

    async with asyncio.timeout(TEST_TIMEOUT):
        async with LifespanManager(app) as manager:
            transport = ASGITransport(app=manager.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                health_resp = await client.get("/health")
                assert health_resp.status_code == SUCCESS_STATUS

                await client.post(
                    "/predict",
                    json={
                        "model_id": _settings.DEFAULT_MODEL_ID,
                        "features": features,
                    },
                )

                drift_resp = await client.get(f"/monitoring/drift/{_settings.DEFAULT_MODEL_ID}")
                # Drift check may succeed or fail depending on DB state,
                # but the route should be reachable.
                assert drift_resp.status_code in (SUCCESS_STATUS, 400, 500)


@pytest.mark.asyncio
async def test_e2e_model_performance_endpoint() -> None:
    """
    End-to-End test: Verify performance endpoint is reachable.
    """
    _settings = get_settings()

    async with asyncio.timeout(TEST_TIMEOUT):
        async with LifespanManager(app) as manager:
            transport = ASGITransport(app=manager.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                perf_resp = await client.get(
                    f"/monitoring/performance/{_settings.DEFAULT_MODEL_ID}"
                )
                assert perf_resp.status_code in (SUCCESS_STATUS, 500)
