"""
Integration test for dynamic batching via /predict endpoint.

MODEL-AGNOSTIC: resolves features from model config or reference data.
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
CONCURRENT_REQUESTS = 3
TEST_TIMEOUT = 30.0


def _find_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _resolve_test_features() -> list[float]:
    """Resolve a valid feature vector for the configured model."""
    root = _find_root()
    settings = get_settings()

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
    from src.infrastructure.bootstrap.model_config_loader import load_all_model_configs

    configs = load_all_model_configs(root / settings.MODEL_CONFIG_DIR)
    model_id = settings.DEFAULT_MODEL_ID
    if model_id in configs:
        n = len(configs[model_id].feature_names)
        if n > 0:
            return [0.5] * n

    return [0.5] * 10


def _resolve_entity_id() -> str | None:
    """Get first entity_id from reference data if available."""
    ref_path = _find_root() / "data" / "reference_features.json"
    if ref_path.exists():
        with open(ref_path) as f:
            records = json.load(f)
        if records and isinstance(records, list):
            ent = records[0].get("entity_id")
            return str(ent) if ent else None
    return None


@pytest.mark.asyncio
async def test_api_predict_dynamic_batching() -> None:
    """
    Integration test for the FastAPI /predict endpoint.
    Uses LifespanManager to ensure DB seeding and component startup happens.
    Dynamically resolves model config — no hardcoded model data.
    """
    _settings = get_settings()
    entity_id = _resolve_entity_id()

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
                # Simulating concurrent requests to trigger BatchManager
                tasks = [client.post("/predict", json=payload) for _ in range(CONCURRENT_REQUESTS)]
                responses = await asyncio.gather(*tasks)

                assert len(responses) == CONCURRENT_REQUESTS
                for response in responses:
                    if response.status_code != SUCCESS_STATUS:
                        print(f"❌ Error: {response.json()}")
                    assert response.status_code == SUCCESS_STATUS
                    data = response.json()
                    assert data["model_id"] == _settings.DEFAULT_MODEL_ID
                    assert "result" in data


@pytest.mark.asyncio
async def test_api_health() -> None:
    """Basic health check integration test."""
    async with asyncio.timeout(TEST_TIMEOUT):
        async with LifespanManager(app) as manager:
            transport = ASGITransport(app=manager.app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")
                assert response.status_code == SUCCESS_STATUS
                assert response.json()["status"] == "healthy"
