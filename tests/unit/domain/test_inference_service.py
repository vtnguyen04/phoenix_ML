from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.domain.feature_store.repositories.feature_store import FeatureStore
from src.domain.inference.entities.model import Model
from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.services.batch_manager import BatchManager
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.inference.services.inference_service import (
    InferenceService,
    PredictionRequest,
)
from src.domain.inference.services.routing_strategy import RoutingStrategy
from src.domain.inference.value_objects.confidence_score import ConfidenceScore
from src.domain.model_registry.repositories.artifact_storage import ArtifactStorage
from src.domain.model_registry.repositories.model_repository import ModelRepository


@pytest.fixture
def mock_components() -> dict[str, Any]:
    return {
        "repo": AsyncMock(spec=ModelRepository),
        "engine": AsyncMock(spec=InferenceEngine),
        "batch": AsyncMock(spec=BatchManager),
        "fs": AsyncMock(spec=FeatureStore),
        "storage": AsyncMock(spec=ArtifactStorage),
        "routing": Mock(spec=RoutingStrategy),
    }


@pytest.fixture
def inference_service(mock_components: dict[str, Any]) -> InferenceService:
    return InferenceService(
        model_repo=mock_components["repo"],
        inference_engine=mock_components["engine"],
        batch_manager=mock_components["batch"],
        feature_store=mock_components["fs"],
        artifact_storage=mock_components["storage"],
        routing_strategy=mock_components["routing"],
        cache_dir=Path("/tmp/test_cache"),
    )


@pytest.mark.asyncio
async def test_inference_service_full_flow(
    inference_service: InferenceService, mock_components: dict[str, Any]
) -> None:
    # Setup
    model = Model(id="m1", version="v1", uri="loc://v1", framework="onnx")
    mock_components["repo"].get_by_id.return_value = model

    # Mock prediction
    expected_pred = Prediction(
        model_id="m1",
        model_version="v1",
        result=1,
        confidence=ConfidenceScore(value=0.95),
        latency_ms=10.0,
    )
    mock_components["batch"].predict.return_value = expected_pred

    # Execute with Path.exists mocked to False to force download
    with patch("pathlib.Path.exists", return_value=False):
        req = PredictionRequest(
            model_id="m1", model_version="v1", features=[1.0, 2.0, 3.0, 4.0]
        )
        prediction = await inference_service.predict(req)

    # Verify
    assert prediction == expected_pred
    mock_components["storage"].download.assert_called_once()
    mock_components["engine"].load.assert_awaited_once_with(model)
    mock_components["batch"].predict.assert_called_once()


@pytest.mark.asyncio
async def test_inference_service_with_feature_lookup(
    inference_service: InferenceService, mock_components: dict[str, Any]
) -> None:
    # Setup
    model = Model(id="m1", version="v1", uri="loc://v1", framework="onnx")
    mock_components["repo"].get_by_id.return_value = model
    mock_components["fs"].get_online_features.return_value = [1.0, 2.0, 3.0, 4.0]

    # Execute (No features provided)
    req = PredictionRequest(model_id="m1", model_version="v1", entity_id="user-123")

    with patch("pathlib.Path.exists", return_value=True):  # Skip download
        await inference_service.predict(req)

    # Verify
    mock_components["fs"].get_online_features.assert_awaited_once_with(
        "user-123", ["f1", "f2", "f3", "f4"]
    )


@pytest.mark.asyncio
async def test_inference_service_uses_model_metadata_features(
    inference_service: InferenceService, mock_components: dict[str, Any]
) -> None:
    model = Model(
        id="m1",
        version="v1",
        uri="loc://v1",
        framework="onnx",
        metadata={"features": ["income", "debt", "age", "credit_history"]},
    )
    mock_components["repo"].get_by_id.return_value = model
    mock_components["fs"].get_online_features.return_value = [2.0, -1.5, 1.0, 1.5]

    req = PredictionRequest(model_id="m1", model_version="v1", entity_id="user-123")

    with patch("pathlib.Path.exists", return_value=True):
        await inference_service.predict(req)

    mock_components["fs"].get_online_features.assert_awaited_once_with(
        "user-123", ["income", "debt", "age", "credit_history"]
    )
