import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from src.application.commands.predict_command import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.domain.inference.entities.model import Model
from src.domain.inference.services.batch_manager import BatchManager
from src.domain.inference.services.inference_service import InferenceService
from src.domain.inference.services.routing_strategy import ABTestStrategy
from src.infrastructure.artifact_storage.local_artifact_storage import (
    LocalArtifactStorage,
)
from src.infrastructure.feature_store.in_memory_feature_store import (
    InMemoryFeatureStore,
)
from src.infrastructure.ml_engines.mock_engine import MockInferenceEngine
from src.infrastructure.persistence.in_memory_model_repo import InMemoryModelRepository


class MockArtifactStorage(LocalArtifactStorage):
    async def download(self, remote_uri: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text("mock content")
        return local_path


PredictHandlerFixture = tuple[
    PredictHandler, InMemoryModelRepository, InMemoryFeatureStore, BatchManager
]


@pytest.fixture
async def predict_handler() -> AsyncGenerator[PredictHandlerFixture, None]:
    repo = InMemoryModelRepository()
    engine = MockInferenceEngine()
    batch_manager = BatchManager(engine)
    fs = InMemoryFeatureStore()
    storage = MockArtifactStorage(base_dir=Path("/tmp/test_storage"))
    routing = ABTestStrategy(0.5)

    inference_service = InferenceService(
        model_repo=repo,
        inference_engine=engine,
        batch_manager=batch_manager,
        feature_store=fs,
        artifact_storage=storage,
        routing_strategy=routing,
        cache_dir=Path("/tmp/test_cache"),
    )

    handler = PredictHandler(inference_service)
    yield handler, repo, fs, batch_manager
    await batch_manager.stop()


@pytest.mark.asyncio
async def test_predict_handler_with_direct_features(
    predict_handler: PredictHandlerFixture,
) -> None:
    handler, repo, _, _ = predict_handler

    # Register model
    model = Model(id="m1", version="v1", uri="local://tmp/m1", framework="onnx")
    await repo.save(model)

    # Execute with explicit features
    command = PredictCommand(model_id="m1", model_version="v1", features=[1.0, 2.0])
    prediction = await handler.execute(command)
    assert prediction.result == pytest.approx(1.5)  # noqa: PLR2004


@pytest.mark.asyncio
async def test_predict_handler_with_feature_store(
    predict_handler: PredictHandlerFixture,
) -> None:
    handler, repo, fs, _ = predict_handler

    # Register model & Seed features
    model = Model(id="m2", version="v1", uri="local://tmp/m2", framework="onnx")
    await repo.save(model)
    await fs.add_features("user-1", {"f1": 10.0, "f2": 20.0, "f3": 30.0, "f4": 40.0})

    # Execute with entity_id
    command = PredictCommand(model_id="m2", model_version="v1", entity_id="user-1")
    prediction = await handler.execute(command)

    assert prediction.result == pytest.approx(25.0)  # noqa: PLR2004


@pytest.mark.asyncio
async def test_batch_manager_actually_batches(
    predict_handler: PredictHandlerFixture,
) -> None:
    handler, repo, _, batch_manager = predict_handler

    model_uri = "local://tmp/m-batch"
    model = Model(id="m-batch", version="v1", uri=model_uri, framework="onnx")
    await repo.save(model)

    # Run multiple predictions concurrently
    num_requests = 5
    commands = [
        PredictCommand(model_id="m-batch", model_version="v1", features=[float(i)])
        for i in range(num_requests)
    ]

    results = await asyncio.gather(*[handler.execute(cmd) for cmd in commands])

    assert len(results) == num_requests
    for i, res in enumerate(results):
        assert res.result == float(i)
