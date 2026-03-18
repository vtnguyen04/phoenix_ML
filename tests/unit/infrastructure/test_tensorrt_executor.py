from pathlib import Path

import numpy as np
import pytest

from src.domain.inference.entities.model import Model
from src.domain.inference.value_objects.feature_vector import FeatureVector
from src.infrastructure.ml_engines.tensorrt_executor import TensorRTExecutor
from src.shared.utils.model_generator import generate_simple_onnx


@pytest.fixture
def mock_model() -> Model:
    return Model(
        id="test-trt-model",
        version="v1",
        uri="local://test",
        framework="tensorrt",
        metadata={"features": ["f1", "f2", "f3"]},
    )


@pytest.mark.asyncio
async def test_tensorrt_executor_predict(tmp_path: Path, mock_model: Model) -> None:
    # Setup mock Model Artifact
    cache_dir = tmp_path / "model_cache"
    model_dir = cache_dir / mock_model.id / mock_model.version
    model_dir.mkdir(parents=True, exist_ok=True)
    generate_simple_onnx(model_dir / "model.onnx")

    executor = TensorRTExecutor(cache_dir=cache_dir)
    features = FeatureVector(values=np.array([0.1] * 30, dtype=np.float32))

    prediction = await executor.predict(mock_model, features)

    assert prediction.model_id == "test-trt-model"
    assert prediction.model_version == "v1"
    assert "result" in prediction.model_dump()
    assert prediction.confidence.value > 0


@pytest.mark.asyncio
async def test_tensorrt_executor_batch_predict(tmp_path: Path, mock_model: Model) -> None:
    cache_dir = tmp_path / "model_cache"
    model_dir = cache_dir / mock_model.id / mock_model.version
    model_dir.mkdir(parents=True, exist_ok=True)
    generate_simple_onnx(model_dir / "model.onnx")

    executor = TensorRTExecutor(cache_dir=cache_dir)
    features_list = [
        FeatureVector(values=np.array([0.1] * 30, dtype=np.float32)),
        FeatureVector(values=np.array([0.4] * 30, dtype=np.float32)),
    ]

    predictions = await executor.batch_predict(mock_model, features_list)

    assert len(predictions) == 2
    for p in predictions:
        assert p.model_id == "test-trt-model"
        assert p.latency_ms > 0
