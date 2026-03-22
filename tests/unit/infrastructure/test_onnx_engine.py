"""Tests for ONNXInferenceEngine with mocked onnxruntime."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from phoenix_ml.domain.inference.entities.model import Model
from phoenix_ml.domain.inference.value_objects.feature_vector import FeatureVector
from phoenix_ml.infrastructure.ml_engines.onnx_engine import ONNXInferenceEngine


@pytest.fixture
def engine(tmp_path: Path) -> ONNXInferenceEngine:
    return ONNXInferenceEngine(cache_dir=tmp_path)


@pytest.fixture
def model() -> Model:
    return Model(
        id="test-model",
        version="v1",
        uri="file:///test",
        framework="onnx",
        metadata={"role": "champion"},
        is_active=True,
    )


async def test_load_file_not_found(engine: ONNXInferenceEngine, model: Model) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        await engine.load(model)


async def test_load_creates_session(
    engine: ONNXInferenceEngine, model: Model, tmp_path: Path
) -> None:
    model_dir = tmp_path / model.id / model.version
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"fake-onnx")

    with patch("src.infrastructure.ml_engines.onnx_engine.ort") as mock_ort:
        mock_session = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session

        await engine.load(model)
        assert model.unique_key in engine._sessions


async def test_load_skips_if_already_loaded(engine: ONNXInferenceEngine, model: Model) -> None:
    engine._sessions[model.unique_key] = MagicMock()
    await engine.load(model)  # should not raise


async def test_batch_predict_classification(engine: ONNXInferenceEngine, model: Model) -> None:
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]

    # Simulate sklearn ONNX output: [labels, [prob_dicts]]
    mock_session.run.return_value = [
        np.array([1, 0]),
        [{0: 0.2, 1: 0.8}, {0: 0.9, 1: 0.1}],
    ]
    engine._sessions[model.unique_key] = mock_session

    features2 = [
        FeatureVector(values=np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        FeatureVector(values=np.array([4.0, 5.0, 6.0], dtype=np.float32)),
    ]
    preds = await engine.batch_predict(model, features2)
    assert len(preds) == 2
    assert preds[0].result == 1
    assert preds[0].confidence.value == pytest.approx(0.8)


async def test_batch_predict_regression(engine: ONNXInferenceEngine, model: Model) -> None:
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_session.run.return_value = [np.array([[150000.0]], dtype=np.float32)]
    engine._sessions[model.unique_key] = mock_session

    features = [FeatureVector(values=np.array([1.0, 2.0], dtype=np.float32))]
    preds = await engine.batch_predict(model, features)
    assert len(preds) == 1
    assert preds[0].result == pytest.approx(150000.0)


async def test_predict_delegates_to_batch(engine: ONNXInferenceEngine, model: Model) -> None:
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_session.run.return_value = [np.array([[0.5]], dtype=np.float32)]
    engine._sessions[model.unique_key] = mock_session

    fv = FeatureVector(values=np.array([1.0], dtype=np.float32))
    pred = await engine.predict(model, fv)
    assert pred.model_id == "test-model"


async def test_optimize_is_noop(engine: ONNXInferenceEngine, model: Model) -> None:
    await engine.optimize(model)  # should not raise
