"""Tests for PluginRegistry and processor plugins."""

import asyncio
from typing import Any

import pytest

from phoenix_ml.domain.inference.services.processor_plugin import (
    ClassificationPostprocessor,
    IPostprocessor,
    IPreprocessor,
    PassthroughPreprocessor,
)
from phoenix_ml.domain.shared.plugin_registry import PluginRegistry
from phoenix_ml.domain.training.services.trainer_plugin import ITrainer, TrainResult

# ─── Mock Plugins ────────────────────────────────────────────────


class MockTrainer(ITrainer):
    """Test trainer that returns fixed results."""

    async def train(self, config: dict[str, Any]) -> TrainResult:
        return TrainResult(
            model_path="/tmp/model.onnx",
            metrics={"accuracy": 0.99},
        )

    async def validate(self, model_path: str, data_path: str) -> dict[str, float]:
        return {"accuracy": 0.99}


class MockPreprocessor(IPreprocessor):
    async def preprocess(
        self, raw_input: dict[str, Any], model_config: dict[str, Any]
    ) -> list[float]:
        return [1.0, 2.0, 3.0]


class MockPostprocessor(IPostprocessor):
    async def postprocess(
        self, model_output: list[float], model_config: dict[str, Any]
    ) -> dict[str, Any]:
        return {"result": "detected", "boxes": 5}


# ─── PluginRegistry ──────────────────────────────────────────────


class TestPluginRegistry:
    def test_empty_registry(self) -> None:
        registry = PluginRegistry()
        assert registry.registered_models == []

    def test_register_and_get_trainer(self) -> None:
        registry = PluginRegistry()
        trainer = MockTrainer()
        registry.register_trainer("yolo-detect", trainer)

        assert registry.has_trainer("yolo-detect")
        assert registry.get_trainer("yolo-detect") is trainer
        assert not registry.has_trainer("nonexistent")

    def test_default_trainer_raises(self) -> None:
        registry = PluginRegistry()
        trainer = registry.get_trainer("nonexistent")
        with pytest.raises(NotImplementedError, match="No trainer registered"):
            asyncio.get_event_loop().run_until_complete(trainer.train({}))

    def test_register_preprocessor(self) -> None:
        registry = PluginRegistry()
        preprocessor = MockPreprocessor()
        registry.register_preprocessor("my-model", preprocessor)
        assert registry.get_preprocessor("my-model") is preprocessor

    def test_default_preprocessor(self) -> None:
        registry = PluginRegistry()
        preprocessor = registry.get_preprocessor("unregistered")
        assert isinstance(preprocessor, PassthroughPreprocessor)

    def test_register_postprocessor(self) -> None:
        registry = PluginRegistry()
        postprocessor = MockPostprocessor()
        registry.register_postprocessor("my-model", postprocessor)
        assert registry.get_postprocessor("my-model") is postprocessor

    def test_default_postprocessor(self) -> None:
        registry = PluginRegistry()
        postprocessor = registry.get_postprocessor("unregistered")
        assert isinstance(postprocessor, ClassificationPostprocessor)

    def test_registered_models(self) -> None:
        registry = PluginRegistry()
        registry.register_trainer("model-a", MockTrainer())
        registry.register_preprocessor("model-b", MockPreprocessor())
        registry.register_postprocessor("model-a", MockPostprocessor())

        models = registry.registered_models
        assert "model-a" in models
        assert "model-b" in models

    def test_summary(self) -> None:
        registry = PluginRegistry()
        registry.register_trainer("test", MockTrainer())
        summary = registry.summary()
        assert "test" in summary
        assert summary["test"]["trainer"] == "MockTrainer"


# ─── Processor Plugins ──────────────────────────────────────────


class TestPassthroughPreprocessor:
    @pytest.mark.asyncio
    async def test_passthrough(self) -> None:
        preprocessor = PassthroughPreprocessor()
        result = await preprocessor.preprocess({"features": [1.0, 2.5, 3.0]}, {})
        assert result == [1.0, 2.5, 3.0]

    @pytest.mark.asyncio
    async def test_passthrough_empty(self) -> None:
        preprocessor = PassthroughPreprocessor()
        result = await preprocessor.preprocess({}, {})
        assert result == []


class TestClassificationPostprocessor:
    @pytest.mark.asyncio
    async def test_binary_classification(self) -> None:
        postprocessor = ClassificationPostprocessor()
        result = await postprocessor.postprocess([0.8], {})
        assert result["prediction"] == 1
        assert result["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_binary_negative(self) -> None:
        postprocessor = ClassificationPostprocessor()
        result = await postprocessor.postprocess([0.3], {})
        assert result["prediction"] == 0
        assert result["confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_multiclass(self) -> None:
        postprocessor = ClassificationPostprocessor()
        result = await postprocessor.postprocess(
            [0.1, 0.7, 0.2],
            {"class_labels": ["cat", "dog", "bird"]},
        )
        assert result["prediction"] == 1
        assert result["label"] == "dog"
        assert result["confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_multiclass_no_labels(self) -> None:
        postprocessor = ClassificationPostprocessor()
        result = await postprocessor.postprocess([0.1, 0.9], {})
        assert result["prediction"] == 1
        assert result["label"] == "1"
