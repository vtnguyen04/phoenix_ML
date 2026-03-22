"""Tests for domain events and value objects at 0% coverage."""

import pytest

from phoenix_ml.domain.inference.events.model_loaded import ModelLoaded
from phoenix_ml.domain.inference.events.prediction_made import PredictionMade
from phoenix_ml.domain.inference.value_objects.latency_budget import LatencyBudget


class TestLatencyBudget:
    def test_valid_budget(self) -> None:
        budget = LatencyBudget(max_latency_ms=50.0)
        assert budget.max_latency_ms == 50.0

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            LatencyBudget(max_latency_ms=0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            LatencyBudget(max_latency_ms=-10)

    def test_is_exceeded_by(self) -> None:
        budget = LatencyBudget(max_latency_ms=20.0)
        assert budget.is_exceeded_by(25.0) is True
        assert budget.is_exceeded_by(15.0) is False
        assert budget.is_exceeded_by(20.0) is False

    def test_frozen(self) -> None:
        budget = LatencyBudget(max_latency_ms=10.0)
        with pytest.raises(AttributeError):
            budget.max_latency_ms = 99.0  # type: ignore[misc]


class TestDomainEvents:
    def test_prediction_made_is_frozen(self) -> None:
        event = PredictionMade(
            prediction_id="p1",
            model_id="credit-risk",
            model_version="v1",
            entity_id=None,
            features=None,
            result=1,
            confidence=0.9,
            latency_ms=5.0,
        )
        assert event.prediction_id == "p1"
        assert event.confidence == 0.9
        with pytest.raises(AttributeError):
            event.confidence = 0.5  # type: ignore[misc]

    def test_model_loaded_is_frozen(self) -> None:
        event = ModelLoaded(model_id="m1", model_version="v1", framework="onnx")
        assert event.model_id == "m1"
        assert event.framework == "onnx"
        with pytest.raises(AttributeError):
            event.model_id = "m2"  # type: ignore[misc]
