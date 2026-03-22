"""Tests for ExplainabilityService."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import numpy as np
import pytest

from phoenix_ml.domain.monitoring.services.explainability_service import ExplainabilityService


@dataclass
class _FakeConfidence:
    value: float = 0.9


@dataclass
class _FakePrediction:
    result: float = 1.0
    confidence: _FakeConfidence = field(default_factory=_FakeConfidence)


@pytest.fixture
def service() -> ExplainabilityService:
    return ExplainabilityService()


@pytest.fixture
def mock_engine() -> AsyncMock:
    engine = AsyncMock()

    async def _predict(model, features):  # type: ignore[no-untyped-def]
        # Simulate: output depends on features[0] heavily, features[1] less
        vals = features.values
        total = float(vals[0]) * 2.0 + float(vals[1]) * 0.5
        return _FakePrediction(result=total, confidence=_FakeConfidence(0.9))

    engine.predict = _predict
    return engine


@pytest.fixture
def mock_model() -> object:
    return type("Model", (), {"id": "test-model", "version": "v1"})()


class TestExplainabilityService:
    async def test_explain_returns_importances(
        self,
        service: ExplainabilityService,
        mock_engine: AsyncMock,
        mock_model: object,
    ) -> None:
        features = np.array([1.0, 1.0], dtype=np.float32)
        result = await service.explain(
            engine=mock_engine,
            model=mock_model,
            features=features,
            feature_names=["income", "age"],
        )
        assert "importances" in result
        assert "income" in result["importances"]
        assert "age" in result["importances"]
        assert result["method"] == "perturbation"

    async def test_top_features_returned(
        self,
        service: ExplainabilityService,
        mock_engine: AsyncMock,
        mock_model: object,
    ) -> None:
        features = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        result = await service.explain(
            engine=mock_engine,
            model=mock_model,
            features=features,
        )
        assert "top_features" in result
        assert len(result["top_features"]) <= 5

    async def test_importances_sum_to_one(
        self,
        service: ExplainabilityService,
        mock_engine: AsyncMock,
        mock_model: object,
    ) -> None:
        features = np.array([1.0, 2.0], dtype=np.float32)
        result = await service.explain(
            engine=mock_engine,
            model=mock_model,
            features=features,
        )
        total = sum(result["importances"].values())
        assert abs(total - 1.0) < 0.01  # Normalized to ~1.0

    async def test_default_feature_names(
        self,
        service: ExplainabilityService,
        mock_engine: AsyncMock,
        mock_model: object,
    ) -> None:
        features = np.array([1.0, 2.0], dtype=np.float32)
        result = await service.explain(
            engine=mock_engine,
            model=mock_model,
            features=features,
        )
        assert "feature_0" in result["importances"]
        assert "feature_1" in result["importances"]

    async def test_prediction_value_returned(
        self,
        service: ExplainabilityService,
        mock_engine: AsyncMock,
        mock_model: object,
    ) -> None:
        features = np.array([1.0, 1.0], dtype=np.float32)
        result = await service.explain(
            engine=mock_engine,
            model=mock_model,
            features=features,
        )
        assert "prediction" in result
        assert "confidence" in result
        assert isinstance(result["prediction"], float)
