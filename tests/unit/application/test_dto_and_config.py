"""Tests for DTOs and config."""

from phoenix_ml.application.dto import PredictionRequestDTO, PredictionResponseDTO
from phoenix_ml.config import Settings


class TestPredictionRequestDTO:
    def test_minimal_request(self) -> None:
        req = PredictionRequestDTO(
            model_id="credit-risk",
            model_version=None,
            entity_id=None,
            features=None,
        )
        assert req.model_id == "credit-risk"
        assert req.model_version is None
        assert req.entity_id is None
        assert req.features is None

    def test_full_request(self) -> None:
        req = PredictionRequestDTO(
            model_id="house-price",
            model_version="v2",
            entity_id="user-1",
            features=[1.0, 2.0, 3.0],
        )
        assert req.model_version == "v2"
        assert req.features == [1.0, 2.0, 3.0]


class TestPredictionResponseDTO:
    def test_response_fields(self) -> None:
        resp = PredictionResponseDTO(
            prediction_id="abc",
            model_id="credit-risk",
            version="v1",
            result=[0.85],
            confidence=0.85,
            latency_ms=5.2,
        )
        assert resp.prediction_id == "abc"
        assert resp.result == [0.85]
        assert resp.confidence == 0.85


class TestSettings:
    def test_has_expected_attributes(self) -> None:
        s = Settings()
        assert isinstance(s.APP_NAME, str)
        assert isinstance(s.APP_VERSION, str)
        assert isinstance(s.DEBUG, bool)
        assert isinstance(s.USE_REDIS, bool)
        assert isinstance(s.REDIS_URL, str)
        assert isinstance(s.KAFKA_URL, str)
        assert isinstance(s.DATABASE_URL, str)
        assert isinstance(s.JAEGER_ENDPOINT, str)
        assert isinstance(s.MLFLOW_TRACKING_URI, str)
        assert isinstance(s.DEFAULT_MODEL_VERSION, str)
        assert isinstance(s.MODEL_CONFIG_DIR, str)
