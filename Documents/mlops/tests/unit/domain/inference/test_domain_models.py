import numpy as np
import pytest
from pydantic import ValidationError

from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.value_objects.confidence_score import ConfidenceScore
from src.domain.inference.value_objects.feature_vector import FeatureVector


class TestFeatureVector:
    def test_create_valid_vector(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        vec = FeatureVector(values=values)
        assert len(vec.values) == 3  # noqa: PLR2004
        assert vec.to_list() == [1.0, 2.0, 3.0]  # noqa: PLR2004

    def test_create_empty_vector_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            FeatureVector(values=np.array([]))

    def test_equality(self) -> None:
        v1 = FeatureVector(values=np.array([1.0, 2.0]))
        v2 = FeatureVector(values=np.array([1.0, 2.0]))
        assert v1 == v2

class TestConfidenceScore:
    def test_valid_score(self) -> None:
        score = ConfidenceScore(value=0.95)
        assert score.value == 0.95  # noqa: PLR2004

    def test_invalid_score_high(self) -> None:
        with pytest.raises(ValidationError):
            ConfidenceScore(value=1.1)

    def test_invalid_score_low(self) -> None:
        with pytest.raises(ValidationError):
            ConfidenceScore(value=-0.1)

class TestPrediction:
    def test_is_confident(self) -> None:
        pred = Prediction(
            model_id="test_model",
            model_version="v1",
            result="cat",
            confidence=ConfidenceScore(value=0.8),
            latency_ms=10.0
        )
        assert pred.is_confident(0.5) is True
        assert pred.is_confident(0.9) is False
