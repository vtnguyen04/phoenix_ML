"""Tests for input validator."""


from src.application.commands.predict_command import PredictCommand
from src.infrastructure.http.middleware.input_validator import (
    validate_prediction_input,
)


class TestInputValidator:
    def test_valid_input(self) -> None:
        cmd = PredictCommand(model_id="test", features=[1.0, 2.0, 3.0])
        errors = validate_prediction_input(cmd)
        assert errors == []

    def test_empty_features(self) -> None:
        cmd = PredictCommand(model_id="test", features=[])
        errors = validate_prediction_input(cmd)
        assert any("empty" in e.lower() for e in errors)

    def test_feature_count_mismatch(self) -> None:
        cmd = PredictCommand(model_id="test", features=[1.0, 2.0])
        config = {"feature_count": 3}
        errors = validate_prediction_input(cmd, model_config=config)
        assert any("mismatch" in e.lower() for e in errors)

    def test_feature_count_matches(self) -> None:
        cmd = PredictCommand(model_id="test", features=[1.0, 2.0, 3.0])
        config = {"feature_count": 3}
        errors = validate_prediction_input(cmd, model_config=config)
        assert errors == []

    def test_nan_detection(self) -> None:
        cmd = PredictCommand(
            model_id="test", features=[1.0, float("nan"), 3.0]
        )
        errors = validate_prediction_input(cmd)
        assert any("NaN" in e for e in errors)

    def test_inf_detection(self) -> None:
        cmd = PredictCommand(
            model_id="test", features=[1.0, float("inf"), 3.0]
        )
        errors = validate_prediction_input(cmd)
        assert any("infinite" in e for e in errors)

    def test_range_validation(self) -> None:
        cmd = PredictCommand(model_id="test", features=[150.0, 50000.0])
        config = {"feature_ranges": {"age": (0, 120), "income": (0, 1e6)}}
        errors = validate_prediction_input(cmd, model_config=config)
        assert any("out of range" in e.lower() for e in errors)

    def test_schema_names_mismatch(self) -> None:
        cmd = PredictCommand(model_id="test", features=[1.0, 2.0])
        config = {"feature_names": ["a", "b", "c"]}
        errors = validate_prediction_input(cmd, model_config=config)
        assert any("schema" in e.lower() for e in errors)
