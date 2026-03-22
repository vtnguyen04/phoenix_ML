"""Tests for shared exception hierarchy."""

from phoenix_ml.shared.exceptions import (
    FeatureStoreError,
    InferenceError,
    ModelNotFoundError,
    PhoenixBaseError,
    ValidationError,
)


def test_phoenix_base_error_is_exception() -> None:
    err = PhoenixBaseError("base error")
    assert str(err) == "base error"
    assert isinstance(err, Exception)


def test_model_not_found_carries_model_id() -> None:
    err = ModelNotFoundError("credit-risk")
    assert "credit-risk" in str(err)
    assert isinstance(err, PhoenixBaseError)


def test_inference_error() -> None:
    err = InferenceError("engine failed")
    assert "engine failed" in str(err)
    assert isinstance(err, PhoenixBaseError)


def test_feature_store_error() -> None:
    err = FeatureStoreError("redis down")
    assert "redis down" in str(err)
    assert isinstance(err, PhoenixBaseError)


def test_validation_error() -> None:
    err = ValidationError("invalid input")
    assert "invalid input" in str(err)
    assert isinstance(err, PhoenixBaseError)
