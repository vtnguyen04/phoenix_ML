"""Tests for prediction cache."""

import time

import pytest

from src.infrastructure.cache.prediction_cache import PredictionCache


@pytest.fixture
def cache() -> PredictionCache:
    return PredictionCache(default_ttl_seconds=2, max_size=100)


class TestPredictionCache:
    def test_miss_returns_none(self, cache: PredictionCache) -> None:
        result = cache.get("model-a", "v1", [1.0, 2.0, 3.0])
        assert result is None

    def test_set_and_get_hit(self, cache: PredictionCache) -> None:
        features = [0.5, 1.2, 0.8]
        value = {"result": 1, "confidence": 0.95, "model_id": "m1"}
        cache.set("m1", "v1", features, value)

        hit = cache.get("m1", "v1", features)
        assert hit is not None
        assert hit["result"] == 1
        assert hit["confidence"] == 0.95

    def test_different_features_miss(self, cache: PredictionCache) -> None:
        cache.set("m1", "v1", [1.0, 2.0], {"result": 0, "model_id": "m1"})
        assert cache.get("m1", "v1", [1.0, 3.0]) is None

    def test_different_model_miss(self, cache: PredictionCache) -> None:
        cache.set("m1", "v1", [1.0], {"result": 0, "model_id": "m1"})
        assert cache.get("m2", "v1", [1.0]) is None

    def test_different_version_miss(self, cache: PredictionCache) -> None:
        cache.set("m1", "v1", [1.0], {"result": 0, "model_id": "m1"})
        assert cache.get("m1", "v2", [1.0]) is None

    def test_ttl_expiry(self) -> None:
        short_cache = PredictionCache(default_ttl_seconds=0, max_size=100)
        short_cache.set("m1", "v1", [1.0], {"result": 1, "model_id": "m1"})
        time.sleep(0.05)
        assert short_cache.get("m1", "v1", [1.0]) is None

    def test_invalidate_model(self, cache: PredictionCache) -> None:
        cache.set("m1", "v1", [1.0], {"result": 0, "model_id": "m1"})
        cache.set("m1", "v1", [2.0], {"result": 1, "model_id": "m1"})
        cache.set("m2", "v1", [1.0], {"result": 0, "model_id": "m2"})

        removed = cache.invalidate_model("m1")
        assert removed == 2
        assert cache.get("m1", "v1", [1.0]) is None
        assert cache.get("m2", "v1", [1.0]) is not None

    def test_clear(self, cache: PredictionCache) -> None:
        cache.set("m1", "v1", [1.0], {"result": 0, "model_id": "m1"})
        cache.set("m2", "v1", [2.0], {"result": 1, "model_id": "m2"})
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0

    def test_max_size_eviction(self) -> None:
        small_cache = PredictionCache(default_ttl_seconds=600, max_size=5)
        for i in range(10):
            small_cache.set("m", "v1", [float(i)], {"result": i, "model_id": "m"})
        # Should have evicted oldest entries
        assert small_cache.size <= 5

    def test_deterministic_key(self, cache: PredictionCache) -> None:
        """Same inputs always produce the same key."""
        key1 = PredictionCache._make_key("m1", "v1", [1.0, 2.0])
        key2 = PredictionCache._make_key("m1", "v1", [1.0, 2.0])
        assert key1 == key2

    def test_feature_order_matters(self, cache: PredictionCache) -> None:
        """Different feature order = different key."""
        key1 = PredictionCache._make_key("m1", "v1", [1.0, 2.0])
        key2 = PredictionCache._make_key("m1", "v1", [2.0, 1.0])
        assert key1 != key2
