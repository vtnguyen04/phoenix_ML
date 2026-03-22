"""Tests for RedisFeatureStore with mocked redis."""

from unittest.mock import AsyncMock, MagicMock

from phoenix_ml.infrastructure.feature_store.redis_feature_store import RedisFeatureStore


async def test_get_online_features_found() -> None:
    store = RedisFeatureStore.__new__(RedisFeatureStore)
    store.redis = MagicMock()
    store.redis.hmget = AsyncMock(return_value=["1.0", "2.0"])

    result = await store.get_online_features("e1", ["f1", "f2"])
    assert result == [1.0, 2.0]


async def test_get_online_features_not_found() -> None:
    store = RedisFeatureStore.__new__(RedisFeatureStore)
    store.redis = MagicMock()
    store.redis.hmget = AsyncMock(return_value=[None, None])

    result = await store.get_online_features("e1", ["f1", "f2"])
    assert result is None


async def test_get_online_features_partial() -> None:
    store = RedisFeatureStore.__new__(RedisFeatureStore)
    store.redis = MagicMock()
    store.redis.hmget = AsyncMock(return_value=["3.5", None])

    result = await store.get_online_features("e1", ["f1", "f2"])
    assert result == [3.5, 0.0]


async def test_get_online_features_exception() -> None:
    store = RedisFeatureStore.__new__(RedisFeatureStore)
    store.redis = MagicMock()
    store.redis.hmget = AsyncMock(side_effect=ConnectionError("redis down"))

    result = await store.get_online_features("e1", ["f1"])
    assert result is None


async def test_add_features() -> None:
    store = RedisFeatureStore.__new__(RedisFeatureStore)
    store.redis = MagicMock()
    store.redis.hset = AsyncMock()

    await store.add_features("e1", {"f1": 1.0, "f2": 2.0})
    store.redis.hset.assert_called_once()
