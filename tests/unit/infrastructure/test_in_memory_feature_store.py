"""Tests for InMemoryFeatureStore implementation."""

import pytest

from phoenix_ml.infrastructure.feature_store.in_memory_feature_store import InMemoryFeatureStore


@pytest.fixture
def store() -> InMemoryFeatureStore:
    return InMemoryFeatureStore()


async def test_add_and_get_features(store: InMemoryFeatureStore) -> None:
    await store.add_features("e1", {"f1": 1.0, "f2": 2.0})
    result = await store.get_online_features("e1", ["f1", "f2"])
    assert result is not None
    assert result == [1.0, 2.0]


async def test_missing_entity_returns_none(store: InMemoryFeatureStore) -> None:
    result = await store.get_online_features("nonexistent", ["f1"])
    assert result is None


async def test_partial_features_default_to_zero(store: InMemoryFeatureStore) -> None:
    await store.add_features("e2", {"a": 10.0})
    result = await store.get_online_features("e2", ["a", "b"])
    assert result is not None
    assert result[0] == 10.0
    assert result[1] == 0.0
