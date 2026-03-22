"""
Tests for InMemoryFeatureStore — in-memory feature storage implementation.
"""

import pytest

from phoenix_ml.infrastructure.feature_store.in_memory_feature_store import (
    InMemoryFeatureStore,
)


class TestInMemoryFeatureStore:
    """Unit tests for InMemoryFeatureStore."""

    @pytest.fixture()
    def store(self) -> InMemoryFeatureStore:
        return InMemoryFeatureStore()

    @pytest.mark.asyncio()
    async def test_add_and_get_features(self, store: InMemoryFeatureStore) -> None:
        features = {"age": 30.0, "income": 50000.0}
        await store.add_features("entity-1", features)
        result = await store.get_online_features("entity-1", ["age", "income"])
        assert result == [30.0, 50000.0]

    @pytest.mark.asyncio()
    async def test_get_missing_entity_returns_none(
        self,
        store: InMemoryFeatureStore,
    ) -> None:
        result = await store.get_online_features("nonexistent", ["age"])
        assert result is None

    @pytest.mark.asyncio()
    async def test_get_missing_feature_defaults_to_zero(
        self,
        store: InMemoryFeatureStore,
    ) -> None:
        await store.add_features("entity-1", {"age": 30.0})
        result = await store.get_online_features("entity-1", ["age", "missing"])
        assert result == [30.0, 0.0]

    @pytest.mark.asyncio()
    async def test_overwrite_features(self, store: InMemoryFeatureStore) -> None:
        await store.add_features("entity-1", {"age": 20.0})
        await store.add_features("entity-1", {"age": 30.0, "income": 5000.0})
        result = await store.get_online_features("entity-1", ["age", "income"])
        assert result == [30.0, 5000.0]

    @pytest.mark.asyncio()
    async def test_multiple_entities(self, store: InMemoryFeatureStore) -> None:
        await store.add_features("a", {"x": 1.0})
        await store.add_features("b", {"x": 2.0})
        assert (await store.get_online_features("a", ["x"])) == [1.0]
        assert (await store.get_online_features("b", ["x"])) == [2.0]

    @pytest.mark.asyncio()
    async def test_feature_order_preserved(self, store: InMemoryFeatureStore) -> None:
        await store.add_features("e1", {"a": 1.0, "b": 2.0, "c": 3.0})
        result = await store.get_online_features("e1", ["c", "a", "b"])
        assert result == [3.0, 1.0, 2.0]
