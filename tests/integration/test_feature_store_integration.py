"""
Integration Test: Feature Store.

Tests online feature retrieval and seeding via the InMemoryFeatureStore,
mirroring real feature lookups during inference.
"""

import pytest

from src.infrastructure.feature_store.in_memory_feature_store import InMemoryFeatureStore


@pytest.fixture
def feature_store() -> InMemoryFeatureStore:
    fs = InMemoryFeatureStore()
    return fs


class TestFeatureStoreIntegration:
    """Integration tests for feature store operations."""

    async def test_add_and_retrieve_features(
        self, feature_store: InMemoryFeatureStore
    ) -> None:
        """Features can be stored and retrieved by entity + names."""
        await feature_store.add_features(
            "customer-001",
            {"age": 30.0, "income": 50000.0, "credit_score": 720.0},
        )

        result = await feature_store.get_online_features(
            "customer-001", ["age", "income", "credit_score"]
        )
        assert result is not None
        assert result == [30.0, 50000.0, 720.0]

    async def test_missing_entity_returns_none(
        self, feature_store: InMemoryFeatureStore
    ) -> None:
        result = await feature_store.get_online_features(
            "nonexistent", ["age"]
        )
        assert result is None

    async def test_missing_feature_defaults_to_zero(
        self, feature_store: InMemoryFeatureStore
    ) -> None:
        """Missing feature names default to 0.0."""
        await feature_store.add_features(
            "customer-002", {"age": 25.0}
        )

        result = await feature_store.get_online_features(
            "customer-002", ["age", "nonexistent_feature"]
        )
        assert result is not None
        assert result == [25.0, 0.0]

    async def test_overwrite_features(
        self, feature_store: InMemoryFeatureStore
    ) -> None:
        """Adding features for same entity overwrites previous data."""
        await feature_store.add_features("customer-003", {"age": 30.0})
        await feature_store.add_features("customer-003", {"age": 35.0, "score": 700.0})

        result = await feature_store.get_online_features(
            "customer-003", ["age", "score"]
        )
        assert result is not None
        assert result == [35.0, 700.0]

    async def test_multiple_entities(
        self, feature_store: InMemoryFeatureStore
    ) -> None:
        """Features for different entities are independent."""
        await feature_store.add_features("a", {"x": 1.0})
        await feature_store.add_features("b", {"x": 2.0})

        r_a = await feature_store.get_online_features("a", ["x"])
        r_b = await feature_store.get_online_features("b", ["x"])

        assert r_a == [1.0]
        assert r_b == [2.0]
