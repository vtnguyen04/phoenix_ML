"""
Integration Test: Model Registry lifecycle.

Tests model CRUD operations and version lifecycle via the in-memory registry,
verifying the same paths the API endpoints exercise.
"""

import pytest

from src.domain.inference.entities.model import Model
from src.infrastructure.persistence.in_memory_model_repo import InMemoryModelRepository


@pytest.fixture
def model_repo() -> InMemoryModelRepository:
    return InMemoryModelRepository()


@pytest.fixture
def sample_model() -> Model:
    return Model(
        id="credit-risk",
        version="v1",
        uri="s3://models/credit-risk/v1/model.onnx",
        framework="onnx",
        metadata={"features": ["f1", "f2", "f3"], "role": "staging"},
    )


class TestModelRegistryIntegration:
    """Integration tests for model registry operations."""

    async def test_save_and_retrieve_model(
        self, model_repo: InMemoryModelRepository, sample_model: Model
    ) -> None:
        """Model can be saved and retrieved by id + version."""
        await model_repo.save(sample_model)
        retrieved = await model_repo.get_by_id("credit-risk", "v1")

        assert retrieved is not None
        assert retrieved.id == "credit-risk"
        assert retrieved.version == "v1"
        assert retrieved.framework == "onnx"

    async def test_get_nonexistent_model_returns_none(
        self, model_repo: InMemoryModelRepository
    ) -> None:
        result = await model_repo.get_by_id("nonexistent", "v1")
        assert result is None

    async def test_version_lifecycle_staging_to_champion(
        self, model_repo: InMemoryModelRepository, sample_model: Model
    ) -> None:
        """Model can progress from staging → champion."""
        await model_repo.save(sample_model)

        await model_repo.update_stage("credit-risk", "v1", "champion")
        champion = await model_repo.get_champion("credit-risk")

        assert champion is not None
        assert champion.version == "v1"
        assert champion.metadata["role"] == "champion"

    async def test_promoting_new_champion_retires_old(
        self, model_repo: InMemoryModelRepository, sample_model: Model
    ) -> None:
        """Promoting a new champion retires the previous one."""
        await model_repo.save(sample_model)
        await model_repo.update_stage("credit-risk", "v1", "champion")

        v2 = Model(
            id="credit-risk",
            version="v2",
            uri="s3://models/credit-risk/v2/model.onnx",
            framework="onnx",
            metadata={"features": ["f1", "f2", "f3"], "role": "staging"},
        )
        await model_repo.save(v2)
        await model_repo.update_stage("credit-risk", "v2", "champion")

        old = await model_repo.get_by_id("credit-risk", "v1")
        new = await model_repo.get_champion("credit-risk")

        assert old is not None
        assert old.metadata["role"] == "retired"
        assert new is not None
        assert new.version == "v2"

    async def test_get_active_versions(self, model_repo: InMemoryModelRepository) -> None:
        """Active versions returns models with active status."""
        for ver in ("v1", "v2", "v3"):
            m = Model(
                id="credit-risk",
                version=ver,
                uri=f"s3://models/credit-risk/{ver}/model.onnx",
                framework="onnx",
                metadata={"role": "staging"},
            )
            await model_repo.save(m)

        active = await model_repo.get_active_versions("credit-risk")
        assert len(active) == 3
        versions = {m.version for m in active}
        assert versions == {"v1", "v2", "v3"}
