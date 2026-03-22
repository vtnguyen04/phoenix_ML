"""Tests for InMemoryModelRepository."""

from phoenix_ml.domain.inference.entities.model import Model
from phoenix_ml.infrastructure.persistence.in_memory_model_repo import InMemoryModelRepository


async def test_save_and_get_champion() -> None:
    repo = InMemoryModelRepository()
    model = Model(
        id="m1",
        version="v1",
        uri="file:///test",
        framework="onnx",
        metadata={"role": "champion"},
        is_active=True,
    )
    await repo.save(model)
    result = await repo.get_champion("m1")
    assert result is not None
    assert result.id == "m1"
    assert result.version == "v1"


async def test_get_champion_returns_none_when_empty() -> None:
    repo = InMemoryModelRepository()
    result = await repo.get_champion("nonexistent")
    assert result is None
