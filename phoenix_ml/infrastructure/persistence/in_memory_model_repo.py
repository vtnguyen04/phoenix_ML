from phoenix_ml.domain.inference.entities.model import Model
from phoenix_ml.domain.model_registry.repositories.model_repository import ModelRepository


class InMemoryModelRepository(ModelRepository):
    """
    In-memory implementation of ModelRepository for testing and rapid prototyping.
    """

    def __init__(self) -> None:
        self._models: dict[str, Model] = {}

    async def save(self, model: Model) -> None:
        self._models[model.unique_key] = model

    async def get_by_id(self, model_id: str, version: str) -> Model | None:
        key = f"{model_id}:{version}"
        return self._models.get(key)

    async def get_active_versions(self, model_id: str) -> list[Model]:
        return [m for m in self._models.values() if m.id == model_id and m.is_active]

    async def get_champion(self, model_id: str) -> Model | None:
        for m in self._models.values():
            if m.id == model_id and m.metadata.get("role") == "champion":
                return m
        return None

    async def update_stage(self, model_id: str, version: str, stage: str) -> None:
        if stage == "champion":
            for m in self._models.values():
                if m.id == model_id and m.metadata.get("role") == "champion":
                    m.metadata["role"] = "retired"

        key = f"{model_id}:{version}"
        if key in self._models:
            self._models[key].metadata["role"] = stage

    async def list_all(self) -> list[Model]:
        return list(self._models.values())
