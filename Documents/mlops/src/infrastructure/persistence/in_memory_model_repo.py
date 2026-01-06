
from src.domain.inference.entities.model import Model
from src.domain.inference.repositories.model_repository import ModelRepository


class InMemoryModelRepository(ModelRepository):
    """
    In-memory implementation of ModelRepository for testing.
    """
    
    def __init__(self) -> None:
        self._models: dict[str, Model] = {}

    async def save(self, model: Model) -> None:
        self._models[model.unique_key] = model

    async def get_by_id(self, model_id: str, version: str) -> Model | None:
        return self._models.get(f"{model_id}:{version}")

    async def list_active_models(self) -> list[Model]:
        return [m for m in self._models.values() if m.is_active]

    async def get_active_versions(self, model_id: str) -> list[Model]:
        return [
            m for m in self._models.values() 
            if m.id == model_id and m.is_active
        ]
