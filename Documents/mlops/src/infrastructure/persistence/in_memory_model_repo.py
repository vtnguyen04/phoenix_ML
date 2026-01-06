from typing import List, Optional, Dict
from src.domain.inference.repositories.model_repository import ModelRepository
from src.domain.inference.entities.model import Model

class InMemoryModelRepository(ModelRepository):
    """
    In-memory implementation of ModelRepository for testing.
    """
    
    def __init__(self):
        self._models: Dict[str, Model] = {}

    async def save(self, model: Model) -> None:
        self._models[model.unique_key] = model

    async def get_by_id(self, model_id: str, version: str) -> Optional[Model]:
        return self._models.get(f"{model_id}:{version}")

    async def list_active_models(self) -> List[Model]:
        return [m for m in self._models.values() if m.is_active]
