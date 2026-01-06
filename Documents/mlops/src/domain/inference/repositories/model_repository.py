from abc import ABC, abstractmethod

from src.domain.inference.entities.model import Model


class ModelRepository(ABC):
    """
    Interface for Model Storage (S3, Local, DB)
    """
    
    @abstractmethod
    async def save(self, model: Model) -> None:
        pass

    @abstractmethod
    async def get_by_id(self, model_id: str, version: str) -> Model | None:
        pass

    @abstractmethod
    async def list_active_models(self) -> list[Model]:
        pass

    @abstractmethod
    async def get_active_versions(self, model_id: str) -> list[Model]:
        """Get all active versions for a specific model group"""
        pass
