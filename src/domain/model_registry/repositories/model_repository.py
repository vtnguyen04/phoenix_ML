from abc import ABC, abstractmethod

from src.domain.inference.entities.model import Model


class ModelRepository(ABC):
    @abstractmethod
    async def save(self, model: Model) -> None:
        pass

    @abstractmethod
    async def get_by_id(self, model_id: str, version: str) -> Model | None:
        pass

    @abstractmethod
    async def get_active_versions(self, model_id: str) -> list[Model]:
        pass

    @abstractmethod
    async def get_champion(self, model_id: str) -> Model | None:
        pass

    @abstractmethod
    async def update_stage(self, model_id: str, version: str, stage: str) -> None:
        pass
