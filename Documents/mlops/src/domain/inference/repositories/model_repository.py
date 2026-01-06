from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.inference.entities.model import Model

class ModelRepository(ABC):
    """
    Interface for Model Storage (S3, Local, DB)
    """
    
    @abstractmethod
    async def save(self, model: Model) -> None:
        pass

    @abstractmethod
    async def get_by_id(self, model_id: str, version: str) -> Optional[Model]:
        pass

    @abstractmethod
    async def list_active_models(self) -> List[Model]:
        pass
