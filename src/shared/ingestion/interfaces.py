from abc import ABC, abstractmethod
from typing import Any


class IDataIngestor(ABC):
    """
    Interface for data ingestion following Clean Architecture.
    Allows swappable implementations (API, Kafka, Direct DB).
    """
    
    @abstractmethod
    async def ingest(self, entity_id: str, data: dict[str, float]) -> bool:
        """Ingests a single data record for an entity."""
        pass

    @abstractmethod
    async def batch_ingest(self, data_list: list[dict[str, Any]]) -> int:
        """Ingests a batch of data records. Returns count of successful ingestions."""
        pass
