import logging
from typing import Any

from src.shared.ingestion.interfaces import IDataIngestor

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Application Service that orchestrates the ingestion process.
    Adheres to Clean Architecture by using the IDataIngestor abstraction.
    """

    def __init__(self, ingestor: IDataIngestor):
        self._ingestor = ingestor

    async def process_raw_data(
        self, entity_id: str, raw_features: dict[str, Any]
    ) -> bool:
        """
        Processes and cleans raw data before ingestion.
        """
        logger.info("Processing data for entity: %s", entity_id)

        # In a real system, perform cleaning, normalization, etc.
        # For this prototype, we assume data is already cleaned.
        cleaned_data = {k: float(v) for k, v in raw_features.items()}

        return await self._ingestor.ingest(entity_id, cleaned_data)

    async def process_batch(self, batch_data: list[dict[str, Any]]) -> int:
        """Processes a batch of raw records."""
        return await self._ingestor.batch_ingest(batch_data)
