import logging
from typing import Any

from phoenix_ml.shared.ingestion.interfaces import IDataIngestor

logger = logging.getLogger(__name__)


class IngestionService:
    """Orchestrates data ingestion via an ``IDataIngestor`` backend."""

    def __init__(self, ingestor: IDataIngestor):
        self._ingestor = ingestor

    async def process_raw_data(self, entity_id: str, raw_features: dict[str, Any]) -> bool:
        """Clean raw features and forward to the ingestor backend."""
        logger.info("Processing data for entity: %s", entity_id)

        cleaned_data = {k: float(v) for k, v in raw_features.items()}

        return await self._ingestor.ingest(entity_id, cleaned_data)

    async def process_batch(self, batch_data: list[dict[str, Any]]) -> int:
        """Processes a batch of raw records."""
        return await self._ingestor.batch_ingest(batch_data)
