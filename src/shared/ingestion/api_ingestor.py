import logging
from typing import Any

import httpx

from src.shared.ingestion.interfaces import IDataIngestor

logger = logging.getLogger(__name__)

SUCCESS_STATUS = 200


class ApiDataIngestor(IDataIngestor):
    """
    Infrastructure implementation of IDataIngestor using HTTP API.
    Points to the running FastAPI server.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def ingest(self, entity_id: str, data: dict[str, float]) -> bool:
        """
        Ingests data records into the Feature Store.
        Currently verifies connectivity via health check for this prototype.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == SUCCESS_STATUS
        except Exception as e:
            logger.error("Ingestion failed: %s", e)
            return False

    async def batch_ingest(self, data_list: list[dict[str, Any]]) -> int:
        success_count = 0
        for item in data_list:
            if await self.ingest(item["entity_id"], item["data"]):
                success_count += 1
        return success_count
