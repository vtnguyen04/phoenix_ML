from typing import Any

import redis.asyncio as redis

from src.shared.ingestion.interfaces import IDataIngestor


class RedisDataIngestor(IDataIngestor):
    """
    Infrastructure implementation of IDataIngestor using Redis directly.
    Best for high-throughput low-latency ingestion.
    """

    def __init__(self, redis_url: str = "redis://localhost:6380"):
        self.redis = redis.from_url(redis_url, decode_responses=True)

    async def ingest(self, entity_id: str, data: dict[str, float]) -> bool:
        key = f"features:{entity_id}"
        await self.redis.hset(key, mapping=data)  # type: ignore[misc]
        return True

    async def batch_ingest(self, data_list: list[dict[str, Any]]) -> int:
        success_count = 0
        async with self.redis.pipeline(transaction=True) as pipe:
            for item in data_list:
                key = f"features:{item['entity_id']}"
                pipe.hset(key, mapping=item["data"])
            results = await pipe.execute()
            success_count = len(results)
        return success_count
