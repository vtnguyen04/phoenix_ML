"""Tests for RedisDataIngestor."""

from unittest.mock import AsyncMock, MagicMock

from src.shared.ingestion.redis_ingestor import RedisDataIngestor


class TestRedisDataIngestor:
    async def test_ingest_calls_hset(self) -> None:
        ingestor = RedisDataIngestor.__new__(RedisDataIngestor)
        ingestor.redis = MagicMock()
        ingestor.redis.hset = AsyncMock(return_value=1)

        success = await ingestor.ingest("e1", {"f1": 1.0})
        assert success is True
        ingestor.redis.hset.assert_called_once()

    async def test_batch_ingest(self) -> None:
        ingestor = RedisDataIngestor.__new__(RedisDataIngestor)
        ingestor.redis = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.hset = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[1, 1, 1])
        mock_pipe.__aenter__ = AsyncMock(return_value=mock_pipe)
        mock_pipe.__aexit__ = AsyncMock(return_value=None)
        ingestor.redis.pipeline = MagicMock(return_value=mock_pipe)

        data = [
            {"entity_id": "e1", "data": {"f1": 1.0}},
            {"entity_id": "e2", "data": {"f2": 2.0}},
            {"entity_id": "e3", "data": {"f3": 3.0}},
        ]
        result = await ingestor.batch_ingest(data)
        assert result == 3
