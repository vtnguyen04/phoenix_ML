"""Tests for ingestion layer: IngestionService, IDataIngestor interface."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from phoenix_ml.shared.ingestion.interfaces import IDataIngestor
from phoenix_ml.shared.ingestion.service import IngestionService

# ─── Mock Ingestor ───────────────────────────────────────────────


class MockIngestor(IDataIngestor):
    """Test double that records calls."""

    def __init__(self) -> None:
        self.ingested: list[tuple[str, dict[str, float]]] = []
        self.batch_data: list[dict[str, Any]] = []

    async def ingest(self, entity_id: str, data: dict[str, float]) -> bool:
        self.ingested.append((entity_id, data))
        return True

    async def batch_ingest(self, data_list: list[dict[str, Any]]) -> int:
        self.batch_data.extend(data_list)
        return len(data_list)


class FailingIngestor(IDataIngestor):
    """Ingestor that always fails."""

    async def ingest(self, entity_id: str, data: dict[str, float]) -> bool:
        return False

    async def batch_ingest(self, data_list: list[dict[str, Any]]) -> int:
        return 0


# ─── IDataIngestor Interface ─────────────────────────────────────


class TestIDataIngestor:
    @pytest.mark.asyncio
    async def test_mock_ingestor_single(self) -> None:
        ingestor = MockIngestor()
        result = await ingestor.ingest("cust-001", {"income": 50000.0, "age": 30.0})
        assert result is True
        assert len(ingestor.ingested) == 1
        assert ingestor.ingested[0][0] == "cust-001"

    @pytest.mark.asyncio
    async def test_mock_ingestor_batch(self) -> None:
        ingestor = MockIngestor()
        batch = [
            {"entity_id": "c1", "income": 10.0},
            {"entity_id": "c2", "income": 20.0},
        ]
        count = await ingestor.batch_ingest(batch)
        assert count == 2

    @pytest.mark.asyncio
    async def test_failing_ingestor(self) -> None:
        ingestor = FailingIngestor()
        result = await ingestor.ingest("cust-001", {"x": 1.0})
        assert result is False
        count = await ingestor.batch_ingest([{"x": 1.0}])
        assert count == 0


# ─── IngestionService ────────────────────────────────────────────


class TestIngestionService:
    @pytest.mark.asyncio
    async def test_process_raw_data_converts_to_float(self) -> None:
        mock = MockIngestor()
        service = IngestionService(ingestor=mock)

        raw = {"income": 50000, "age": "30", "score": 0.95}
        result = await service.process_raw_data("cust-001", raw)

        assert result is True
        assert len(mock.ingested) == 1
        entity_id, data = mock.ingested[0]
        assert entity_id == "cust-001"
        assert isinstance(data["income"], float)
        assert data["age"] == 30.0

    @pytest.mark.asyncio
    async def test_process_raw_data_failure(self) -> None:
        failing = FailingIngestor()
        service = IngestionService(ingestor=failing)

        result = await service.process_raw_data("cust-001", {"x": 1})
        assert result is False

    @pytest.mark.asyncio
    async def test_process_batch(self) -> None:
        mock = MockIngestor()
        service = IngestionService(ingestor=mock)

        batch = [{"a": 1}, {"b": 2}, {"c": 3}]
        count = await service.process_batch(batch)
        assert count == 3

    @pytest.mark.asyncio
    async def test_uses_injected_ingestor(self) -> None:
        """Verify DIP — service uses abstract IDataIngestor, not concrete."""
        mock_ingestor = AsyncMock(spec=IDataIngestor)
        mock_ingestor.ingest.return_value = True
        service = IngestionService(ingestor=mock_ingestor)

        await service.process_raw_data("test", {"val": 42})
        mock_ingestor.ingest.assert_called_once()
