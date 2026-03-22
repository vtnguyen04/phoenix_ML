"""Tests for IngestionService."""

from unittest.mock import AsyncMock, MagicMock

from phoenix_ml.shared.ingestion.interfaces import IDataIngestor
from phoenix_ml.shared.ingestion.service import IngestionService


async def test_process_raw_data_delegates_to_ingestor() -> None:
    mock_ingestor = MagicMock(spec=IDataIngestor)
    mock_ingestor.ingest = AsyncMock(return_value=True)

    service = IngestionService(mock_ingestor)
    result = await service.process_raw_data("entity-1", {"f1": 1.0, "f2": 2.0})

    assert result is True
    mock_ingestor.ingest.assert_called_once()


async def test_process_batch_delegates_batch_ingest() -> None:
    mock_ingestor = MagicMock(spec=IDataIngestor)
    mock_ingestor.batch_ingest = AsyncMock(return_value=3)

    service = IngestionService(mock_ingestor)
    result = await service.process_batch(
        [
            {"entity_id": "e1", "data": {"f1": 1.0}},
            {"entity_id": "e2", "data": {"f1": 2.0}},
            {"entity_id": "e3", "data": {"f1": 3.0}},
        ]
    )

    assert result == 3
    mock_ingestor.batch_ingest.assert_called_once()
