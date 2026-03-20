"""Tests for ApiDataIngestor and CreditDataCollector."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.shared.ingestion.api_ingestor import ApiDataIngestor


class TestApiDataIngestor:
    async def test_ingest_success(self) -> None:
        ingestor = ApiDataIngestor(base_url="http://test:8000")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await ingestor.ingest("e1", {"f1": 1.0})
            assert result is True

    async def test_ingest_failure(self) -> None:
        ingestor = ApiDataIngestor(base_url="http://test:8000")

        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await ingestor.ingest("e1", {"f1": 1.0})
            assert result is False

    async def test_ingest_exception(self) -> None:
        ingestor = ApiDataIngestor(base_url="http://test:8000")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = ConnectionError("refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            result = await ingestor.ingest("e1", {"f1": 1.0})
            assert result is False

    async def test_batch_ingest(self) -> None:
        ingestor = ApiDataIngestor(base_url="http://test:8000")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            data = [
                {"entity_id": "e1", "data": {"f1": 1.0}},
                {"entity_id": "e2", "data": {"f2": 2.0}},
            ]
            result = await ingestor.batch_ingest(data)
            assert result == 2
