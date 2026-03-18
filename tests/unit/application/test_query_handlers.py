"""Unit tests for CQRS Query Handlers."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.handlers.query_handlers import (
    GetDriftReportQueryHandler,
    GetModelPerformanceQueryHandler,
    GetModelQueryHandler,
    GetPredictionLogsQueryHandler,
)
from src.application.queries import (
    GetDriftReportQuery,
    GetModelPerformanceQuery,
    GetModelQuery,
    GetPredictionLogsQuery,
)


@pytest.fixture
def mock_model_repo() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_drift_repo() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_log_repo() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_evaluator() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# GetModelQueryHandler
# ---------------------------------------------------------------------------


class TestGetModelQueryHandler:
    async def test_get_model_with_version(self, mock_model_repo: AsyncMock) -> None:
        mock_model = MagicMock(id="credit-risk", version="v1")
        mock_model_repo.get_by_id.return_value = mock_model

        handler = GetModelQueryHandler(mock_model_repo)
        result = await handler.execute(
            GetModelQuery(model_id="credit-risk", version="v1")
        )

        assert result is mock_model
        mock_model_repo.get_by_id.assert_awaited_once_with("credit-risk", "v1")

    async def test_get_model_no_version_returns_champion(
        self, mock_model_repo: AsyncMock
    ) -> None:
        mock_model = MagicMock(id="credit-risk", version="v2")
        mock_model_repo.get_champion.return_value = mock_model

        handler = GetModelQueryHandler(mock_model_repo)
        result = await handler.execute(GetModelQuery(model_id="credit-risk"))

        assert result is mock_model
        mock_model_repo.get_champion.assert_awaited_once_with("credit-risk")

    async def test_get_model_not_found(self, mock_model_repo: AsyncMock) -> None:
        mock_model_repo.get_champion.return_value = None
        handler = GetModelQueryHandler(mock_model_repo)
        result = await handler.execute(GetModelQuery(model_id="unknown"))
        assert result is None


# ---------------------------------------------------------------------------
# GetDriftReportQueryHandler
# ---------------------------------------------------------------------------


class TestGetDriftReportQueryHandler:
    async def test_get_drift_reports(self, mock_drift_repo: AsyncMock) -> None:
        mock_reports = [MagicMock(), MagicMock()]
        mock_drift_repo.get_history.return_value = mock_reports

        handler = GetDriftReportQueryHandler(mock_drift_repo)
        result = await handler.execute(
            GetDriftReportQuery(model_id="credit-risk", limit=5)
        )

        assert result == mock_reports
        mock_drift_repo.get_history.assert_awaited_once_with("credit-risk", 5)


# ---------------------------------------------------------------------------
# GetPredictionLogsQueryHandler
# ---------------------------------------------------------------------------


class TestGetPredictionLogsQueryHandler:
    async def test_get_prediction_logs(self, mock_log_repo: AsyncMock) -> None:
        mock_command = MagicMock(model_id="m1", model_version="v1")
        mock_prediction = MagicMock(
            result=[0.8], confidence=MagicMock(value=0.95), latency_ms=12.5
        )
        mock_log_repo.get_recent_logs.return_value = [
            (mock_command, mock_prediction),
        ]

        handler = GetPredictionLogsQueryHandler(mock_log_repo)
        result = await handler.execute(
            GetPredictionLogsQuery(model_id="m1", limit=50)
        )

        assert len(result) == 1
        assert result[0]["model_id"] == "m1"
        assert result[0]["confidence"] == 0.95


# ---------------------------------------------------------------------------
# GetModelPerformanceQueryHandler
# ---------------------------------------------------------------------------


class TestGetModelPerformanceQueryHandler:
    async def test_performance_with_logs(
        self, mock_log_repo: AsyncMock, mock_evaluator: MagicMock
    ) -> None:
        pred = MagicMock(
            result=[0.9], confidence=MagicMock(value=0.92), latency_ms=15.0
        )
        mock_log_repo.get_recent_logs.return_value = [
            (MagicMock(), pred),
            (MagicMock(), pred),
        ]

        handler = GetModelPerformanceQueryHandler(mock_log_repo, mock_evaluator)
        result = await handler.execute(
            GetModelPerformanceQuery(model_id="m1")
        )

        assert result["total_predictions"] == 2
        assert result["metrics"]["avg_latency_ms"] == 15.0
        assert result["metrics"]["avg_confidence"] == 0.92

    async def test_performance_no_logs(
        self, mock_log_repo: AsyncMock, mock_evaluator: MagicMock
    ) -> None:
        mock_log_repo.get_recent_logs.return_value = []

        handler = GetModelPerformanceQueryHandler(mock_log_repo, mock_evaluator)
        result = await handler.execute(
            GetModelPerformanceQuery(model_id="m1")
        )

        assert result["total_predictions"] == 0
        assert result["metrics"] == {}
