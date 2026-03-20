from unittest.mock import AsyncMock

import pytest

from src.domain.monitoring.services.rollback_manager import (
    ChallengerMetrics,
    RollbackManager,
)


@pytest.fixture
def mock_model_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.update_stage = AsyncMock()
    return repo


@pytest.fixture
def rollback_manager(mock_model_repo: AsyncMock) -> RollbackManager:
    return RollbackManager(
        model_repo=mock_model_repo,
        error_rate_threshold=0.10,
        latency_threshold_ms=500.0,
        min_requests=50,
    )


@pytest.mark.asyncio
async def test_no_rollback_insufficient_data(
    rollback_manager: RollbackManager,
) -> None:
    metrics = ChallengerMetrics(
        model_id="m1",
        challenger_version="v2",
        champion_version="v1",
        total_requests=10,
        error_count=5,
        avg_latency_ms=100.0,
    )
    decision = await rollback_manager.evaluate_challenger(metrics)
    assert decision.should_rollback is False
    assert "Insufficient data" in decision.reason


@pytest.mark.asyncio
async def test_rollback_on_high_error_rate(
    rollback_manager: RollbackManager,
    mock_model_repo: AsyncMock,
) -> None:
    metrics = ChallengerMetrics(
        model_id="m1",
        challenger_version="v2",
        champion_version="v1",
        total_requests=100,
        error_count=15,
        avg_latency_ms=100.0,
    )
    decision = await rollback_manager.evaluate_challenger(metrics)
    assert decision.should_rollback is True
    assert "Error rate" in decision.reason
    mock_model_repo.update_stage.assert_awaited_once_with("m1", "v2", "archived")


@pytest.mark.asyncio
async def test_rollback_on_latency_spike(
    rollback_manager: RollbackManager,
    mock_model_repo: AsyncMock,
) -> None:
    metrics = ChallengerMetrics(
        model_id="m1",
        challenger_version="v2",
        champion_version="v1",
        total_requests=100,
        error_count=2,
        avg_latency_ms=600.0,
    )
    decision = await rollback_manager.evaluate_challenger(metrics)
    assert decision.should_rollback is True
    assert "latency" in decision.reason
    mock_model_repo.update_stage.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_rollback_when_healthy(
    rollback_manager: RollbackManager,
) -> None:
    metrics = ChallengerMetrics(
        model_id="m1",
        challenger_version="v2",
        champion_version="v1",
        total_requests=100,
        error_count=5,
        avg_latency_ms=200.0,
    )
    decision = await rollback_manager.evaluate_challenger(metrics)
    assert decision.should_rollback is False
    assert decision.reason == "Challenger is healthy"
    assert decision.error_rate == 0.05
