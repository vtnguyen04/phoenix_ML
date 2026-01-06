import pytest
from unittest.mock import AsyncMock, Mock

from src.application.services.monitoring_service import MonitoringService
from src.application.commands.predict_command import PredictCommand
from src.domain.inference.entities.prediction import Prediction
from src.domain.monitoring.entities.drift_report import DriftReport
from src.domain.monitoring.repositories.prediction_log_repository import PredictionLogRepository
from src.domain.monitoring.services.drift_calculator import DriftCalculator

@pytest.fixture
def mock_log_repo() -> PredictionLogRepository:
    return AsyncMock(spec=PredictionLogRepository)

@pytest.fixture
def mock_drift_calculator() -> DriftCalculator:
    return Mock(spec=DriftCalculator)

@pytest.fixture
def monitoring_service(
    mock_log_repo: PredictionLogRepository, 
    mock_drift_calculator: DriftCalculator
) -> MonitoringService:
    return MonitoringService(mock_log_repo, mock_drift_calculator)

@pytest.mark.asyncio
async def test_check_drift_success(
    monitoring_service: MonitoringService,
    mock_log_repo: AsyncMock,
    mock_drift_calculator: Mock
) -> None:
    # Setup Mock Data
    logs = [
        (PredictCommand(model_id="m1", model_version="v1", features=[1.0]), Mock(spec=Prediction)),
        (PredictCommand(model_id="m1", model_version="v1", features=[2.0]), Mock(spec=Prediction)),
        (PredictCommand(model_id="m1", model_version="v1", features=[3.0]), Mock(spec=Prediction)),
        (PredictCommand(model_id="m1", model_version="v1", features=[4.0]), Mock(spec=Prediction)),
        (PredictCommand(model_id="m1", model_version="v1", features=[5.0]), Mock(spec=Prediction)),
    ]
    mock_log_repo.get_recent_logs.return_value = logs
    
    mock_drift_calculator.calculate_drift.return_value = DriftReport(
        drift_type="ks",
        feature_name="feature_0",
        drift_detected=True,
        p_value=0.01,
        statistic=0.5,
        threshold=0.05
    )

    # Execute
    report = await monitoring_service.check_drift("m1", reference_data=[0.0]*5, feature_index=0)

    # Verify
    assert report.drift_detected is True
    mock_drift_calculator.calculate_drift.assert_called_once()
    mock_log_repo.get_recent_logs.assert_awaited_once_with("m1", limit=1000)

@pytest.mark.asyncio
async def test_check_drift_not_enough_data(
    monitoring_service: MonitoringService,
    mock_log_repo: AsyncMock
) -> None:
    # Only 2 logs
    logs = [
        (PredictCommand(model_id="m1", model_version="v1", features=[1.0]), Mock(spec=Prediction)),
        (PredictCommand(model_id="m1", model_version="v1", features=[2.0]), Mock(spec=Prediction)),
    ]
    mock_log_repo.get_recent_logs.return_value = logs

    with pytest.raises(ValueError, match="Not enough data points"):
        await monitoring_service.check_drift("m1", reference_data=[], feature_index=0)
