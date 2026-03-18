from unittest.mock import MagicMock, patch

import pytest

from src.domain.inference.entities.model import Model
from src.infrastructure.persistence.mlflow_model_registry import MlflowModelRegistry


@pytest.fixture
def mlflow_registry() -> MlflowModelRegistry:
    with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
        return MlflowModelRegistry(tracking_uri="http://localhost:5000")


@pytest.mark.asyncio
async def test_mlflow_save_model(mlflow_registry: MlflowModelRegistry) -> None:
    model = Model(
        id="m1",
        version="v1",
        uri="local://path/to/model.onnx",
        framework="onnx",
        metadata={"metrics": {"accuracy": 0.9}, "role": "champion"},
    )

    mock_client = MagicMock()
    mv = MagicMock()
    mv.name = "m1"
    mv.version = "1"
    mv.current_stage = "None"
    mv.source = "mlflow-artifacts:/models/m1/1"
    mv.run_id = None
    mv.creation_timestamp = None
    mock_client.search_model_versions.return_value = [mv]

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client), patch(
        "mlflow.start_run"
    ), patch("mlflow.log_params"), patch("mlflow.log_param"), patch(
        "mlflow.log_metric"
    ) as mock_log_metric, patch("mlflow.onnx.log_model") as mock_log_model, patch(
        "onnx.load"
    ) as mock_onnx_load:
        mock_onnx_load.return_value = MagicMock()
        await mlflow_registry.save(model)
        mock_log_model.assert_called_once()
        mock_log_metric.assert_called()
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="m1", version="1", stage="Production"
        )


@pytest.mark.asyncio
async def test_mlflow_update_stage(mlflow_registry: MlflowModelRegistry) -> None:
    mv = MagicMock()
    mv.name = "m1"
    mv.version = "7"
    mv.source = "mlflow-artifacts:/models/m1/7"
    mv.current_stage = "Staging"
    mv.run_id = "run-123"
    mv.creation_timestamp = None

    run = MagicMock()
    run.data.metrics = {}
    run.data.params = {"phoenix_model_version": "v1"}

    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [mv]
    mock_client.get_run.return_value = run

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        await mlflow_registry.update_stage("m1", "v1", "champion")
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="m1", version="7", stage="Production"
        )


@pytest.mark.asyncio
async def test_mlflow_get_by_id(mlflow_registry: MlflowModelRegistry) -> None:
    mv = MagicMock()
    mv.name = "m1"
    mv.version = "7"
    mv.source = "mlflow-artifacts:/models/m1/7"
    mv.current_stage = "Production"
    mv.run_id = "run-123"
    mv.creation_timestamp = None

    run = MagicMock()
    run.data.metrics = {"accuracy": 0.91}
    run.data.params = {"phoenix_model_version": "v1"}

    mock_client = MagicMock()
    mock_client.get_model_version.return_value = mv
    mock_client.get_run.return_value = run

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        got = await mlflow_registry.get_by_id("m1", "7")

    assert got is not None
    assert got.id == "m1"
    assert got.version == "7"
    assert got.metadata["metrics"]["accuracy"] == 0.91


@pytest.mark.asyncio
async def test_mlflow_get_champion_none(mlflow_registry: MlflowModelRegistry) -> None:
    mv = MagicMock()
    mv.name = "m1"
    mv.version = "1"
    mv.source = "mlflow-artifacts:/models/m1/1"
    mv.current_stage = "Staging"
    mv.run_id = None

    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [mv]

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        got = await mlflow_registry.get_champion("m1")

    assert got is None
