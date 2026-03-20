from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.domain.inference.entities.model import Model, ModelStage
from src.infrastructure.persistence.mlflow_model_registry import MlflowModelRegistry


@pytest.fixture
def mlflow_registry() -> MlflowModelRegistry:
    with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"):
        return MlflowModelRegistry(tracking_uri="http://localhost:5000")


def _mv(  # noqa: PLR0913
    name: str = "m1",
    version: str = "1",
    stage: str = "Production",
    source: str = "mlflow-artifacts:/models/m1/1",
    run_id: str | None = "run-123",
    ts: int | None = 1700000000000,
) -> MagicMock:
    mv = MagicMock()
    mv.name, mv.version, mv.source = name, version, source
    mv.current_stage, mv.run_id, mv.creation_timestamp = stage, run_id, ts
    return mv


def _run(metrics: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> MagicMock:
    r = MagicMock()
    r.data.metrics = metrics or {}
    r.data.params = params or {}
    return r


# ── Static Helpers ─────────────────────────────────────────────


class TestStaticHelpers:
    def test_require_local_path_valid(self) -> None:
        from pathlib import Path

        path = MlflowModelRegistry._require_local_path("local:///tmp/model.onnx")
        assert path == Path("/tmp/model.onnx")

    def test_require_local_path_invalid(self) -> None:
        with pytest.raises(ValueError, match="local://"):
            MlflowModelRegistry._require_local_path("s3://bucket/model.onnx")

    @pytest.mark.parametrize(
        "role,expected",
        [
            ("champion", "Production"),
            ("production", "Production"),
            ("prod", "Production"),
            ("challenger", "Staging"),
            ("staging", "Staging"),
            ("stage", "Staging"),
            ("archived", "Archived"),
            ("retired", "Archived"),
        ],
    )
    def test_map_role_to_stage(self, role: str, expected: str) -> None:
        assert MlflowModelRegistry._map_role_to_mlflow_stage(role) == expected

    def test_map_role_unknown(self) -> None:
        assert MlflowModelRegistry._map_role_to_mlflow_stage("custom") is None

    def test_map_role_non_string(self) -> None:
        assert MlflowModelRegistry._map_role_to_mlflow_stage(42) is None

    @pytest.mark.parametrize(
        "stage,expected",
        [
            ("Production", ModelStage.PRODUCTION),
            ("Staging", ModelStage.STAGING),
            ("Archived", ModelStage.ARCHIVED),
            ("None", ModelStage.DEVELOPMENT),
        ],
    )
    def test_mlflow_stage_to_domain(self, stage: str, expected: ModelStage) -> None:
        assert MlflowModelRegistry._mlflow_stage_to_domain(stage) == expected

    def test_ensure_onnx_ok(self) -> None:
        MlflowModelRegistry._ensure_supported_framework("onnx")

    def test_ensure_non_onnx_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            MlflowModelRegistry._ensure_supported_framework("pytorch")

    def test_log_numeric_metrics_dict(self) -> None:
        with patch("mlflow.log_metric") as mock:
            MlflowModelRegistry._log_numeric_metrics({"a": 0.9, "b": 0.1})
            assert mock.call_count == 2

    def test_log_numeric_metrics_not_dict(self) -> None:
        with patch("mlflow.log_metric") as mock:
            MlflowModelRegistry._log_numeric_metrics("string")
            mock.assert_not_called()

    def test_safe_created_at_valid(self) -> None:
        dt = MlflowModelRegistry._safe_created_at(1700000000000)
        assert dt.year == 2023

    def test_safe_created_at_none(self) -> None:
        from datetime import datetime

        assert isinstance(MlflowModelRegistry._safe_created_at(None), datetime)

    def test_safe_run_metrics_no_run(self) -> None:
        assert MlflowModelRegistry._safe_run_metrics(MagicMock(), None) == {}

    def test_safe_run_metrics_success(self) -> None:
        c = MagicMock()
        c.get_run.return_value = _run(metrics={"acc": 0.95})
        assert MlflowModelRegistry._safe_run_metrics(c, "run-1") == {"acc": 0.95}

    def test_safe_run_metrics_exception(self) -> None:
        c = MagicMock()
        c.get_run.side_effect = Exception("fail")
        assert MlflowModelRegistry._safe_run_metrics(c, "r") == {}

    def test_safe_phoenix_version_found(self) -> None:
        c = MagicMock()
        c.get_run.return_value = _run(params={"phoenix_model_version": "v1"})
        assert MlflowModelRegistry._safe_phoenix_version(c, "r") == "v1"

    def test_safe_phoenix_version_no_run(self) -> None:
        assert MlflowModelRegistry._safe_phoenix_version(MagicMock(), None) is None

    def test_safe_phoenix_version_no_param(self) -> None:
        c = MagicMock()
        c.get_run.return_value = _run(params={})
        assert MlflowModelRegistry._safe_phoenix_version(c, "r") is None

    def test_safe_phoenix_version_exception(self) -> None:
        c = MagicMock()
        c.get_run.side_effect = Exception("fail")
        assert MlflowModelRegistry._safe_phoenix_version(c, "r") is None

    def test_safe_get_model_version_ok(self) -> None:
        c = MagicMock()
        c.get_model_version.return_value = _mv()
        result = MlflowModelRegistry._safe_get_model_version(c, "m1", "1")
        assert result is not None

    def test_safe_get_model_version_error(self) -> None:
        c = MagicMock()
        c.get_model_version.side_effect = Exception("fail")
        assert MlflowModelRegistry._safe_get_model_version(c, "m1", "1") is None

    def test_latest_mlflow_version(self) -> None:
        c = MagicMock()
        mvs = [_mv(version="1"), _mv(version="3"), _mv(version="2")]
        c.search_model_versions.return_value = mvs
        latest = MlflowModelRegistry._latest_mlflow_version(c, "m1")
        assert latest is not None
        assert latest.version == "3"

    def test_latest_mlflow_version_empty(self) -> None:
        c = MagicMock()
        c.search_model_versions.return_value = []
        assert MlflowModelRegistry._latest_mlflow_version(c, "m1") is None

    def test_find_by_phoenix_version_found(self) -> None:
        c = MagicMock()
        mv = _mv(version="5", run_id="r1")
        c.search_model_versions.return_value = [mv]
        c.get_run.return_value = _run(params={"phoenix_model_version": "v1"})
        result = MlflowModelRegistry._find_by_phoenix_version(c, "m1", "v1")
        assert result is not None
        assert result.version == "5"

    def test_find_by_phoenix_version_not_found(self) -> None:
        c = MagicMock()
        c.search_model_versions.return_value = []
        assert MlflowModelRegistry._find_by_phoenix_version(c, "m1", "v99") is None


# ── Async Operations ──────────────────────────────────────────


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
    mv = _mv(stage="None", run_id=None, ts=None)
    mock_client.search_model_versions.return_value = [mv]

    with (
        patch("mlflow.tracking.MlflowClient", return_value=mock_client),
        patch("mlflow.start_run"),
        patch("mlflow.log_params"),
        patch("mlflow.log_param"),
        patch("mlflow.log_metric") as mock_log,
        patch("mlflow.onnx.log_model") as mock_model,
        patch("onnx.load", return_value=MagicMock()),
    ):
        await mlflow_registry.save(model)
        mock_model.assert_called_once()
        mock_log.assert_called()


@pytest.mark.asyncio
async def test_mlflow_update_stage(mlflow_registry: MlflowModelRegistry) -> None:
    mv = _mv(version="7", stage="Staging")
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [mv]
    mock_client.get_run.return_value = _run(params={"phoenix_model_version": "v1"})

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        await mlflow_registry.update_stage("m1", "v1", "champion")
        mock_client.transition_model_version_stage.assert_called_once()


@pytest.mark.asyncio
async def test_mlflow_get_by_id(mlflow_registry: MlflowModelRegistry) -> None:
    mock_client = MagicMock()
    mock_client.get_model_version.return_value = _mv(version="7")
    mock_client.get_run.return_value = _run(metrics={"accuracy": 0.91})

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        got = await mlflow_registry.get_by_id("m1", "7")
    assert got is not None
    assert got.id == "m1"


@pytest.mark.asyncio
async def test_mlflow_get_by_id_phoenix_version(mlflow_registry: MlflowModelRegistry) -> None:
    mv = _mv(version="5")
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [mv]
    mock_client.get_run.return_value = _run(params={"phoenix_model_version": "v1"})

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        got = await mlflow_registry.get_by_id("m1", "v1")
    assert got is not None


@pytest.mark.asyncio
async def test_mlflow_get_champion_none(mlflow_registry: MlflowModelRegistry) -> None:
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [_mv(stage="Staging")]

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        assert await mlflow_registry.get_champion("m1") is None


@pytest.mark.asyncio
async def test_mlflow_get_champion_found(mlflow_registry: MlflowModelRegistry) -> None:
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [_mv(stage="Production")]
    mock_client.get_run.return_value = _run()

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        result = await mlflow_registry.get_champion("m1")
    assert result is not None


@pytest.mark.asyncio
async def test_mlflow_get_active_versions(mlflow_registry: MlflowModelRegistry) -> None:
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        _mv(stage="Production"),
        _mv(version="2", stage="Archived"),
    ]
    mock_client.get_run.return_value = _run()

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        result = await mlflow_registry.get_active_versions("m1")
    assert len(result) == 1


@pytest.mark.asyncio
async def test_mlflow_update_stage_not_found(mlflow_registry: MlflowModelRegistry) -> None:
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = []

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        with pytest.raises(ValueError, match="not found"):
            await mlflow_registry.update_stage("m1", "v99", "champion")
