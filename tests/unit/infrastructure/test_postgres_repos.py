"""Tests for Postgres repositories using mocked AsyncSession."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from phoenix_ml.application.commands.predict_command import PredictCommand
from phoenix_ml.domain.inference.entities.model import Model
from phoenix_ml.domain.inference.entities.prediction import Prediction
from phoenix_ml.domain.inference.value_objects.confidence_score import ConfidenceScore
from phoenix_ml.domain.monitoring.entities.drift_report import DriftReport
from phoenix_ml.infrastructure.persistence.postgres_drift_repo import PostgresDriftReportRepository
from phoenix_ml.infrastructure.persistence.postgres_log_repo import PostgresPredictionLogRepository
from phoenix_ml.infrastructure.persistence.postgres_model_registry import PostgresModelRegistry


def _mock_session() -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.execute = AsyncMock()
    session.merge = AsyncMock()
    session.add = MagicMock()
    return session


def _mock_model_orm(
    model_id: str = "m1", version: str = "v1", stage: str = "champion"
) -> MagicMock:
    orm = MagicMock()
    orm.id = model_id
    orm.version = version
    orm.uri = "s3://test"
    orm.framework = "onnx"
    orm.stage = stage
    orm.metadata_json = {"role": stage}
    orm.metrics_json = {}
    orm.created_at = datetime.now(UTC)
    orm.is_active = True
    return orm


# ── PostgresModelRegistry ──────────────────────────────────────


class TestPostgresModelRegistry:
    async def test_save(self) -> None:
        session = _mock_session()
        repo = PostgresModelRegistry(session)
        model = Model(
            id="m1",
            version="v1",
            uri="s3://test",
            framework="onnx",
            metadata={"role": "champion"},
            is_active=True,
        )
        await repo.save(model)
        session.merge.assert_called_once()
        session.commit.assert_called_once()

    async def test_get_by_id_found(self) -> None:
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = _mock_model_orm()
        session.execute.return_value = mock_result

        repo = PostgresModelRegistry(session)
        result = await repo.get_by_id("m1", "v1")
        assert result is not None
        assert result.id == "m1"

    async def test_get_by_id_not_found(self) -> None:
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        repo = PostgresModelRegistry(session)
        result = await repo.get_by_id("missing", "v1")
        assert result is None

    async def test_get_champion_not_found(self) -> None:
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        repo = PostgresModelRegistry(session)
        result = await repo.get_champion("m1")
        assert result is None

    async def test_get_active_versions(self) -> None:
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [_mock_model_orm()]
        session.execute.return_value = mock_result

        repo = PostgresModelRegistry(session)
        result = await repo.get_active_versions("m1")
        assert len(result) == 1
        assert result[0].id == "m1"

    async def test_update_stage_champion_demotes_old(self) -> None:
        session = _mock_session()
        repo = PostgresModelRegistry(session)
        await repo.update_stage("m1", "v2", "champion")
        # Two execute calls: demote old champion + promote new
        assert session.execute.call_count == 2
        session.commit.assert_called_once()

    async def test_update_stage_archived(self) -> None:
        session = _mock_session()
        repo = PostgresModelRegistry(session)
        await repo.update_stage("m1", "v1", "archived")
        session.execute.assert_called_once()
        session.commit.assert_called_once()


# ── PostgresDriftReportRepository ──────────────────────────────


class TestPostgresDriftReportRepo:
    async def test_save(self) -> None:
        session = _mock_session()
        repo = PostgresDriftReportRepository(session)
        report = DriftReport(
            feature_name="f1",
            drift_detected=True,
            p_value=0.01,
            statistic=0.3,
            threshold=0.05,
            method="ks",
            recommendation="retrain",
            sample_size=1000,
            analyzed_at=datetime.now(UTC),
        )
        await repo.save("m1", report)
        session.add.assert_called_once()
        session.commit.assert_called_once()

    async def test_get_history_empty(self) -> None:
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute.return_value = mock_result

        repo = PostgresDriftReportRepository(session)
        result = await repo.get_history("m1")
        assert result == []


# ── PostgresPredictionLogRepository ────────────────────────────


class TestPostgresLogRepo:
    async def test_log(self) -> None:
        session = _mock_session()
        repo = PostgresPredictionLogRepository(session)
        cmd = PredictCommand(model_id="m1", model_version="v1", features=[1.0])
        pred = Prediction(
            model_id="m1",
            model_version="v1",
            result=1,
            confidence=ConfidenceScore(value=0.9),
            latency_ms=5.0,
        )
        await repo.log(cmd, pred)
        session.add.assert_called_once()
        session.commit.assert_called_once()

    async def test_update_ground_truth(self) -> None:
        session = _mock_session()
        repo = PostgresPredictionLogRepository(session)
        await repo.update_ground_truth("pred-1", 1)
        session.execute.assert_called_once()
        session.commit.assert_called_once()

    async def test_get_recent_logs_empty(self) -> None:
        session = _mock_session()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute.return_value = mock_result

        repo = PostgresPredictionLogRepository(session)
        result = await repo.get_recent_logs("m1")
        assert result == []
