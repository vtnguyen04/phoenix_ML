"""Tests for SQLAlchemy ORM models and database Base."""

from phoenix_ml.infrastructure.persistence.database import Base
from phoenix_ml.infrastructure.persistence.models import (
    DriftReportORM,
    ModelORM,
    PredictionLogORM,
)


class TestModelORM:
    def test_tablename(self) -> None:
        assert ModelORM.__tablename__ == "models"

    def test_columns_exist(self) -> None:
        cols = {c.name for c in ModelORM.__table__.columns}
        assert "id" in cols
        assert "version" in cols
        assert "uri" in cols
        assert "framework" in cols
        assert "stage" in cols
        assert "metadata_json" in cols
        assert "is_active" in cols
        assert "created_at" in cols


class TestPredictionLogORM:
    def test_tablename(self) -> None:
        assert PredictionLogORM.__tablename__ == "prediction_logs"

    def test_columns_exist(self) -> None:
        cols = {c.name for c in PredictionLogORM.__table__.columns}
        assert "id" in cols
        assert "model_id" in cols
        assert "features" in cols
        assert "result" in cols
        assert "confidence" in cols
        assert "ground_truth" in cols


class TestDriftReportORM:
    def test_tablename(self) -> None:
        assert DriftReportORM.__tablename__ == "drift_reports"

    def test_columns_exist(self) -> None:
        cols = {c.name for c in DriftReportORM.__table__.columns}
        assert "model_id" in cols
        assert "feature_name" in cols
        assert "drift_detected" in cols
        assert "p_value" in cols
        assert "method" in cols


class TestDatabaseBase:
    def test_base_metadata_exists(self) -> None:
        assert Base.metadata is not None
        assert len(Base.metadata.tables) >= 3
