from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.persistence.database import Base


class ModelORM(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[str] = mapped_column(String, primary_key=True)
    uri: Mapped[str] = mapped_column(String, nullable=False)
    framework: Mapped[str] = mapped_column(String, nullable=False)
    stage: Mapped[str] = mapped_column(String, nullable=False, default="development")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default={})
    metrics_json: Mapped[dict[str, float]] = mapped_column(JSON, default={})
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class PredictionLogORM(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    model_id: Mapped[str] = mapped_column(String, nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    features: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    result: Mapped[int] = mapped_column(JSON, nullable=False)
    confidence: Mapped[float] = mapped_column(JSON, nullable=False)
    latency_ms: Mapped[float] = mapped_column(JSON, nullable=False)
    ground_truth: Mapped[int | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DriftReportORM(Base):
    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(String, nullable=False)
    feature_name: Mapped[str] = mapped_column(String, nullable=False)
    drift_detected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    p_value: Mapped[float] = mapped_column(JSON, nullable=False)
    statistic: Mapped[float] = mapped_column(JSON, nullable=False)
    threshold: Mapped[float] = mapped_column(JSON, nullable=False)
    method: Mapped[str] = mapped_column(String, nullable=False)
    recommendation: Mapped[str] = mapped_column(String, nullable=False)
    sample_size: Mapped[int] = mapped_column(JSON, nullable=False)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
