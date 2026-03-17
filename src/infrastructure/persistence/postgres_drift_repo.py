from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.monitoring.entities.drift_report import DriftReport
from src.domain.monitoring.repositories.drift_report_repository import (
    DriftReportRepository,
)
from src.infrastructure.persistence.models import DriftReportORM


class PostgresDriftReportRepository(DriftReportRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, model_id: str, report: DriftReport) -> None:
        orm = DriftReportORM(
            model_id=model_id,
            feature_name=report.feature_name,
            drift_detected=report.drift_detected,
            p_value=report.p_value,
            statistic=report.statistic,
            threshold=report.threshold,
            method=report.method,
            recommendation=report.recommendation,
            sample_size=report.sample_size,
            analyzed_at=report.analyzed_at,
        )
        self._session.add(orm)
        await self._session.commit()

    async def get_history(self, model_id: str, limit: int = 100) -> list[DriftReport]:
        stmt = (
            select(DriftReportORM)
            .where(DriftReportORM.model_id == model_id)
            .order_by(DriftReportORM.analyzed_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        orms = result.scalars().all()

        return [
            DriftReport(
                feature_name=o.feature_name,
                drift_detected=o.drift_detected,
                p_value=o.p_value,
                statistic=o.statistic,
                threshold=o.threshold,
                method=o.method,
                recommendation=o.recommendation,
                sample_size=o.sample_size,
                analyzed_at=o.analyzed_at,
            )
            for o in orms
        ]
