from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.inference.entities.model import Model
from src.domain.model_registry.repositories.model_repository import ModelRepository
from src.infrastructure.persistence.models import ModelORM


class PostgresModelRegistry(ModelRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, model: Model) -> None:
        metadata = model.metadata or {}
        orm = ModelORM(
            id=model.id,
            version=model.version,
            uri=model.uri,
            framework=model.framework,
            metadata_json=metadata,
            metrics_json=metadata.get("metrics", {}),
            created_at=model.created_at,
            is_active=model.is_active,
            stage=metadata.get("role", "challenger"),
        )
        await self._session.merge(orm)
        await self._session.commit()

    async def get_by_id(self, model_id: str, version: str) -> Model | None:
        stmt = select(ModelORM).where(ModelORM.id == model_id, ModelORM.version == version)
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if not orm:
            return None
        return self._to_entity(orm)

    async def get_active_versions(self, model_id: str) -> list[Model]:
        stmt = select(ModelORM).where(ModelORM.id == model_id, ModelORM.is_active)
        result = await self._session.execute(stmt)
        return [self._to_entity(o) for o in result.scalars().all()]

    async def get_champion(self, model_id: str) -> Model | None:
        stmt = select(ModelORM).where(ModelORM.id == model_id, ModelORM.stage == "champion")
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        return self._to_entity(orm) if orm else None

    async def update_stage(self, model_id: str, version: str, stage: str) -> None:
        is_active = stage not in ("archived", "retired")

        if stage == "champion":
            # Demote old champion — also deactivate it
            stmt = (
                update(ModelORM)
                .where(ModelORM.id == model_id, ModelORM.stage == "champion")
                .values(stage="retired", is_active=False)
            )
            await self._session.execute(stmt)

        stmt = (
            update(ModelORM)
            .where(ModelORM.id == model_id, ModelORM.version == version)
            .values(stage=stage, is_active=is_active)
        )
        await self._session.execute(stmt)
        await self._session.commit()

    def _to_entity(self, orm: ModelORM) -> Model:
        metadata = dict(orm.metadata_json)
        metadata["role"] = orm.stage
        metadata["metrics"] = orm.metrics_json
        return Model(
            id=orm.id,
            version=orm.version,
            uri=orm.uri,
            framework=orm.framework,
            metadata=metadata,
            created_at=orm.created_at,
            is_active=orm.is_active,
        )
