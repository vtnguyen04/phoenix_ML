
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.inference.entities.model import Model, ModelStage
from src.domain.inference.repositories.model_repository import ModelRepository
from src.infrastructure.persistence.models import ModelORM


class PostgresModelRepository(ModelRepository):
    """
    PostgreSQL implementation of ModelRepository using SQLAlchemy Async.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, model: Model) -> None:
        orm_model = ModelORM(
            id=model.id,
            version=model.version,
            uri=model.uri,
            framework=model.framework,
            stage=model.stage.value,
            metadata_json=model.metadata,
            created_at=model.created_at,
            is_active=model.is_active
        )
        await self._session.merge(orm_model)
        await self._session.commit()

    async def get_by_id(self, model_id: str, version: str) -> Model | None:
        query = select(ModelORM).where(
            ModelORM.id == model_id, 
            ModelORM.version == version
        )
        result = await self._session.execute(query)
        orm_model = result.scalar_one_or_none()
        
        if not orm_model:
            return None
            
        return self._to_domain(orm_model)

    async def list_active_models(self) -> list[Model]:
        query = select(ModelORM).where(ModelORM.is_active == True) # noqa: E712
        result = await self._session.execute(query)
        return [self._to_domain(m) for m in result.scalars().all()]

    def _to_domain(self, orm: ModelORM) -> Model:
        return Model(
            id=orm.id,
            version=orm.version,
            uri=orm.uri,
            framework=orm.framework,
            stage=ModelStage(orm.stage),
            metadata=orm.metadata_json,
            created_at=orm.created_at,
            is_active=orm.is_active
        )
