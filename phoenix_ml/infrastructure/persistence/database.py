from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from phoenix_ml.config import get_settings

settings = get_settings()

# Create Async Engine
engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG, future=True)

# Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def get_db_optional() -> AsyncSession | None:
    """Optional DB session — returns None when database is unavailable.

    Use this for endpoints that should work without a database
    (e.g., /predict can skip logging when DB is down).
    """
    try:
        session = AsyncSessionLocal()
        # Test the connection is actually usable
        from sqlalchemy import text  # noqa: PLC0415

        await session.execute(text("SELECT 1"))
        return session
    except Exception:
        return None

