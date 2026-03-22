"""Deep health check — verify all dependencies (DB, Redis, Kafka, models)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health check result for a single component."""

    name: str
    healthy: bool
    latency_ms: float = 0.0
    details: str = ""


@dataclass
class DeepHealthReport:
    """Aggregated health check report."""

    overall: str = "healthy"
    components: list[HealthStatus] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.overall,
            "components": [
                {
                    "name": c.name,
                    "status": "healthy" if c.healthy else "unhealthy",
                    "latency_ms": round(c.latency_ms, 2),
                    "details": c.details,
                }
                for c in self.components
            ],
        }


class HealthChecker:
    """Check health of all infrastructure dependencies."""

    async def check_all(self) -> DeepHealthReport:
        """Run all health checks."""
        from datetime import UTC, datetime  # noqa: PLC0415

        report = DeepHealthReport(timestamp=datetime.now(UTC).isoformat())

        report.components.append(await self._check_database())
        report.components.append(await self._check_redis())
        report.components.append(await self._check_kafka())
        report.components.append(await self._check_models())
        report.components.append(self._check_disk())

        unhealthy = [c for c in report.components if not c.healthy]
        if unhealthy:
            report.overall = "degraded" if len(unhealthy) < len(report.components) else "unhealthy"

        return report

    async def _check_database(self) -> HealthStatus:
        """Check PostgreSQL connectivity."""
        start = time.monotonic()
        try:
            from sqlalchemy import text  # noqa: PLC0415

            from phoenix_ml.infrastructure.persistence.database import (  # noqa: PLC0415
                get_db,
            )

            async for db in get_db():
                await db.execute(text("SELECT 1"))
                break
            latency = (time.monotonic() - start) * 1000
            return HealthStatus("postgresql", True, latency, "Connected")
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return HealthStatus(
                "postgresql", False, latency, str(e)[:100]
            )

    async def _check_redis(self) -> HealthStatus:
        """Check Redis connectivity."""
        start = time.monotonic()
        try:
            import redis.asyncio as aioredis  # noqa: PLC0415

            from phoenix_ml.config import get_settings  # noqa: PLC0415

            settings = get_settings()
            r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            pong = await r.ping()  # type: ignore[misc]
            await r.aclose()
            latency = (time.monotonic() - start) * 1000
            return HealthStatus("redis", True, latency, f"Connected (pong={pong})")
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return HealthStatus("redis", False, latency, str(e)[:100])

    async def _check_kafka(self) -> HealthStatus:
        """Check Kafka broker connectivity."""
        start = time.monotonic()
        try:
            from phoenix_ml.config import get_settings  # noqa: PLC0415

            settings = get_settings()
            # Try a simple socket connection to the kafka broker
            import asyncio  # noqa: PLC0415

            host, port_str = settings.KAFKA_URL.split(":")
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, int(port_str)), timeout=3.0
            )
            writer.close()
            await writer.wait_closed()
            latency = (time.monotonic() - start) * 1000
            return HealthStatus("kafka", True, latency, f"Broker at {settings.KAFKA_URL}")
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return HealthStatus("kafka", False, latency, str(e)[:100])

    async def _check_models(self) -> HealthStatus:
        """Check if models are loaded."""
        start = time.monotonic()
        try:
            from phoenix_ml.infrastructure.bootstrap.container import (  # noqa: PLC0415
                inference_engine,
            )

            loaded = getattr(inference_engine, "_models", {})
            count = len(loaded)
            latency = (time.monotonic() - start) * 1000
            return HealthStatus("models", True, latency, f"{count} model(s) loaded")
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return HealthStatus("models", False, latency, str(e)[:100])

    def _check_disk(self) -> HealthStatus:
        """Check disk space availability."""
        import shutil  # noqa: PLC0415

        start = time.monotonic()
        try:
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)
            latency = (time.monotonic() - start) * 1000
            healthy = free_gb > 1.0  # At least 1GB free
            return HealthStatus(
                "disk",
                healthy,
                latency,
                f"{free_gb:.1f}GB free / {total / (1024**3):.1f}GB total",
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return HealthStatus("disk", False, latency, str(e)[:100])
