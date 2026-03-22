import json
import logging
from typing import Any

from aiokafka import AIOKafkaProducer

logger = logging.getLogger(__name__)


class KafkaProducer:
    """
    Production-ready Kafka Producer for real-time event streaming.
    Fallbacks to no-op mode if connection fails (useful for local development/test).
    """

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self._producer: AIOKafkaProducer | None = None
        self._is_noop = False

    async def start(self) -> None:
        if not self.bootstrap_servers:
            self._is_noop = True
            logger.info("⏭️ Kafka skipped: bootstrap_servers is empty")
            return
        if self._producer is None and not self._is_noop:
            try:
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                )
                await self._producer.start()
                logger.info("✅ Kafka Producer started at %s", self.bootstrap_servers)
            except Exception as e:
                logger.warning("⚠️ Kafka connection failed: %s. Falling back to no-op mode.", e)
                self._is_noop = True
                self._producer = None

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()

    async def publish(self, topic: str, event: Any) -> None:
        """
        Publishes an event to Kafka topic.
        """
        if self._is_noop:
            return

        if self._producer is None:
            await self.start()

        if self._producer:
            await self._producer.send_and_wait(topic, event)
