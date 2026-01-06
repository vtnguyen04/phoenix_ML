import json
import logging
from typing import Any

from aiokafka import AIOKafkaProducer

logger = logging.getLogger(__name__)

class KafkaProducer:
    """
    Production-ready Kafka Producer for real-time event streaming.
    """
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        if self._producer is None:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8")
            )
            await self._producer.start()
            logger.info(f"✅ Kafka Producer started at {self.bootstrap_servers}")

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()

    async def publish(self, topic: str, event: Any) -> None:
        """
        Publishes an event to Kafka topic.
        """
        if self._producer is None:
            await self.start()
        
        if self._producer:
            await self._producer.send_and_wait(topic, event)