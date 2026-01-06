import asyncio
import json
from typing import Any


class KafkaProducer:
    """
    Infrastructure service for publishing events to Kafka.
    """
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        # self.producer = AIOKafkaProducer(...)

    async def publish(self, topic: str, event: Any) -> None:
        """
        Publishes an event to the specified topic.
        """
        payload = json.dumps(event, default=str).encode("utf-8")
        # await self.producer.send_and_wait(topic, payload)
        # Mock behavior
        await asyncio.sleep(0.01)
        print(f"[Kafka] Published to {topic}: {len(payload)} bytes")
