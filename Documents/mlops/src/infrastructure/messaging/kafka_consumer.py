import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class KafkaConsumer:
    """
    Infrastructure service for consuming events from Kafka.
    """
    def __init__(self, bootstrap_servers: str, group_id: str):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._running = False

    async def start(
        self, 
        topic: str, 
        handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Starts consuming messages from the topic and processes them with the handler.
        """
        self._running = True
        print(f"[Kafka] Listening on {topic}...")
        while self._running:
            # msg = await self.consumer.getone()
            # await handler(json.loads(msg.value))
            await asyncio.sleep(1)

    def stop(self) -> None:
        self._running = False
