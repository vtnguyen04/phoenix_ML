from abc import ABC, abstractmethod
from typing import Any

class MessageProducer(ABC):
    """
    Interface for publishing messages to an event bus (e.g., Kafka, RabbitMQ).
    """
    @abstractmethod
    async def publish(self, topic: str, event: Any) -> None:
        """
        Publishes an event to a specific topic.
        """
        pass
