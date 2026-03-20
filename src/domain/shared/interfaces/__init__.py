"""
Domain shared interfaces — all domain port ABCs in one place.

ABCs:
  - MessageProducer  (from message_producer.py)
  - EventPublisher   (external event publishing)
  - CacheBackend     (cache operations)
"""

from src.domain.shared.interfaces.message_producer import MessageProducer

from abc import ABC, abstractmethod
from typing import Any


class EventPublisher(ABC):
    """Interface for publishing domain events to external systems."""

    @abstractmethod
    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Publishes an event to the specified topic."""

    @abstractmethod
    async def start(self) -> None:
        """Starts the publisher connection."""

    @abstractmethod
    async def stop(self) -> None:
        """Stops the publisher connection."""


class CacheBackend(ABC):
    """Interface for cache operations (Redis, Memcached, In-Memory)."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Retrieves a cached value by key."""

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Stores a value with optional TTL in seconds."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Removes a cached value by key."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Checks if a key exists in the cache."""


__all__ = ["CacheBackend", "EventPublisher", "MessageProducer"]
