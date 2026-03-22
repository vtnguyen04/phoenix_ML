"""
Domain Event Bus — Observer Pattern Implementation.

Decouples event producers (handlers) from consumers (metrics, logging, Kafka).
Adding a new side-effect = register a new subscriber. Zero handler code changes.
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class DomainEventBus:
    """
    Simple synchronous event bus for domain events.

    Usage:
        bus = DomainEventBus()
        bus.subscribe(PredictionCompleted, metrics_handler)
        bus.subscribe(PredictionCompleted, kafka_handler)
        bus.publish(PredictionCompleted(...))  # both handlers called
    """

    def __init__(self) -> None:
        self._subscribers: dict[type, list[Callable[..., Any]]] = defaultdict(list)

    def subscribe(self, event_type: type, handler: Callable[..., Any]) -> None:
        """Register a handler for a given event type."""
        self._subscribers[event_type].append(handler)
        logger.debug(
            "EventBus: subscribed %s to %s",
            getattr(handler, "__name__", repr(handler)),
            event_type.__name__,
        )

    def publish(self, event: Any) -> None:
        """
        Dispatch event to all registered subscribers.
        Errors in one subscriber do not block others.
        """
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "EventBus: subscriber %s failed for %s",
                    getattr(handler, "__name__", repr(handler)),
                    event_type.__name__,
                )

    @property
    def subscriber_count(self) -> int:
        return sum(len(v) for v in self._subscribers.values())
