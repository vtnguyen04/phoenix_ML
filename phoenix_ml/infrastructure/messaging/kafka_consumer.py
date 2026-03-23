"""Async Kafka consumer with event routing and retry.

Consumes JSON messages from a Kafka topic, deserializes them, and
dispatches to registered handlers based on the ``event_type`` field.

Message format:
    ``{"event_type": "<type>", ...payload}``

Retry policy:
    Failed handler calls are retried up to ``max_retries`` times with
    exponential backoff (``retry_base_delay * 2^attempt``). Exhausted
    messages are logged as dead-letter entries.

Error handling:
    - Kafka unreachable at startup: enters no-op mode (idle loop).
    - Unrecognized ``event_type``: silently skipped unless a default
      handler is set via ``set_default_handler()``.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiokafka import AIOKafkaConsumer

logger = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


class KafkaConsumer:
    """Consumes JSON messages from Kafka and dispatches to registered handlers.

    Args:
        bootstrap_servers: Kafka broker address (e.g. ``kafka:9092``).
        group_id: Consumer group ID for offset management.
        max_retries: Max retry attempts per message before dead-lettering.
        retry_base_delay: Base delay in seconds for exponential backoff.
        dlq_topic: Topic name for dead-letter messages.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        dlq_topic: str = "phoenix-dlq",
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.dlq_topic = dlq_topic

        self._consumer: AIOKafkaConsumer | None = None
        self._running = False
        self._is_noop = False

        # Event routing: event_type → list of handlers
        self._handlers: dict[str, list[EventHandler]] = {}
        self._default_handler: EventHandler | None = None

        # Stats
        self._processed_count = 0
        self._error_count = 0
        self._dlq_count = 0

    # ── Handler Registration ─────────────────────────────────────

    def register_handler(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> None:
        """Register a handler for a specific event type.

        Multiple handlers can be registered for the same event_type.
        The ``event_type`` is matched against message['event_type'].
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.info(
            "📌 Registered handler for event_type='%s': %s",
            event_type,
            handler.__name__ if hasattr(handler, "__name__") else str(handler),
        )

    def set_default_handler(self, handler: EventHandler) -> None:
        """Set fallback handler for unrecognized event types."""
        self._default_handler = handler

    # ── Consumer Lifecycle ───────────────────────────────────────

    async def start(
        self,
        topic: str,
        handler: EventHandler | None = None,
    ) -> None:
        """Start consuming from topic.

        If ``handler`` is provided, it's used as the default handler.
        Otherwise, dispatches to registered handlers based on event_type.
        """
        if handler:
            self._default_handler = handler

        if not self.bootstrap_servers:
            self._is_noop = True
            logger.info("⏭️ Kafka Consumer skipped: bootstrap_servers is empty")
            return

        self._running = True

        try:
            self._consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
            )
            await self._consumer.start()
            logger.info(
                "✅ Kafka Consumer started (topic=%s, group=%s, handlers=%d)",
                topic,
                self.group_id,
                sum(len(h) for h in self._handlers.values()),
            )
        except Exception as e:
            logger.warning(
                "⚠️ Kafka Consumer connection failed: %s. No-op mode.",
                e,
            )
            self._is_noop = True
            self._consumer = None
            while self._running:
                await asyncio.sleep(1)
            return

        try:
            async for message in self._consumer:
                if not self._running:
                    break
                await self._dispatch_with_retry(message.value, topic, message.offset)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                logger.error("❌ Kafka Consumer loop error: %s", e)
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Signal the consumer to stop and clean up."""
        self._running = False
        if self._consumer:
            await self._cleanup()

    # ── Event Dispatch with Retry ────────────────────────────────

    async def _dispatch_with_retry(
        self,
        event: dict[str, Any],
        topic: str,
        offset: int,
    ) -> None:
        """Dispatch event to handler(s) with retry and DLQ on failure."""
        event_type = event.get("event_type", "unknown")
        handlers = self._handlers.get(event_type, [])

        if not handlers and self._default_handler:
            handlers = [self._default_handler]

        if not handlers:
            logger.debug(
                "No handler for event_type='%s', skipping (offset=%d)",
                event_type,
                offset,
            )
            return

        for handler in handlers:
            success = False
            last_error: Exception | None = None

            for attempt in range(self.max_retries):
                try:
                    await handler(event)
                    success = True
                    self._processed_count += 1
                    break
                except Exception as e:
                    last_error = e
                    self._error_count += 1
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "⚠️ Handler error (attempt %d/%d, retry in %.1fs): %s",
                        attempt + 1,
                        self.max_retries,
                        delay,
                        e,
                    )
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(delay)

            if not success:
                # Send to Dead Letter Queue
                self._dlq_count += 1
                logger.error(
                    "💀 Message sent to DLQ after %d retries (topic=%s, offset=%d): %s",
                    self.max_retries,
                    topic,
                    offset,
                    last_error,
                )
                await self._send_to_dlq(event, topic, offset, str(last_error))

    async def _send_to_dlq(
        self,
        event: dict[str, Any],
        source_topic: str,
        offset: int,
        error: str,
    ) -> None:
        """Send failed message to Dead Letter Queue topic."""
        dlq_message = {
            "original_event": event,
            "source_topic": source_topic,
            "offset": offset,
            "error": error,
            "max_retries": self.max_retries,
        }
        # Log DLQ message (actual Kafka produce would go here if producer available)
        logger.error("📬 DLQ message: %s", json.dumps(dlq_message, default=str)[:500])

    # ── Stats ────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, int]:
        """Consumer processing statistics."""
        return {
            "processed": self._processed_count,
            "errors": self._error_count,
            "dlq": self._dlq_count,
            "registered_event_types": len(self._handlers),
        }

    # ── Cleanup ──────────────────────────────────────────────────

    async def _cleanup(self) -> None:
        """Commit offsets and close the consumer."""
        if self._consumer:
            try:
                await self._consumer.stop()
            except Exception as e:
                logger.warning("⚠️ Error stopping Kafka Consumer: %s", e)
            finally:
                self._consumer = None
