"""
KafkaConsumer — Production-ready async Kafka consumer.

Consumes JSON messages from a Kafka topic and dispatches them to a
user-provided async handler callback. Falls back to no-op mode when
Kafka is unavailable (e.g. local development / CI).

Usage::

    consumer = KafkaConsumer("kafka:9092", group_id="ml-consumers")
    await consumer.start("inference-events", handler=my_handler)
    # ... later ...
    await consumer.stop()
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from aiokafka import AIOKafkaConsumer

logger = logging.getLogger(__name__)


class KafkaConsumer:
    """Infrastructure service for consuming events from Kafka.

    Features:
    - Graceful fallback to no-op when Kafka is unreachable
    - Automatic JSON deserialization
    - Configurable consumer group for horizontal scaling
    - Clean shutdown with commit of offsets
    """

    def __init__(self, bootstrap_servers: str, group_id: str) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer: AIOKafkaConsumer | None = None
        self._running = False
        self._is_noop = False

    async def start(
        self,
        topic: str,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Start consuming messages from *topic* and dispatch to *handler*.

        Each message value is JSON-decoded into a ``dict`` before being
        passed to the handler.  If Kafka is unreachable the consumer
        enters **no-op mode** and the coroutine idles until stopped.
        """
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
                "✅ Kafka Consumer started (topic=%s, group=%s)",
                topic,
                self.group_id,
            )
        except Exception as e:
            logger.warning(
                "⚠️ Kafka Consumer connection failed: %s. Falling back to no-op mode.",
                e,
            )
            self._is_noop = True
            self._consumer = None
            # Idle in no-op mode until stopped
            while self._running:
                await asyncio.sleep(1)
            return

        try:
            async for message in self._consumer:
                if not self._running:
                    break
                try:
                    await handler(message.value)
                except Exception as e:
                    logger.error(
                        "⚠️ Handler error for message on %s (offset=%d): %s",
                        topic,
                        message.offset,
                        e,
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                logger.error("❌ Kafka Consumer loop error: %s", e)
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Signal the consumer to stop and clean up resources."""
        self._running = False
        if self._consumer:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Commit offsets and close the underlying AIOKafkaConsumer."""
        if self._consumer:
            try:
                await self._consumer.stop()
            except Exception as e:
                logger.warning("⚠️ Error stopping Kafka Consumer: %s", e)
            finally:
                self._consumer = None
