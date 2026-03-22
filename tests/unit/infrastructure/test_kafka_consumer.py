"""Tests for KafkaConsumer."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

from phoenix_ml.infrastructure.messaging.kafka_consumer import KafkaConsumer


async def test_start_sets_running() -> None:
    consumer = KafkaConsumer(bootstrap_servers="localhost:9092", group_id="test-group")
    assert consumer._running is False

    async def dummy_handler(msg: dict[str, Any]) -> None:
        pass

    # Patch AIOKafkaConsumer so it doesn't actually connect
    with patch("src.infrastructure.messaging.kafka_consumer.AIOKafkaConsumer") as MockConsumer:
        mock_instance = AsyncMock()
        # Make the consumer iterable but stop immediately
        mock_instance.__aiter__ = AsyncMock(return_value=AsyncMock())
        mock_instance.__aiter__.return_value.__anext__ = AsyncMock(
            side_effect=asyncio.CancelledError
        )
        MockConsumer.return_value = mock_instance

        task = asyncio.create_task(consumer.start("test-topic", handler=dummy_handler))
        await asyncio.sleep(0.05)
        assert consumer._running is True
        await consumer.stop()
        await asyncio.sleep(0.05)
        assert consumer._running is False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def test_stop_sets_running_false() -> None:
    consumer = KafkaConsumer(bootstrap_servers="localhost:9092", group_id="test-group")
    consumer._running = True
    await consumer.stop()
    assert consumer._running is False


def test_init_attributes() -> None:
    consumer = KafkaConsumer(bootstrap_servers="kafka:9092", group_id="ml-group")
    assert consumer.bootstrap_servers == "kafka:9092"
    assert consumer.group_id == "ml-group"
    assert consumer._running is False


async def test_noop_when_bootstrap_empty() -> None:
    """Consumer should enter no-op mode when bootstrap_servers is empty."""
    consumer = KafkaConsumer(bootstrap_servers="", group_id="test-group")

    async def dummy_handler(msg: dict[str, Any]) -> None:
        pass

    await consumer.start("test-topic", handler=dummy_handler)
    assert consumer._is_noop is True
    assert consumer._consumer is None


async def test_noop_fallback_on_connection_failure() -> None:
    """Consumer should fall back to no-op when Kafka is unreachable."""
    consumer = KafkaConsumer(bootstrap_servers="unreachable:9092", group_id="test-group")

    async def dummy_handler(msg: dict[str, Any]) -> None:
        pass

    with patch("src.infrastructure.messaging.kafka_consumer.AIOKafkaConsumer") as MockConsumer:
        mock_instance = AsyncMock()
        mock_instance.start.side_effect = Exception("Connection refused")
        MockConsumer.return_value = mock_instance

        task = asyncio.create_task(consumer.start("test-topic", handler=dummy_handler))
        await asyncio.sleep(0.1)
        assert consumer._is_noop is True
        await consumer.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
