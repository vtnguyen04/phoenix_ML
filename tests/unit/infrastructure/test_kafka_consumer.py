"""Tests for KafkaConsumer."""

import asyncio
from typing import Any

from src.infrastructure.messaging.kafka_consumer import KafkaConsumer


async def test_start_sets_running() -> None:
    consumer = KafkaConsumer(bootstrap_servers="localhost:9092", group_id="test-group")
    assert consumer._running is False

    async def dummy_handler(msg: dict[str, Any]) -> None:
        pass

    # Start in background, then stop immediately
    task = asyncio.create_task(consumer.start("test-topic", handler=dummy_handler))
    await asyncio.sleep(0.05)
    assert consumer._running is True
    consumer.stop()
    await asyncio.sleep(0.05)
    assert consumer._running is False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_stop_sets_running_false() -> None:
    consumer = KafkaConsumer(bootstrap_servers="localhost:9092", group_id="test-group")
    consumer._running = True
    consumer.stop()
    assert consumer._running is False


def test_init_attributes() -> None:
    consumer = KafkaConsumer(bootstrap_servers="kafka:9092", group_id="ml-group")
    assert consumer.bootstrap_servers == "kafka:9092"
    assert consumer.group_id == "ml-group"
    assert consumer._running is False
