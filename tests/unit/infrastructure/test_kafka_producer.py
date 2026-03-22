"""Tests for KafkaProducer with mocked aiokafka."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from phoenix_ml.infrastructure.messaging.kafka_producer import KafkaProducer


@pytest.fixture
def producer() -> KafkaProducer:
    return KafkaProducer(bootstrap_servers="localhost:9092")


async def test_start_creates_producer(producer: KafkaProducer) -> None:
    with patch("phoenix_ml.infrastructure.messaging.kafka_producer.AIOKafkaProducer") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.start = AsyncMock()
        mock_cls.return_value = mock_instance

        await producer.start()
        mock_instance.start.assert_called_once()
        assert producer._producer is not None


async def test_start_falls_back_to_noop_on_error(producer: KafkaProducer) -> None:
    with patch("phoenix_ml.infrastructure.messaging.kafka_producer.AIOKafkaProducer") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.start = AsyncMock(side_effect=ConnectionError("refused"))
        mock_cls.return_value = mock_instance

        await producer.start()
        assert producer._is_noop is True
        assert producer._producer is None


async def test_publish_noop_does_nothing(producer: KafkaProducer) -> None:
    producer._is_noop = True
    await producer.publish("topic", {"event": "test"})  # should not raise


async def test_publish_sends_to_kafka(producer: KafkaProducer) -> None:
    mock_prod = MagicMock()
    mock_prod.send_and_wait = AsyncMock()
    producer._producer = mock_prod

    await producer.publish("predictions", {"model_id": "m1"})
    mock_prod.send_and_wait.assert_called_once_with("predictions", {"model_id": "m1"})


async def test_stop_stops_producer(producer: KafkaProducer) -> None:
    mock_prod = MagicMock()
    mock_prod.stop = AsyncMock()
    producer._producer = mock_prod

    await producer.stop()
    mock_prod.stop.assert_called_once()


async def test_stop_noop_when_no_producer(producer: KafkaProducer) -> None:
    await producer.stop()  # should not raise
