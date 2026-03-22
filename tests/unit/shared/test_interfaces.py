"""Tests for shared abstract interfaces (EventPublisher, CacheBackend, MessageProducer)."""

import pytest

from phoenix_ml.domain.shared.interfaces import CacheBackend, EventPublisher
from phoenix_ml.domain.shared.interfaces.message_producer import MessageProducer


class TestEventPublisherInterface:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            EventPublisher()  # type: ignore[abstract]


class TestCacheBackendInterface:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            CacheBackend()  # type: ignore[abstract]


class TestMessageProducerInterface:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            MessageProducer()  # type: ignore[abstract]
