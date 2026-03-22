import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for the Circuit Breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_requests: int = 3


class CircuitBreaker:
    """
    Circuit Breaker pattern for fault-tolerant inference.

    States:
      CLOSED    → Normal operation. Failures increment counter.
      OPEN      → All calls routed to fallback.
      HALF_OPEN → Limited calls allowed to test recovery.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._half_open_successes = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    async def execute(
        self,
        func: Callable[[], Awaitable[Any]],
        fallback: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Execute a function with circuit breaker protection."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed > self.config.recovery_timeout:
                logger.info("Circuit transitioning OPEN → HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
            else:
                return await fallback()

        try:
            result = await func()
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            if self._state == CircuitState.OPEN:
                return await fallback()
            raise

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.config.half_open_requests:
                logger.info("Circuit transitioning HALF_OPEN → CLOSED")
                self._state = CircuitState.CLOSED
                self._failure_count = 0

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.config.failure_threshold:
            logger.warning(
                "Circuit transitioning to OPEN after %d failures",
                self._failure_count,
            )
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_successes = 0
