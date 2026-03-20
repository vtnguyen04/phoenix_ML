"""
Integration Test: Circuit Breaker state transitions.

Tests the full lifecycle: CLOSED → OPEN → HALF_OPEN → CLOSED by injecting
failures into async callables, verifying fallback routing and recovery.
"""

import pytest

from src.domain.inference.services.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)


async def _success() -> str:
    return "ok"


async def _failure() -> str:
    raise RuntimeError("service down")


async def _fallback() -> str:
    return "fallback"


@pytest.fixture
def breaker() -> CircuitBreaker:
    """A circuit breaker that trips after 3 failures, recovers after 0.1s."""
    return CircuitBreaker(
        CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.1,
            half_open_requests=2,
        )
    )


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker state machine."""

    async def test_closed_state_passes_through(self, breaker: CircuitBreaker) -> None:
        """In CLOSED state, successful calls pass through normally."""
        result = await breaker.execute(_success, _fallback)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    async def test_failures_trip_circuit_to_open(self, breaker: CircuitBreaker) -> None:
        """After threshold failures, circuit trips to OPEN."""
        for i in range(3):
            if i < 2:
                # First failures still raise (circuit not yet OPEN)
                with pytest.raises(RuntimeError):
                    await breaker.execute(_failure, _fallback)
            else:
                # On the 3rd failure the circuit trips and returns fallback
                result = await breaker.execute(_failure, _fallback)
                assert result == "fallback"

        assert breaker.state == CircuitState.OPEN

    async def test_open_state_routes_to_fallback(self, breaker: CircuitBreaker) -> None:
        """While OPEN, all calls go to fallback without calling the function."""
        # Trip the breaker
        for _ in range(3):
            try:
                await breaker.execute(_failure, _fallback)
            except RuntimeError:
                pass

        assert breaker.state == CircuitState.OPEN

        # Now all calls should go to fallback
        for _ in range(5):
            result = await breaker.execute(_success, _fallback)
            assert result == "fallback"

    async def test_open_to_half_open_after_timeout(self, breaker: CircuitBreaker) -> None:
        """After recovery_timeout, circuit transitions OPEN → HALF_OPEN."""
        import asyncio  # noqa: PLC0415

        # Trip the breaker
        for _ in range(3):
            try:
                await breaker.execute(_failure, _fallback)
            except RuntimeError:
                pass

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next call should transition to HALF_OPEN and succeed
        result = await breaker.execute(_success, _fallback)
        assert result == "ok"
        assert breaker.state == CircuitState.HALF_OPEN  # type: ignore[comparison-overlap]

    async def test_half_open_to_closed_on_successes(self, breaker: CircuitBreaker) -> None:
        """Enough successes in HALF_OPEN → CLOSED."""
        import asyncio  # noqa: PLC0415

        # Trip → wait → half_open
        for _ in range(3):
            try:
                await breaker.execute(_failure, _fallback)
            except RuntimeError:
                pass

        await asyncio.sleep(0.15)

        # Execute half_open_requests (2) successes
        for _ in range(2):
            await breaker.execute(_success, _fallback)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    async def test_reset_clears_state(self, breaker: CircuitBreaker) -> None:
        """Manual reset restores CLOSED state."""
        # Add some failures
        for _ in range(2):
            try:
                await breaker.execute(_failure, _fallback)
            except RuntimeError:
                pass

        assert breaker.failure_count == 2

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
