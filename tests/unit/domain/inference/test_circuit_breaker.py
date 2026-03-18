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
    return "fallback_result"


@pytest.mark.asyncio
async def test_closed_state_on_success() -> None:
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
    result = await cb.execute(_success, _fallback)
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_opens_after_threshold_failures() -> None:
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60))

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cb.execute(_failure, _fallback)

    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 2

    result = await cb.execute(_failure, _fallback)
    assert cb.state == CircuitState.OPEN
    assert result == "fallback_result"


@pytest.mark.asyncio
async def test_open_returns_fallback() -> None:
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, recovery_timeout=999))

    result = await cb.execute(_failure, _fallback)
    assert cb.state == CircuitState.OPEN

    result = await cb.execute(_success, _fallback)
    assert result == "fallback_result"


@pytest.mark.asyncio
async def test_half_open_recovery() -> None:
    cb = CircuitBreaker(
        CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0, half_open_requests=2
        )
    )

    await cb.execute(_failure, _fallback)
    assert cb.state == CircuitState.OPEN

    result = await cb.execute(_success, _fallback)
    assert cb.state == CircuitState.HALF_OPEN
    assert result == "ok"

    result = await cb.execute(_success, _fallback)
    assert cb.state == CircuitState.CLOSED
    assert result == "ok"


@pytest.mark.asyncio
async def test_reset() -> None:
    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
    await cb.execute(_failure, _fallback)
    assert cb.state == CircuitState.OPEN

    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
