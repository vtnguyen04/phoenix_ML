"""
Application Decorators — Cross-Cutting Concerns via Decorator Pattern.

Stack these decorators on handler methods to add timing, logging,
error-counting etc. without polluting business logic.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timed(operation_name: str | None = None) -> Callable[[F], F]:
    """
    Decorator that logs execution time of async handler methods.

    Usage:
        @timed("predict")
        async def execute(self, command): ...
    """

    def decorator(func: F) -> F:
        label = operation_name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start
                logger.info(
                    "⏱️ %s completed in %.3fs",
                    label,
                    elapsed,
                )
                return result
            except Exception:
                elapsed = time.time() - start
                logger.error(
                    "⏱️ %s failed after %.3fs",
                    label,
                    elapsed,
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def logged(func: F) -> F:
    """
    Decorator that logs entry and exit of async handler methods.

    Usage:
        @logged
        async def execute(self, command): ...
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__qualname__
        logger.info("→ %s called", func_name)
        try:
            result = await func(*args, **kwargs)
            logger.info("← %s returned", func_name)
            return result
        except Exception as e:
            logger.error("✗ %s raised %s: %s", func_name, type(e).__name__, e)
            raise

    return wrapper  # type: ignore[return-value]


def retry(max_retries: int = 3, backoff: float = 1.0) -> Callable[[F], F]:
    """
    Decorator that retries an async function on failure.

    Usage:
        @retry(max_retries=3, backoff=2.0)
        async def call_external_service(self): ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            import asyncio  # noqa: PLC0415

            last_exc: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries:
                        wait = backoff * attempt
                        logger.warning(
                            "🔄 %s attempt %d/%d failed, retrying in %.1fs: %s",
                            func.__qualname__,
                            attempt,
                            max_retries,
                            wait,
                            e,
                        )
                        await asyncio.sleep(wait)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
