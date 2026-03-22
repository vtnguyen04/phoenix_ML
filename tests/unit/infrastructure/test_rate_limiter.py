"""Tests for rate limiting middleware."""

import pytest

from phoenix_ml.infrastructure.http.middleware.rate_limit_middleware import _InMemoryRateLimiter


@pytest.fixture
def limiter() -> _InMemoryRateLimiter:
    return _InMemoryRateLimiter()


class TestInMemoryRateLimiter:
    def test_allows_requests_within_limit(self, limiter: _InMemoryRateLimiter) -> None:
        for i in range(5):
            allowed, remaining = limiter.is_allowed("client1", limit=5)
            assert allowed is True
            assert remaining == 5 - i - 1

    def test_denies_over_limit(self, limiter: _InMemoryRateLimiter) -> None:
        for _ in range(10):
            limiter.is_allowed("client2", limit=10)
        allowed, remaining = limiter.is_allowed("client2", limit=10)
        assert allowed is False
        assert remaining == 0

    def test_separate_keys_independent(self, limiter: _InMemoryRateLimiter) -> None:
        for _ in range(5):
            limiter.is_allowed("a", limit=5)

        # "a" exhausted, "b" should still work
        allowed, _ = limiter.is_allowed("b", limit=5)
        assert allowed is True

        allowed, _ = limiter.is_allowed("a", limit=5)
        assert allowed is False

    def test_limit_of_one(self, limiter: _InMemoryRateLimiter) -> None:
        allowed, remaining = limiter.is_allowed("single", limit=1)
        assert allowed is True
        assert remaining == 0

        allowed, _ = limiter.is_allowed("single", limit=1)
        assert allowed is False

    def test_large_limit(self, limiter: _InMemoryRateLimiter) -> None:
        for _ in range(999):
            limiter.is_allowed("big", limit=1000)
        allowed, remaining = limiter.is_allowed("big", limit=1000)
        assert allowed is True
        assert remaining == 0

    def test_remaining_decreases(self, limiter: _InMemoryRateLimiter) -> None:
        _, rem1 = limiter.is_allowed("dec", limit=3)
        assert rem1 == 2
        _, rem2 = limiter.is_allowed("dec", limit=3)
        assert rem2 == 1
        _, rem3 = limiter.is_allowed("dec", limit=3)
        assert rem3 == 0
