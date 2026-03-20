"""Tests for token bucket rate limiter."""

from __future__ import annotations

import time

from utils.rate_limiter import (
    MultiEndpointLimiter,
    RateLimiterStats,
    TokenBucketLimiter,
    rate_limit,
)


class TestTokenBucketLimiter:
    def test_allows_within_burst(self) -> None:
        limiter = TokenBucketLimiter(rate=10.0, burst=5)
        for _ in range(5):
            assert limiter.acquire()

    def test_throttles_beyond_burst(self) -> None:
        limiter = TokenBucketLimiter(rate=10.0, burst=3)
        for _ in range(3):
            limiter.acquire()
        assert not limiter.acquire()

    def test_refills_over_time(self) -> None:
        limiter = TokenBucketLimiter(rate=100.0, burst=5)
        for _ in range(5):
            limiter.acquire()
        time.sleep(0.1)
        assert limiter.acquire()

    def test_available_tokens(self) -> None:
        limiter = TokenBucketLimiter(rate=10.0, burst=10)
        assert limiter.available_tokens == 10.0
        limiter.acquire(3)
        assert limiter.available_tokens <= 7.1

    def test_stats(self) -> None:
        limiter = TokenBucketLimiter(rate=10.0, burst=2)
        limiter.acquire()
        limiter.acquire()
        limiter.acquire()
        s = limiter.stats
        assert s.allowed == 2
        assert s.throttled == 1
        assert s.total_requests == 3

    def test_reset(self) -> None:
        limiter = TokenBucketLimiter(rate=10.0, burst=5)
        for _ in range(5):
            limiter.acquire()
        assert not limiter.acquire()
        limiter.reset()
        assert limiter.acquire()

    def test_multiple_tokens(self) -> None:
        limiter = TokenBucketLimiter(rate=10.0, burst=10)
        assert limiter.acquire(5)
        assert limiter.acquire(5)
        assert not limiter.acquire(1)

    def test_acquire_blocking(self) -> None:
        limiter = TokenBucketLimiter(rate=100.0, burst=1)
        limiter.acquire()
        start = time.monotonic()
        assert limiter.acquire_blocking(timeout=1.0)
        elapsed = time.monotonic() - start
        assert elapsed < 0.5

    def test_acquire_blocking_timeout(self) -> None:
        limiter = TokenBucketLimiter(rate=0.1, burst=1)
        limiter.acquire()
        assert not limiter.acquire_blocking(timeout=0.05)


class TestMultiEndpointLimiter:
    def test_default_endpoint(self) -> None:
        limiter = MultiEndpointLimiter(default_rate=10.0, default_burst=5)
        for _ in range(5):
            assert limiter.acquire()
        assert not limiter.acquire()

    def test_registered_endpoint(self) -> None:
        limiter = MultiEndpointLimiter()
        limiter.register("api", rate=10.0, burst=2)
        assert limiter.acquire("api")
        assert limiter.acquire("api")
        assert not limiter.acquire("api")

    def test_independent_endpoints(self) -> None:
        limiter = MultiEndpointLimiter()
        limiter.register("api1", rate=10.0, burst=1)
        limiter.register("api2", rate=10.0, burst=1)
        assert limiter.acquire("api1")
        assert limiter.acquire("api2")
        assert not limiter.acquire("api1")
        assert not limiter.acquire("api2")

    def test_stats(self) -> None:
        limiter = MultiEndpointLimiter()
        limiter.register("api", rate=10.0, burst=1)
        limiter.acquire("api")
        limiter.acquire("api")
        s = limiter.stats("api")
        assert s.total_requests == 2

    def test_all_stats(self) -> None:
        limiter = MultiEndpointLimiter()
        limiter.acquire("a")
        limiter.acquire("b")
        all_s = limiter.all_stats()
        assert "a" in all_s
        assert "b" in all_s


class TestRateLimitDecorator:
    def test_decorated_function(self) -> None:
        @rate_limit(rate=100.0, burst=5)
        def my_func(x: int) -> int:
            return x * 2

        assert my_func(3) == 6

    def test_preserves_name(self) -> None:
        @rate_limit(rate=10.0, burst=5)
        def my_func() -> None:
            pass

        assert my_func.__name__ == "my_func"

    def test_limiter_attached(self) -> None:
        @rate_limit(rate=10.0, burst=5)
        def my_func() -> None:
            pass

        assert hasattr(my_func, "_limiter")
        assert isinstance(my_func._limiter, TokenBucketLimiter)


class TestRateLimiterStats:
    def test_throttle_rate(self) -> None:
        stats = RateLimiterStats(total_requests=10, allowed=7, throttled=3)
        assert abs(stats.throttle_rate - 0.3) < 1e-6

    def test_throttle_rate_zero(self) -> None:
        stats = RateLimiterStats()
        assert stats.throttle_rate == 0.0
