"""Token bucket rate limiter for API and search requests.

Prevents overwhelming external services (embedding APIs, LLMs, databases)
with configurable burst capacity and refill rates.

Supports:
    - Token bucket algorithm (smooth rate limiting)
    - Per-endpoint rate limiting
    - Async-compatible (uses time.monotonic, no blocking)
    - Decorator for easy function wrapping

Usage::

    limiter = TokenBucketLimiter(rate=10.0, burst=20)
    if limiter.acquire():
        make_api_call()

    # Or as decorator:
    @rate_limit(rate=5.0, burst=10)
    def call_embedding_api(text):
        ...
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from threading import Lock


@dataclass
class RateLimiterStats:
    """Rate limiter statistics.

    Attributes:
        total_requests: Total acquire() calls.
        allowed: Requests that were allowed.
        throttled: Requests that were rejected.
        total_wait_seconds: Cumulative wait time.
    """

    total_requests: int = 0
    allowed: int = 0
    throttled: int = 0
    total_wait_seconds: float = 0.0

    @property
    def throttle_rate(self) -> float:
        """Fraction of requests throttled."""
        return self.throttled / self.total_requests if self.total_requests > 0 else 0.0


class TokenBucketLimiter:
    """Token bucket rate limiter.

    Allows bursts up to `burst` tokens, refilling at `rate` tokens/second.

    Args:
        rate: Tokens added per second.
        burst: Maximum token capacity (burst size).

    Example::

        limiter = TokenBucketLimiter(rate=10.0, burst=20)

        # Non-blocking
        if limiter.acquire():
            do_work()

        # Blocking (waits until token available)
        limiter.acquire_blocking()
        do_work()
    """

    def __init__(self, rate: float = 10.0, burst: int = 20) -> None:
        """Initialise with rate and burst capacity."""
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = Lock()
        self._stats = RateLimiterStats()

    def _refill(self) -> None:
        """Add tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed, False if throttled.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were available, False otherwise.
        """
        with self._lock:
            self._refill()
            self._stats.total_requests += 1

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.allowed += 1
                return True

            self._stats.throttled += 1
            return False

    def acquire_blocking(self, tokens: int = 1, timeout: float = 30.0) -> bool:
        """Wait until tokens are available. Returns False if timeout exceeded.

        Args:
            tokens: Number of tokens to consume.
            timeout: Maximum seconds to wait.

        Returns:
            True if tokens acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.acquire(tokens):
                wait_time = time.monotonic() - (deadline - timeout)
                self._stats.total_wait_seconds += max(0, wait_time)
                return True
            sleep_time = min(tokens / self._rate, deadline - time.monotonic())
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._stats.throttled += 1
        return False

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens."""
        with self._lock:
            self._refill()
            return self._tokens

    @property
    def stats(self) -> RateLimiterStats:
        """Return rate limiter statistics."""
        return self._stats

    def reset(self) -> None:
        """Reset tokens to full burst capacity."""
        with self._lock:
            self._tokens = float(self._burst)
            self._last_refill = time.monotonic()


class MultiEndpointLimiter:
    """Rate limiter with per-endpoint limits.

    Different endpoints can have different rate limits.

    Args:
        default_rate: Default rate for unregistered endpoints.
        default_burst: Default burst for unregistered endpoints.

    Example::

        limiter = MultiEndpointLimiter(default_rate=10.0)
        limiter.register("embedding_api", rate=5.0, burst=10)
        limiter.register("llm_api", rate=2.0, burst=5)

        if limiter.acquire("embedding_api"):
            call_embedding_api()
    """

    def __init__(self, default_rate: float = 10.0, default_burst: int = 20) -> None:
        """Initialise with default rate limits."""
        self._default_rate = default_rate
        self._default_burst = default_burst
        self._limiters: dict[str, TokenBucketLimiter] = {}

    def register(self, endpoint: str, rate: float, burst: int) -> None:
        """Register a rate limit for a specific endpoint.

        Args:
            endpoint: Endpoint name/identifier.
            rate: Tokens per second.
            burst: Maximum burst capacity.
        """
        self._limiters[endpoint] = TokenBucketLimiter(rate=rate, burst=burst)

    def acquire(self, endpoint: str = "default", tokens: int = 1) -> bool:
        """Acquire tokens for an endpoint.

        Args:
            endpoint: Endpoint to rate limit.
            tokens: Tokens to consume.

        Returns:
            True if allowed.
        """
        if endpoint not in self._limiters:
            self._limiters[endpoint] = TokenBucketLimiter(
                rate=self._default_rate, burst=self._default_burst
            )
        return self._limiters[endpoint].acquire(tokens)

    def stats(self, endpoint: str = "default") -> RateLimiterStats:
        """Get stats for an endpoint."""
        if endpoint in self._limiters:
            return self._limiters[endpoint].stats
        return RateLimiterStats()

    def all_stats(self) -> dict[str, RateLimiterStats]:
        """Get stats for all endpoints."""
        return {name: limiter.stats for name, limiter in self._limiters.items()}


def rate_limit(rate: float = 10.0, burst: int = 20) -> callable:
    """Decorator to rate-limit a function.

    Args:
        rate: Tokens per second.
        burst: Maximum burst capacity.

    Example::

        @rate_limit(rate=5.0, burst=10)
        def call_api(query):
            return requests.get(f"/search?q={query}")
    """
    limiter = TokenBucketLimiter(rate=rate, burst=burst)

    def decorator(func: callable) -> callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter.acquire_blocking()
            return func(*args, **kwargs)

        wrapper._limiter = limiter
        return wrapper

    return decorator
