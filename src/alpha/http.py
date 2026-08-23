"""Rate-limited, retrying HTTP client.

Every outbound request in this system goes through :class:`HttpClient`. Public
market-data APIs ban aggressively and silently: GeckoTerminal's free tier caps
at 30 requests/minute and returns ``403`` (not ``429``) both when you exceed it
and when you send a default ``python-urllib`` User-Agent. Centralising the
transport means a single place enforces politeness, so a bug in one collector
cannot get the whole system blocked.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

log = logging.getLogger(__name__)

# A browser-like UA is required: GeckoTerminal 403s the urllib/requests defaults.
DEFAULT_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) alpha-research/0.1"


class RateLimitExceeded(RuntimeError):
    """Raised when a host keeps refusing us after the retry budget is spent."""


@dataclass
class TokenBucket:
    """Thread-safe token bucket.

    ``rate`` tokens are added per second up to ``capacity``. Callers block in
    :meth:`acquire` until a token is available, so bursts are allowed up to
    ``capacity`` but the long-run average never exceeds ``rate``.
    """

    rate: float
    capacity: float
    _tokens: float = field(init=False)
    _last: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rate <= 0 or self.capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self._tokens = float(self.capacity)
        self._last = time.monotonic()

    def acquire(self, tokens: float = 1.0, timeout: float | None = None) -> float:
        """Block until ``tokens`` are available. Returns seconds spent waiting."""
        if tokens > self.capacity:
            raise ValueError(f"requested {tokens} tokens > capacity {self.capacity}")
        deadline = None if timeout is None else time.monotonic() + timeout
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                sleep_for = deficit / self.rate
            if deadline is not None and time.monotonic() + sleep_for > deadline:
                raise TimeoutError(f"rate limit wait exceeded timeout ({timeout}s)")
            # Sleep outside the lock so other threads can make progress.
            time.sleep(min(sleep_for, 0.25))
            waited += min(sleep_for, 0.25)


class HttpClient:
    """HTTP client with per-host rate limiting and bounded retries.

    Retries are attempted on connection errors, timeouts, ``429`` and ``5xx``.
    A ``Retry-After`` header is always honoured when present. ``403`` is treated
    as retryable *once* with a long backoff, because GeckoTerminal uses it as a
    soft rate-limit signal rather than a hard authorization failure.
    """

    RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 522, 524})

    def __init__(
        self,
        *,
        requests_per_minute: float = 25.0,
        burst: int = 5,
        max_retries: int = 4,
        timeout: float = 25.0,
        user_agent: str = DEFAULT_UA,
        backoff_base: float = 1.6,
    ) -> None:
        self.bucket = TokenBucket(rate=requests_per_minute / 60.0, capacity=burst)
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_base = backoff_base
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        # Observability: cheap counters make rate-limit problems obvious in logs.
        self.stats: dict[str, int] = {"requests": 0, "retries": 0, "failures": 0, "rate_limited": 0}

    def get_json(self, url: str, params: dict[str, Any] | None = None, **kw: Any) -> Any | None:
        """GET ``url`` and parse JSON. Returns ``None`` on 404 or exhausted retries."""
        resp = self.request("GET", url, params=params, **kw)
        if resp is None:
            return None
        try:
            return resp.json()
        except ValueError:
            log.warning("non-JSON response from %s: %s", url, resp.text[:200])
            return None

    def request(self, method: str, url: str, **kw: Any) -> requests.Response | None:
        kw.setdefault("timeout", self.timeout)
        last_exc: Exception | None = None
        forbidden_retries = 0

        for attempt in range(self.max_retries + 1):
            self.bucket.acquire()
            self.stats["requests"] += 1
            try:
                resp = self.session.request(method, url, **kw)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                log.debug("network error %s (attempt %d): %s", url, attempt, exc)
            else:
                if resp.status_code == 404:
                    return None
                if resp.ok:
                    return resp

                # 403 from these APIs usually means "slow down", not "no access".
                retryable = resp.status_code in self.RETRYABLE_STATUS
                if resp.status_code == 403 and forbidden_retries == 0:
                    retryable = True
                    forbidden_retries += 1
                    self.stats["rate_limited"] += 1
                    log.warning("403 from %s — backing off hard (soft rate limit)", url)
                if resp.status_code == 429:
                    self.stats["rate_limited"] += 1

                if not retryable:
                    log.warning("non-retryable %s from %s: %s", resp.status_code, url, resp.text[:200])
                    self.stats["failures"] += 1
                    return None

                delay = self._retry_after(resp)
                if delay is None:
                    delay = self._backoff(attempt, hard=resp.status_code in (403, 429))
                if attempt < self.max_retries:
                    self.stats["retries"] += 1
                    time.sleep(delay)
                continue

            if attempt < self.max_retries:
                self.stats["retries"] += 1
                time.sleep(self._backoff(attempt))

        self.stats["failures"] += 1
        log.error("giving up on %s after %d attempts (%s)", url, self.max_retries + 1, last_exc)
        return None

    def _backoff(self, attempt: int, *, hard: bool = False) -> float:
        """Exponential backoff with full jitter; ``hard`` adds a floor for rate limits."""
        base = self.backoff_base ** attempt
        delay = random.uniform(0, base)
        if hard:
            delay = max(delay, 5.0 * (attempt + 1))
        return min(delay, 60.0)

    @staticmethod
    def _retry_after(resp: requests.Response) -> float | None:
        raw = resp.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return min(float(raw), 60.0)
        except ValueError:
            return None

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
