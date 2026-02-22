from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
from threading import Lock
from time import monotonic


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_s: int


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, *, key: str, max_requests: int, window_s: int) -> RateLimitDecision:
        now = monotonic()
        with self._lock:
            bucket = self._events.setdefault(key, deque())
            self._prune(bucket=bucket, now=now, window_s=window_s)

            if len(bucket) >= max_requests:
                oldest = bucket[0]
                retry_after = max(1, ceil((oldest + window_s) - now))
                return RateLimitDecision(allowed=False, retry_after_s=retry_after)

            bucket.append(now)
            return RateLimitDecision(allowed=True, retry_after_s=0)

    @staticmethod
    def _prune(*, bucket: deque[float], now: float, window_s: int) -> None:
        threshold = now - window_s
        while bucket and bucket[0] <= threshold:
            bucket.popleft()


class AuthRateLimiter:
    def __init__(
        self,
        *,
        enabled: bool,
        login_max_requests: int,
        login_window_s: int,
        refresh_max_requests: int,
        refresh_window_s: int,
    ) -> None:
        self.enabled = enabled
        self.login_max_requests = login_max_requests
        self.login_window_s = login_window_s
        self.refresh_max_requests = refresh_max_requests
        self.refresh_window_s = refresh_window_s
        self.backend = InMemoryRateLimiter()

    def check_login(self, *, client_ip: str, principal: str) -> RateLimitDecision:
        if not self.enabled:
            return RateLimitDecision(allowed=True, retry_after_s=0)
        key = f"auth:login:{client_ip}:{principal.lower().strip()}"
        return self.backend.allow(
            key=key,
            max_requests=self.login_max_requests,
            window_s=self.login_window_s,
        )

    def check_refresh(self, *, client_ip: str, principal: str) -> RateLimitDecision:
        if not self.enabled:
            return RateLimitDecision(allowed=True, retry_after_s=0)
        key = f"auth:refresh:{client_ip}:{principal}"
        return self.backend.allow(
            key=key,
            max_requests=self.refresh_max_requests,
            window_s=self.refresh_window_s,
        )
