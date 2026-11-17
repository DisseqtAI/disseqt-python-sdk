"""The judge limiter: a concurrency cap and an optional rate limit around the judge calls the
runner makes. The judge client below it (plan 04) stays authoritative; pass
``judge_concurrency=0`` when it already limits."""

import asyncio
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, Protocol

from ..assertions.judge import Judge, JudgeTask, JudgeVerdict


class Limiter(Protocol):
    def acquire(self) -> None: ...

    def release(self) -> None: ...

    async def aacquire(self) -> None: ...


class JudgeLimiter:
    """Thread- and asyncio-safe. The async path polls with a short sleep, so no thread is parked
    and no loop-bound object is shared between loops."""

    def __init__(self, max_concurrency: int = 4, rps: float | None = None, burst: int = 1):
        self._sem = threading.BoundedSemaphore(max_concurrency) if max_concurrency else None
        self._rps = rps
        self._burst = max(1, burst)
        self._tokens = float(self._burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self.peak = 0
        self._active = 0
        self.max_concurrency = max_concurrency

    def _take_token(self) -> float:
        """0.0 if a token was taken, else the seconds to wait."""
        if not self._rps:
            return 0.0
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self._burst, self._tokens + (now - self._last) * self._rps)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            return (1.0 - self._tokens) / self._rps

    def _entered(self) -> None:
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)

    def acquire(self) -> None:
        if self._sem is not None:
            self._sem.acquire()
        while (wait := self._take_token()) > 0:
            time.sleep(wait)
        self._entered()

    async def aacquire(self) -> None:
        while self._sem is not None and not self._sem.acquire(blocking=False):
            await asyncio.sleep(0.005)
        while (wait := self._take_token()) > 0:
            await asyncio.sleep(wait)
        self._entered()

    def release(self) -> None:
        with self._lock:
            self._active -= 1
        if self._sem is not None:
            self._sem.release()

    @contextmanager
    def slot(self) -> Iterator[None]:
        self.acquire()
        try:
            yield
        finally:
            self.release()


class LimitedJudge:
    """Wraps a ``Judge`` so every task goes through the limiter."""

    def __init__(self, judge: Judge, limiter: Limiter):
        self._judge = judge
        self._limiter = limiter

    def judge(self, task: JudgeTask) -> JudgeVerdict:
        self._limiter.acquire()
        try:
            return self._judge.judge(task)
        finally:
            self._limiter.release()

    async def ajudge(self, task: JudgeTask) -> JudgeVerdict:
        await self._limiter.aacquire()
        try:
            return await self._judge.ajudge(task)
        finally:
            self._limiter.release()

    def eval_scope(self) -> Any:
        inner = getattr(self._judge, "eval_scope", None)
        return inner() if callable(inner) else nullcontext()
