"""``DaemonPool``: a bounded pool of **daemon** threads.

``ThreadPoolExecutor`` joins its threads at interpreter exit, so one unit stuck in a synchronous
judge call would hang pytest at exit. Daemon threads never block shutdown.
"""

import concurrent.futures
import queue
import threading
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class Executor(Protocol):
    def submit(self, fn: Callable[..., T], *args: Any) -> "concurrent.futures.Future[T]": ...

    def shutdown(self, wait: bool = False) -> None: ...


class DaemonPool:
    def __init__(self, max_workers: int):
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._max = max_workers
        self._q: queue.SimpleQueue[
            tuple[concurrent.futures.Future[Any], Callable[..., Any], tuple[Any, ...]] | None
        ] = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._closed = False

    def _spawn(self) -> None:
        t = threading.Thread(
            target=self._work, name=f"disseqt-pool-{len(self._threads)}", daemon=True
        )
        t.start()
        self._threads.append(t)

    def _work(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            fut, fn, args = item
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn(*args))
            except BaseException as exc:  # delivered to whoever waits on the future
                fut.set_exception(exc)

    def submit(self, fn: Callable[..., T], *args: Any) -> "concurrent.futures.Future[T]":
        fut: concurrent.futures.Future[T] = concurrent.futures.Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("the pool is shut down")
            if len(self._threads) < self._max:
                self._spawn()
        self._q.put((fut, fn, args))
        return fut

    def shutdown(self, wait: bool = False) -> None:
        """Cancel queued work; running units are abandoned (daemon threads end with the process)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        drained: list[Any] = []
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                item[0].cancel()
                drained.append(item)
        for _ in self._threads:
            self._q.put(None)
        if wait:
            for t in self._threads:
                t.join(timeout=2.0)

    @property
    def threads(self) -> int:
        return len(self._threads)
