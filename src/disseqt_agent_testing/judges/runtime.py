"""One private daemon event-loop thread per engine.

``judge()`` submits with ``run_coroutine_threadsafe`` and blocks (so it refuses to run on the
loop thread itself); ``ajudge()`` awaits the wrapped future. Never ``asyncio.run`` on the caller's thread, never ``nest_asyncio``. Deadlines live
*inside* the loop; cancellation (``close()``, ``KeyboardInterrupt`` while blocked, a cancelled
``ajudge``) is done in the loop that owns the task, because cancelling the cross-thread future does
not reliably stop the task on Python 3.10.
"""

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class JudgeRuntime:
    def __init__(self, name: str = "disseqt-judge-loop"):
        self._name = name
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._tasks: set[asyncio.Task[Any]] = set()  # touched only on the loop thread
        self._closed = False

    @property
    def loop_thread(self) -> threading.Thread | None:
        return self._thread

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._closed:
                raise RuntimeError("the judge engine is closed")
            if self._loop is None:
                ready = threading.Event()
                box: list[asyncio.AbstractEventLoop] = []

                def target() -> None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    box.append(loop)
                    ready.set()
                    try:
                        loop.run_forever()
                    finally:
                        loop.close()

                self._thread = threading.Thread(target=target, name=self._name, daemon=True)
                self._thread.start()
                ready.wait()
                self._loop = box[0]
            return self._loop

    def _submit(
        self, coro: Coroutine[Any, Any, T]
    ) -> tuple[concurrent.futures.Future[T], list[asyncio.Task[Any]]]:
        try:
            loop = self._ensure()
        except BaseException:
            coro.close()
            raise
        holder: list[asyncio.Task[Any]] = []

        async def runner() -> T:
            task = asyncio.current_task()
            assert task is not None
            holder.append(task)
            self._tasks.add(task)
            try:
                return await coro
            finally:
                self._tasks.discard(task)

        return asyncio.run_coroutine_threadsafe(runner(), loop), holder

    def _cancel(self, holder: list[asyncio.Task[Any]]) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(lambda: [t.cancel() for t in holder])

    def run(self, coro: Coroutine[Any, Any, T], *, grace_s: float = 5.0) -> T:
        """Run on the loop thread and block. A ``KeyboardInterrupt`` cancels the in-flight task
        in its own loop and waits (bounded) for it to finish before propagating."""
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError(
                "judge() was called on the judge engine's own loop thread, where blocking would "
                "deadlock the engine; await ajudge() there instead"
            )
        fut, holder = self._submit(coro)
        try:
            return fut.result()
        except BaseException:
            if not fut.done():
                self._cancel(holder)
                try:
                    fut.exception(timeout=grace_s)
                except BaseException:  # noqa: S110 - the original exception is what matters
                    pass
            raise

    async def arun(self, coro: Coroutine[Any, Any, T]) -> T:
        fut, holder = self._submit(coro)
        try:
            return await asyncio.wrap_future(fut)
        except asyncio.CancelledError:
            self._cancel(holder)
            raise

    def pending(self) -> int:
        """Tasks still running in the loop (for tests and ``stats()``)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return 0
        if threading.current_thread() is self._thread:
            return len(self._tasks)
        done: concurrent.futures.Future[int] = concurrent.futures.Future()
        loop.call_soon_threadsafe(lambda: done.set_result(len(self._tasks)))
        return done.result(timeout=2.0)

    def close(self, grace_s: float = 5.0) -> None:
        """Cancel in-flight tasks in-loop, wait for them, then stop the loop thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
        if loop is None or thread is None:
            return

        async def drain() -> None:
            tasks = list(self._tasks)
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.wait(tasks, timeout=grace_s)

        try:
            asyncio.run_coroutine_threadsafe(drain(), loop).result(timeout=grace_s + 1)
        except Exception:  # noqa: S110 - closing must never raise
            pass
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=grace_s)
