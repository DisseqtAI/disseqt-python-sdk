"""The network gate: a process-global, lock-protected flag (not a contextvar, because judge calls
hop to the engine's private thread). Fakes (``networked=False``) are never gated."""

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_LOCK = threading.RLock()
_OVERRIDES: list[tuple[object, bool]] = []  # (token, flag): each block removes its own entry


def is_ci() -> bool:
    return os.environ.get("CI", "").strip().lower() not in ("", "0", "false", "no")


def requests_allowed(config_allow: bool | None, cache_mode: str) -> bool:
    """Override stack, then ``JudgeConfig.allow_requests``, then ``DISSEQT_ALLOW_JUDGE_REQUESTS``,
    then the default rule: disallowed when ``CI`` is truthy and the cache mode is ``replay``."""
    with _LOCK:
        if _OVERRIDES:
            return _OVERRIDES[-1][1]
    if config_allow is not None:
        return config_allow
    env = os.environ.get("DISSEQT_ALLOW_JUDGE_REQUESTS")
    if env is not None and env.strip() != "":
        return env.strip().lower() not in ("0", "false", "no")
    return not (is_ci() and cache_mode == "replay")


@contextmanager
def allow_judge_requests(flag: bool = True) -> Iterator[None]:
    """``with allow_judge_requests(False):`` overrides the gate temporarily (nesting restores).
    The override is process-wide: while blocks from several threads overlap, the one entered last
    and still open applies to every thread."""
    entry = (object(), bool(flag))
    with _LOCK:
        _OVERRIDES.append(entry)
    try:
        yield
    finally:
        with _LOCK:
            _OVERRIDES[:] = [e for e in _OVERRIDES if e[0] is not entry[0]]
