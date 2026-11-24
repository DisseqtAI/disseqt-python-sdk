"""Metric registry: ``register_metric`` plus entry-point loading (``disseqt_agent_testing.metrics``)."""

import threading
from collections.abc import Callable
from importlib import metadata
from typing import TypeVar

from .base import BaseMetric

_LOCK = threading.RLock()
_METRICS: dict[str, BaseMetric] = {}
_ENTRY_POINTS_LOADED = False
_ENTRY_POINTS_LOADING = False
ENTRY_POINT_GROUP = "disseqt_agent_testing.metrics"
T = TypeVar("T", bound=type[BaseMetric])


def register_metric(name: str, version: str, *, override: bool = False) -> Callable[[T], T]:
    """Class decorator: ``@register_metric("my_metric", "1")``. Duplicates raise unless
    ``override=True``."""

    def decorate(cls: T) -> T:
        cls.name = name
        cls.version = version
        with _LOCK:
            if name in _METRICS and not override:
                raise ValueError(f"metric {name!r} is already registered")
            _METRICS[name] = cls()
        return cls

    return decorate


def unregister_metric(name: str) -> None:
    with _LOCK:
        _METRICS.pop(name, None)


def _load_entry_points() -> None:
    """Load plugin metrics once. The lock is held for the whole load, so another thread waits
    for the complete registry instead of planning without the plugins."""
    global _ENTRY_POINTS_LOADED, _ENTRY_POINTS_LOADING
    if _ENTRY_POINTS_LOADED:
        return
    with _LOCK:
        # the lock is re-entrant: a plugin that reads the registry while it is being imported
        # must not start a second load
        if _ENTRY_POINTS_LOADED or _ENTRY_POINTS_LOADING:
            return
        _ENTRY_POINTS_LOADING = True
        try:
            try:
                eps = metadata.entry_points(group=ENTRY_POINT_GROUP)
            except Exception:  # a broken environment must not break evaluation
                return
            for ep in eps:
                try:
                    ep.load()  # importing registers the metric through its decorator
                except Exception:  # a broken third-party metric is skipped, not fatal
                    continue
        finally:
            _ENTRY_POINTS_LOADING = False
            _ENTRY_POINTS_LOADED = True


def get_metric(name: str) -> BaseMetric:
    _load_entry_points()
    with _LOCK:
        metric = _METRICS.get(name)
        known = sorted(_METRICS)
    if metric is None:
        raise KeyError(f"unknown metric {name!r}; registered: {known}")
    return metric


def list_metrics() -> tuple[BaseMetric, ...]:
    """All registered metrics, ordered by ``rank`` then name."""
    _load_entry_points()
    with _LOCK:
        return tuple(sorted(_METRICS.values(), key=lambda m: (m.rank, m.name)))
