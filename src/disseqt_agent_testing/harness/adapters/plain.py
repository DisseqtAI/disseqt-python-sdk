"""``PlainAdapter``: drive an ordinary Python agent (function, callable or object)."""

import inspect
import sys
from collections.abc import Callable
from typing import Any

from ..errors import E_CLASS, E_ENTRY, E_ENTRY_ATTR, E_FRAMEWORK, HarnessConfigError
from .base import FRAMEWORK_MODULES, BaseAdapter

CANDIDATES: tuple[str, ...] = ("arun", "run", "ainvoke", "invoke", "__call__")


def _accepts_one_positional(fn: Callable[..., Any]) -> bool:
    try:
        inspect.signature(fn).bind("text")
    except TypeError:
        return False
    except ValueError:  # builtins without a signature: assume it is fine
        return True
    return True


def is_async_callable(fn: Callable[..., Any]) -> bool:
    import functools

    while isinstance(fn, functools.partial):
        fn = fn.func
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
        getattr(fn, "__call__", None)  # noqa: B004
    )


class BoundEntry:
    """A resolved entry point: ``call(text)`` and whether it is a coroutine function."""

    __slots__ = ("call", "is_async", "label")

    def __init__(self, call: Callable[[str], Any], is_async: bool, label: str):
        self.call = call
        self.is_async = is_async
        self.label = label


def _type_name(product: Any) -> str:
    return type(product).__name__


def framework_of(product: Any) -> str | None:
    """The framework a product belongs to, without importing anything.

    Only objects from an *installed* package count (a project's own ``agents`` package does not).
    A module that is not loaded at all is treated as a framework (cannot be proven otherwise).
    """
    module = getattr(type(product), "__module__", "") or ""
    top = module.split(".")[0]
    if top not in FRAMEWORK_MODULES:
        return None
    loaded = sys.modules.get(top)
    path = str(getattr(loaded, "__file__", "") or "") if loaded is not None else ""
    if loaded is not None and path and "site-packages" not in path and "dist-packages" not in path:
        return None
    return top


class PlainAdapter(BaseAdapter):
    name = "plain"

    def matches(self, product: Any) -> bool:
        return True

    def bind_entry(self, product: Any, entry: Any) -> BoundEntry:
        if inspect.isclass(product):
            raise HarnessConfigError(E_CLASS.format(cls=product.__name__))
        cls_name = _type_name(product)
        if isinstance(entry, str):
            method = getattr(product, entry, None)
            if method is None:
                raise HarnessConfigError(E_ENTRY_ATTR.format(entry=entry, cls=cls_name))
            return BoundEntry(method, is_async_callable(method), entry)
        if callable(entry):
            return BoundEntry(lambda text: entry(product, text), is_async_callable(entry), "entry")
        framework = framework_of(product)
        if framework is not None:
            raise HarnessConfigError(
                E_FRAMEWORK.format(
                    type=f"{type(product).__module__}.{type(product).__qualname__}",
                    framework=framework,
                )
            )
        for name in CANDIDATES:
            if getattr(type(product), name, None) is None:  # class lookup: a Mock has no `run`
                continue
            method = (
                product
                if name == "__call__" and inspect.isroutine(product)
                else getattr(product, name, None)
            )
            if method is None or not callable(method):
                continue
            if not _accepts_one_positional(method):
                continue
            return BoundEntry(method, is_async_callable(method), name)
        raise HarnessConfigError(E_ENTRY.format(cls=cls_name, tried=", ".join(CANDIDATES)))
