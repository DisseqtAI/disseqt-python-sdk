"""Agent-factory resolution: ``"module:attr"`` imports and seed binding."""

import functools
import importlib
import inspect
from collections.abc import Callable
from typing import Any


class FactoryImportError(Exception):
    def __init__(self, path: str, exc: BaseException):
        super().__init__(f"{type(exc).__name__}: {exc}")
        self.path = path
        self.exc_type = type(exc).__name__
        self.detail = str(exc)


def import_factory(path: str) -> Callable[..., Any]:
    """Import ``module:attr`` (``attr`` may be dotted). Raises ``FactoryImportError``."""
    module_name, _, attr = path.partition(":")
    try:
        target: Any = importlib.import_module(module_name)
        for part in attr.split("."):
            target = getattr(target, part)
    except Exception as exc:  # ImportError, AttributeError, or an error raised by the module itself
        raise FactoryImportError(path, exc) from None
    if not callable(target):
        raise FactoryImportError(path, TypeError(f"{path} is not callable"))
    return target  # type: ignore[no-any-return]


def takes_seed(factory: Callable[..., Any]) -> bool:
    """A factory that declares a ``seed`` parameter gets one per repeat."""
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False
    p = params.get("seed")
    return p is not None and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)


def bind_seed(factory: Callable[..., Any], seed: int) -> Callable[..., Any]:
    """``functools.partial(factory, seed=seed)``: the harness drops bound keywords from injection."""
    return functools.partial(factory, seed=seed)
