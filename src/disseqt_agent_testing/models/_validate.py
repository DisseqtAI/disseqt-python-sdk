"""Small validation helpers shared by the models (all raise ``ModelValidationError``)."""

import math
from collections.abc import Iterable
from typing import Any, NoReturn

from ._frozen import has_surrogate
from .errors import ModelValidationError


def fail(path: str, problem: str) -> NoReturn:
    raise ModelValidationError(f"{path}: {problem}")


def set_field(obj: object, name: str, value: Any) -> None:
    """Assign a field of a frozen dataclass (used for coercion in ``__post_init__``)."""
    object.__setattr__(obj, name, value)


def check_str(value: object, path: str, *, non_empty: bool = True) -> str:
    if not isinstance(value, str):
        fail(path, f"must be a string, got {type(value).__name__}")
    if non_empty and not value.strip():
        fail(path, "must not be empty")
    if has_surrogate(value):
        fail(path, "holds a lone surrogate (not valid UTF-8)")
    return value


def check_name(value: object, path: str) -> str:
    """An authored tool or fake name: non-empty, no whitespace or control characters."""
    text = check_str(value, path)
    if any(char.isspace() or not char.isprintable() for char in text):
        fail(path, f"must not contain whitespace or control characters, got {text!r}")
    return text


def check_opt_str(value: object, path: str) -> str | None:
    return None if value is None else check_str(value, path)


def check_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        fail(path, f"must be a bool, got {type(value).__name__}")
    return value


def check_int(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(path, f"must be an int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        fail(path, f"must be >= {minimum}, got {value}")
    return value


def check_opt_int(value: object, path: str, *, minimum: int | None = None) -> int | None:
    return None if value is None else check_int(value, path, minimum=minimum)


def check_number(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    exclusive_minimum: bool = False,
    maximum: float | None = None,
) -> float:
    """A finite int or float (never a bool), returned as ``float``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(path, f"must be a number, got {type(value).__name__}")
    try:
        number = float(value)
    except OverflowError:
        fail(path, "is too large to be a finite number")
    if not math.isfinite(number):
        fail(path, f"must be finite, got {value!r}")
    if minimum is not None:
        if exclusive_minimum and number <= minimum:
            fail(path, f"must be > {minimum:g}, got {number:g}")
        if not exclusive_minimum and number < minimum:
            fail(path, f"must be >= {minimum:g}, got {number:g}")
    if maximum is not None and number > maximum:
        fail(path, f"must be <= {maximum:g}, got {number:g}")
    return number


def check_opt_number(value: object, path: str, **kwargs: Any) -> float | None:
    return None if value is None else check_number(value, path, **kwargs)


def check_choice(value: object, path: str, choices: Iterable[str]) -> str:
    options = tuple(choices)
    if not isinstance(value, str) or value not in options:
        fail(path, f"must be one of {list(options)}, got {value!r}")
    return value


def as_tuple(value: object, path: str) -> tuple[Any, ...]:
    """Accept a list or tuple (never a bare string or mapping) and return a tuple."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    fail(path, f"must be a list or tuple, got {type(value).__name__}")


def str_tuple(value: object, path: str) -> tuple[str, ...]:
    items = as_tuple(value, path)
    return tuple(check_str(item, f"{path}[{index}]") for index, item in enumerate(items))
