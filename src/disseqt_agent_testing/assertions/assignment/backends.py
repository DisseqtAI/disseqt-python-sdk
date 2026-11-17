"""Assignment backends: ``python`` (vendored) and ``scipy`` (lazy, optional)."""

import importlib
import threading
from collections.abc import Callable, Sequence

from ..config import ExpectConfigError
from .hungarian import assign_max

Backend = Callable[[Sequence[Sequence[float]]], list[tuple[int, int]]]

_LOCK = threading.RLock()
_BACKENDS: dict[str, Backend] = {}


def register_assignment_backend(name: str, fn: Backend, *, override: bool = False) -> None:
    with _LOCK:
        if name in _BACKENDS and not override:
            raise ValueError(f"assignment backend {name!r} is already registered")
        _BACKENDS[name] = fn


def unregister_assignment_backend(name: str) -> None:
    with _LOCK:
        _BACKENDS.pop(name, None)


def _scipy_backend(weights: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    np = importlib.import_module("numpy")  # a SciPy dependency
    optimize = importlib.import_module("scipy.optimize")

    matrix = np.asarray(weights, dtype=float)
    if matrix.size == 0:
        return []
    rows, cols = optimize.linear_sum_assignment(matrix, maximize=True)
    return sorted((int(r), int(c)) for r, c in zip(rows, cols, strict=True))


def scipy_available() -> bool:
    try:
        importlib.import_module("scipy.optimize")
    except ImportError:
        return False
    return True


def get_backend(name: str) -> Backend:
    """``auto`` is SciPy when importable, else the vendored solver."""
    if name == "auto":
        name = "scipy" if scipy_available() else "python"
    with _LOCK:
        backend = _BACKENDS.get(name)
        known = sorted(_BACKENDS)
    if backend is None:
        raise ExpectConfigError(f"unknown assignment backend {name!r}; known: {known}")
    if name == "scipy" and not scipy_available():
        raise ExpectConfigError("assignment_backend='scipy' needs SciPy: pip install scipy")
    return backend


register_assignment_backend("python", assign_max, override=True)
register_assignment_backend("scipy", _scipy_backend, override=True)
