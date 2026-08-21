"""
Base class for provider instrumentors.

Each provider (openai, anthropic, ...) subclasses `DisseqtInstrumentor` and
implements `_instrument()` / `_uninstrument()`, using `wrap_function_wrapper`
to monkey-patch target methods on the provider SDK.

Design mirrors OpenTelemetry's `BaseInstrumentor` + `wrapt`, but without the
OTel dependency — patches call DisseqtSpan APIs directly.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import wrapt

from disseqt_agentic_sdk.utils.logging import get_logger

if TYPE_CHECKING:
    from disseqt_agentic_sdk.client import DisseqtAgenticClient

logger = get_logger(__name__)


class DisseqtInstrumentor(ABC):
    """
    Abstract base for provider instrumentors.

    Subclasses declare:
      * `package_name`   — the pip package to detect (e.g. "openai").
      * `min_version`    — optional lower bound; instrumentation is skipped if
                            the installed version is older.
      * `_instrument()`  — call `self._wrap(module, name, wrapper)` for each
                            target method.
      * `_uninstrument()` — restore original methods.

    The `client` is passed to `instrument()` and captured on the instance so
    patch closures can reach it (e.g. to auto-create traces).
    """

    package_name: str = ""
    min_version: str | None = None

    def __init__(self) -> None:
        self._client: DisseqtAgenticClient | None = None
        # (module, attr, wrapper_fn) — wrapper_fn identifies our layer
        # inside a wrapt FunctionWrapper chain (see _restore_wrapped).
        self._patched: list[tuple[str, str, Any]] = []
        self._is_instrumented = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def instrument(self, client: DisseqtAgenticClient) -> bool:
        """
        Apply instrumentation. Returns True on success, False if skipped
        (package not installed, version too old, or already instrumented).
        """
        if self._is_instrumented:
            logger.debug(f"{self.package_name}: already instrumented, skipping")
            return False

        version = self._detect_version()
        if version is None:
            logger.debug(f"{self.package_name}: package not installed, skipping")
            return False

        if self.min_version and _version_lt(version, self.min_version):
            logger.warning(f"{self.package_name} {version} < required {self.min_version}, skipping")
            return False

        self._client = client
        self._version = version

        try:
            self._instrument()
            self._is_instrumented = True
            logger.info(f"Instrumented {self.package_name} {version}")
            return True
        except Exception as e:
            # Roll back any partial patches so we don't leave the SDK with
            # some methods wrapped and no way to find them later.
            logger.warning(f"Failed to instrument {self.package_name}: {e}; rolling back")
            self._unwind_patches()
            self._client = None
            return False

    def uninstrument(self) -> None:
        """Restore original methods on the provider SDK."""
        if not self._is_instrumented:
            return
        try:
            self._uninstrument()
        finally:
            self._unwind_patches()
            self._is_instrumented = False
            # Drop the client reference so wrapper closures created during
            # _instrument() (which capture `self`) no longer keep the client
            # alive. Long-running processes that instrument/uninstrument
            # repeatedly would otherwise leak clients.
            self._client = None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def _instrument(self) -> None:
        """Apply monkey-patches. Use `self._wrap(...)` for each target."""

    def _uninstrument(self) -> None:  # noqa: B027 — subclasses may override; default is a no-op.
        """Default: rely on the tracked-patches unwind in `uninstrument()`."""

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------
    def _wrap(self, module_name: str, attr: str, wrapper: Any) -> None:
        """
        Monkey-patch `module_name.attr` with `wrapper` (a wrapt-compatible
        function `wrapper(wrapped, instance, args, kwargs) -> Any`).
        """
        try:
            wrapt.wrap_function_wrapper(module_name, attr, wrapper)
        except (ImportError, AttributeError) as e:
            logger.debug(f"{self.package_name}: skip patch {module_name}.{attr}: {e}")
            return
        # Capture the wrapt FunctionWrapper we just installed so we can
        # identify our layer later even if another library stacks more
        # wrappers on top.
        try:
            module = importlib.import_module(module_name)
            installed = _get_attr(module, attr)
        except Exception:  # noqa: BLE001
            installed = None
        self._patched.append((module_name, attr, installed))

    def _unwind_patches(self) -> None:
        """Restore each tracked patch and clear the list."""
        for module_name, attr, installed in self._patched:
            try:
                module = importlib.import_module(module_name)
                _restore_wrapped(module, attr, installed, self.package_name)
            except Exception:  # noqa: BLE001
                pass
        self._patched.clear()

    @property
    def client(self) -> DisseqtAgenticClient:
        """The client this instrumentor was installed against."""
        if self._client is None:
            raise RuntimeError(f"{self.package_name} instrumentor not installed")
        return self._client

    @property
    def version(self) -> str:
        """Installed version of the instrumented package."""
        return getattr(self, "_version", "unknown")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _detect_version(self) -> str | None:
        try:
            return importlib.metadata.version(self.package_name)
        except importlib.metadata.PackageNotFoundError:
            return None


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------
def _version_lt(a: str, b: str) -> bool:
    """Loose version compare — good enough for MAJOR.MINOR.PATCH gate checks."""

    def _parts(v: str) -> tuple[int, ...]:
        parts = []
        for part in v.split(".")[:3]:
            digits = "".join(ch for ch in part if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    return _parts(a) < _parts(b)


def _get_attr(module: Any, dotted: str) -> Any:
    """
    Resolve `module.dotted` without triggering descriptor protocol on the
    leaf. Needed so we can capture the wrapt FunctionWrapper object that's
    actually stored in a class's __dict__, not the BoundFunctionWrapper that
    descriptor access would hand back.
    """
    parts = dotted.split(".")
    obj = module
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return inspect.getattr_static(obj, parts[-1])


def _restore_wrapped(module: Any, dotted: str, installed: Any, provider: str) -> None:
    """
    Splice our wrapt FunctionWrapper out of the chain at `module.dotted`.

    `installed` is the exact FunctionWrapper we put in place. We identify
    our layer by object identity, so subsequent wrappers added by other
    libraries don't confuse us.

    - If our wrapper is still on top: replace the leaf with its `__wrapped__`.
    - If our wrapper is buried under another library's wrapper: skip and
      warn. Splicing mid-chain requires rebuilding wrapt FunctionWrappers,
      which isn't safe to do blindly; leave the other library's chain intact
      and let their uninstall (or a process restart) restore the original.
    - If our wrapper is gone from the chain: nothing to do.
    """
    if installed is None:
        return
    parts = dotted.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    try:
        fn = inspect.getattr_static(parent, leaf)
    except AttributeError:
        return

    if fn is installed:
        setattr(parent, leaf, fn.__wrapped__)
        return

    node = getattr(fn, "__wrapped__", None)
    while node is not None:
        if node is installed:
            logger.warning(
                f"{provider}: cannot restore {dotted}: another library wrapped on top; "
                "leaving chain intact"
            )
            return
        node = getattr(node, "__wrapped__", None)
