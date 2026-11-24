"""Immutable JSON building blocks.

``FrozenDict`` is a read-only, hashable ``dict``. ``freeze_json`` (strict) is used for everything a
test author writes; ``capture_json`` (lenient) is used when recording a real agent run, where any
Python object may appear and recording must never crash the run.
"""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, NoReturn, TypeAlias, TypeVar

from .errors import ModelValidationError

_K = TypeVar("_K")
_V = TypeVar("_V")

MAX_DEPTH = 100
CAPTURE_MAX_DEPTH = 20
CAPTURE_REPR_LIMIT = 2000


class FrozenDict(dict[_K, _V]):
    """A read-only, hashable ``dict``.

    A frozen dataclass only blocks re-assigning a field; without this, ``expect.tools[0].args["x"]
    = 1`` would still work. Hashing is valid because values must be hashable (scalars, tuples,
    ``FrozenDict``): the constructor rejects lists, dicts and sets (``freeze_json`` produces valid
    values). Limit: this stops accidental and ordinary mutation, including re-calling
    ``__init__``; Python code can still reach the C-level ``dict.update(frozen, ...)``, which no
    pure-Python class can block.
    """

    __slots__ = ("_ready",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if getattr(self, "_ready", False):
            raise TypeError("FrozenDict is read-only")  # block re-initialising to mutate
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, (list, dict, set)) and not isinstance(value, FrozenDict):
                raise TypeError(
                    f"FrozenDict values must be hashable (key {key!r} is a {type(value).__name__}); "
                    "use freeze_json to freeze nested data"
                )
        self._ready = True

    def _readonly(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("FrozenDict is read-only")

    __setitem__ = _readonly
    __delitem__ = _readonly
    __ior__ = _readonly
    clear = _readonly
    pop = _readonly
    popitem = _readonly
    setdefault = _readonly
    update = _readonly

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[Any, ...]:
        # Default dict-subclass unpickling calls __setitem__, which would raise.
        return (FrozenDict, (dict(self),))


if TYPE_CHECKING:
    JSONValue: TypeAlias = (
        None | bool | int | float | str | tuple["JSONValue", ...] | "FrozenDict[str, JSONValue]"
    )
else:
    # Runtime marker so the serializer can recognise "any JSON value" fields from type hints
    # without evaluating a recursive alias.
    JSONValue = Annotated[Any, "disseqt:json"]

JSON_MARKER = "disseqt:json"
JSON_OBJECT_MARKER = "disseqt:json-object"
MATCHER_MAP_MARKER = "disseqt:matcher-map"
MATCHER_OR_STR_MARKER = "disseqt:matcher-or-str"
MATCHER_LIST_MARKER = "disseqt:matcher-list"

# Field types: permissive for static checkers (callers may pass plain dicts and values), precise
# at runtime. The marker tells the serializer how to decode the field; ``__post_init__`` freezes
# the stored value, so what a consumer reads is always a ``FrozenDict`` / ``Matcher`` mapping.
if TYPE_CHECKING:
    JSONObject: TypeAlias = Mapping[str, Any]
    MatcherMap: TypeAlias = Mapping[str, Any]
    MatcherOrStr: TypeAlias = Any
    MatcherList: TypeAlias = tuple[Any, ...]
else:
    JSONObject = Annotated[Mapping[str, Any], JSON_OBJECT_MARKER]
    MatcherMap = Annotated[Mapping[str, Any], MATCHER_MAP_MARKER]
    MatcherOrStr = Annotated[Any, MATCHER_OR_STR_MARKER]
    MatcherList = Annotated[tuple[Any, ...], MATCHER_LIST_MARKER]


def has_surrogate(text: str) -> bool:
    """True if ``text`` holds a lone surrogate (it cannot be written out as UTF-8 JSON)."""
    if text.isascii():
        return False
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


MAX_INT_DIGITS = 4000  # str() of larger ints raises on 3.10.7+; keep authored data portable
MAX_INT_BITS = 13287  # just under 10**4000


def freeze_json(value: object, path: str = "value", *, _depth: int = 0) -> "JSONValue":
    """Deep-freeze an authored value. Strict: anything not plain JSON is rejected."""
    if _depth > MAX_DEPTH:
        raise ModelValidationError(f"{path}: nesting is deeper than {MAX_DEPTH} levels")
    if isinstance(value, str) and has_surrogate(value):
        raise ModelValidationError(f"{path}: string holds a lone surrogate (not valid UTF-8)")
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value).bit_length() > MAX_INT_BITS:
            raise ModelValidationError(f"{path}: integer has more than {MAX_INT_DIGITS} digits")
        return value
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelValidationError(f"{path}: float({str(value)!r}) is not JSON-able")
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ModelValidationError(f"{path}: key {key!r} is not a string")
            if has_surrogate(key):
                raise ModelValidationError(
                    f"{path}: a key holds a lone surrogate (not valid UTF-8)"
                )
            out[key] = freeze_json(item, f"{path}.{key}", _depth=_depth + 1)
        return FrozenDict(out)
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(item, f"{path}[{index}]", _depth=_depth + 1)
            for index, item in enumerate(value)
        )
    raise ModelValidationError(
        f"{path}: {type(value).__name__} is not JSON-able "
        "(use str, int, float, bool, None, list or dict; dates must be ISO strings)"
    )


def freeze_mapping(value: object, path: str) -> "FrozenDict[str, Any]":
    """Freeze a value that must be a JSON object (``None`` is accepted as empty)."""
    if value is None:
        return FrozenDict()
    if not isinstance(value, dict):
        raise ModelValidationError(f"{path}: expected an object, got {type(value).__name__}")
    frozen = freeze_json(value, path)
    assert isinstance(frozen, FrozenDict)
    return frozen


def thaw(value: object) -> Any:
    """Return plain ``dict`` / ``list`` copies of a frozen value (safe for callers to mutate)."""
    if isinstance(value, dict):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(item) for item in value]
    return value


def _safe_repr(value: object) -> str:
    try:
        text = repr(value)
    except Exception:  # a hostile __repr__ must not break recording
        text = f"<unrepresentable {type(value).__name__}>"
    return text[:CAPTURE_REPR_LIMIT]


def capture_json(value: object) -> "tuple[JSONValue, bool]":
    """Leniently turn any Python value into frozen JSON. Returns ``(value, lossy)``.

    JSON-able parts are kept exactly. Anything else becomes ``repr`` (truncated to 2,000
    characters); NaN and infinities become the strings ``"NaN"`` / ``"Infinity"``; integers over
    the size limit of ``freeze_json`` become ``"<int with N bits>"``; non-string dict keys are
    stringified; structures deeper than 20 levels (including cycles) are cut. Whenever
    something is changed, ``lossy`` is ``True``. This function never raises.
    """
    lossy = False

    def walk(item: object, depth: int) -> Any:
        nonlocal lossy
        if depth > CAPTURE_MAX_DEPTH:
            lossy = True
            return _safe_repr(item)
        if isinstance(item, str) and has_surrogate(item):
            lossy = True  # keep recording safe: replace what cannot be written as UTF-8
            return item.encode("utf-8", "replace").decode("utf-8")
        if item is None or isinstance(item, (bool, str)):
            return item
        if isinstance(item, int):
            bits = abs(item).bit_length()
            if bits <= MAX_INT_BITS:
                return item
            lossy = True  # freeze_json refuses it, so the step could not be built
            return f"<int with {bits} bits>"
        if isinstance(item, float):
            if math.isfinite(item):
                return item
            lossy = True
            return "NaN" if math.isnan(item) else ("Infinity" if item > 0 else "-Infinity")
        if isinstance(item, dict):
            out: dict[str, Any] = {}
            for key, sub in item.items():
                if not isinstance(key, str):
                    lossy = True
                    key = _safe_repr(key)
                elif has_surrogate(key):
                    lossy = True
                    key = key.encode("utf-8", "replace").decode("utf-8")
                out[key] = walk(sub, depth + 1)
            return FrozenDict(out)
        if isinstance(item, (list, tuple)):
            return tuple(walk(sub, depth + 1) for sub in item)
        lossy = True
        return _safe_repr(item)

    try:
        captured = walk(value, 0)
    except Exception:  # defensive: recording must never crash a run
        return "<capture failed>", True
    return captured, lossy
