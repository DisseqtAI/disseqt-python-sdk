"""Generic, type-hint driven ``to_dict`` / ``from_dict`` for every model.

Encoding writes every field in declaration order (classes may opt into omitting defaults with
``SERDE_OMIT_DEFAULTS``). Decoding is strict (no coercion except ``int`` for ``float`` fields and
YAML-style integer keys) and never imports or executes anything. A new model field needs no change
here as long as its type is one of the supported hint shapes (``check_supported_hints`` walks them).
"""

import dataclasses
import functools
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Literal, TypeVar, Union, get_args, get_origin

from ._frozen import (
    JSON_MARKER,
    JSON_OBJECT_MARKER,
    MATCHER_LIST_MARKER,
    MATCHER_MAP_MARKER,
    MATCHER_OR_STR_MARKER,
    FrozenDict,
    freeze_json,
)
from ._migrate import upgrade
from ._registry import matcher_class, matcher_tag
from .errors import ModelValidationError
from .matchers import Eq, Matcher

T = TypeVar("T")
UNKNOWN_POLICIES = ("error", "extra", "ignore")
_NONE_TYPE = type(None)


# Hostile or corrupt files can nest matchers and blocks arbitrarily deep; fail with a clear error
# long before Python's recursion limit turns it into a RecursionError.
MAX_DECODE_DEPTH = 64


@dataclass
class _Ctx:
    unknown: str
    fill_ids: bool
    warnings: list[str] = field(default_factory=list)
    call_counter: int = 0
    depth: int = 0
    used_ids: set[str] = field(default_factory=set)


# ----------------------------------------------------------------------------- encoding


def to_dict(obj: object) -> dict[str, Any]:
    """Encode a model (dataclass instance) as plain JSON-compatible data."""
    if isinstance(obj, Matcher):
        tagged = _encode_matcher(obj, shorthand=False)  # a standalone matcher is always tagged
        assert isinstance(tagged, dict)
        return tagged
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"to_dict expects a model instance, got {type(obj).__name__}")
    encoded = _encode(obj)
    assert isinstance(encoded, dict)
    return encoded


def _encode(value: object, *, shorthand: bool = True) -> Any:
    if isinstance(value, Matcher):
        return _encode_matcher(value, shorthand=shorthand)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _encode_dataclass(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _encode(item, shorthand=shorthand) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, shorthand=shorthand) for item in value]
    return value


def _has_marker(tp: Any, marker: str) -> bool:
    if get_origin(tp) is Annotated:
        base, found = get_args(tp)[:2]
        return bool(found == marker) or _has_marker(base, marker)
    return any(_has_marker(arg, marker) for arg in get_args(tp))


@functools.cache
def _matcher_or_str_fields(cls: type) -> frozenset[str]:
    """Fields where a bare string means "contains this text", so ``Eq`` must be written tagged."""
    return frozenset(
        name for name, hint in _hints(cls).items() if _has_marker(hint, MATCHER_OR_STR_MARKER)
    )


def _holds_tag_key(value: object) -> bool:
    """True if any object inside ``value`` (at any depth) has a ``"$m"`` key."""
    if isinstance(value, Mapping):
        return "$m" in value or any(_holds_tag_key(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_tag_key(item) for item in value)
    return False


def _is_default(f: dataclasses.Field[Any], value: object) -> bool:
    if f.default is not dataclasses.MISSING:
        return bool(value == f.default)
    if f.default_factory is not dataclasses.MISSING:
        return bool(value == f.default_factory())
    return False


def _encode_dataclass(obj: Any) -> dict[str, Any]:
    cls: type = type(obj)
    omit = bool(getattr(cls, "SERDE_OMIT_DEFAULTS", False))
    always: tuple[str, ...] = getattr(cls, "SERDE_ALWAYS", ())
    out: dict[str, Any] = {}
    version = getattr(cls, "SCHEMA_VERSION", None)
    if version is not None:
        out["schema_version"] = version
    tagged_only = _matcher_or_str_fields(cls)
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if omit and f.name not in always and _is_default(f, value):
            continue
        out[f.name] = _encode(value, shorthand=f.name not in tagged_only)
    return out


def _encode_matcher(matcher: Matcher, *, shorthand: bool = True) -> Any:
    tag = matcher_tag(type(matcher))
    if tag is None:
        raise TypeError(f"matcher class {type(matcher).__name__} is not registered")
    if shorthand and isinstance(matcher, Eq) and matcher.coerce_numeric:
        if not _holds_tag_key(matcher.value):
            return _encode(matcher.value)  # lossless shorthand: a plain JSON value
    out: dict[str, Any] = {"$m": tag}
    for f in dataclasses.fields(matcher):
        out[f.name] = _encode(getattr(matcher, f.name))
    return out


# ----------------------------------------------------------------------------- decoding


def from_dict(
    cls: type[T],
    data: Mapping[str, Any],
    *,
    unknown: str = "error",
    fill_ids: bool = False,
    warnings: list[str] | None = None,
) -> T:
    """Decode ``data`` into ``cls``.

    ``unknown`` controls unknown fields: ``"error"`` (default, catches typos in authored files),
    ``"extra"`` (move them into the model's ``extra``), ``"ignore"`` (drop them; each drop is
    appended to ``warnings`` if given). ``fill_ids=True`` fills a missing ``MsgToolCall.id`` as
    ``call_<n>`` (``n`` = position among all calls in ``data``). Versioned documents are upgraded
    through ``_migrate`` first. Raises ``ModelValidationError`` with a dotted path.
    """
    if unknown not in UNKNOWN_POLICIES:
        raise ValueError(f"unknown must be one of {list(UNKNOWN_POLICIES)}, got {unknown!r}")
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"from_dict expects a model class, got {cls!r}")
    ctx = _Ctx(unknown=unknown, fill_ids=fill_ids)
    if fill_ids:
        ctx.used_ids = _explicit_ids(data)
    if isinstance(cls, type) and issubclass(cls, Matcher):
        matcher = _decode_matcher(data, cls.__name__, ctx)
        if not isinstance(matcher, cls):
            raise _fail(cls.__name__, f"data decodes to {type(matcher).__name__}")
        result: Any = matcher
    else:
        result = _decode_dataclass(cls, data, cls.__name__, ctx)
    if warnings is not None:
        warnings.extend(ctx.warnings)
    return result  # type: ignore[no-any-return]


def _explicit_ids(data: object, _depth: int = 0) -> set[str]:
    """Every string stored under an ``id`` key anywhere in ``data`` (a safe superset)."""
    found: set[str] = set()
    if _depth > 200:
        return found
    if isinstance(data, Mapping):
        for key, value in data.items():
            if key == "id" and isinstance(value, str):
                found.add(value)
            else:
                found |= _explicit_ids(value, _depth + 1)
    elif isinstance(data, (list, tuple)):
        for item in data:
            found |= _explicit_ids(item, _depth + 1)
    return found


@functools.cache
def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls, include_extras=True)


def _fail(path: str, problem: str) -> ModelValidationError:
    return ModelValidationError(f"{path}: {problem}")


def _type_name(value: object) -> str:
    return "null" if value is None else type(value).__name__


def _decode_dataclass(cls: type, data: object, path: str, ctx: _Ctx) -> Any:
    if not isinstance(data, Mapping):
        raise _fail(path, f"expected an object, got {_type_name(data)}")
    body = dict(data)
    version = getattr(cls, "SCHEMA_VERSION", None)
    if version is not None:
        body = upgrade(cls.__name__, body, version)
        body.pop("schema_version", None)
    fill = getattr(cls, "SERDE_FILL_ID", None)
    if fill is not None:
        ctx.call_counter += 1
        field_name, template = fill
        if ctx.fill_ids and field_name not in body:
            number = ctx.call_counter
            while template.format(n=number) in ctx.used_ids:  # never reuse an explicit id
                number += 1
            body[field_name] = template.format(n=number)
            ctx.used_ids.add(body[field_name])

    fields = [f for f in dataclasses.fields(cls) if f.init]
    names = {f.name for f in fields}
    unknown = [key for key in body if key not in names]
    if unknown:
        if ctx.unknown == "error":
            raise _fail(
                path,
                f"unknown field(s) {sorted(map(str, unknown))}; valid: {sorted(names)}",
            )
        if ctx.unknown == "extra":
            if "extra" not in names:
                raise _fail(
                    path,
                    f"unknown field(s) {sorted(map(str, unknown))} and the model has no 'extra'",
                )
            moved = {key: body.pop(key) for key in unknown}
            existing = body.get("extra", {})
            if not isinstance(existing, Mapping):
                raise _fail(f"{path}.extra", f"expected an object, got {_type_name(existing)}")
            body["extra"] = {**existing, **moved}
        else:
            for key in unknown:
                body.pop(key)
            ctx.warnings.append(f"{path}: ignored unknown field(s) {sorted(map(str, unknown))}")

    hints = _hints(cls)
    none_ok: tuple[str, ...] = getattr(cls, "SERDE_NONE_AS_DEFAULT", ())
    kwargs: dict[str, Any] = {}
    for f in fields:
        if f.name in body and body[f.name] is None and f.name in none_ok:
            continue  # e.g. Message.content: null loads as the default ""
        if f.name in body:
            kwargs[f.name] = _decode(hints[f.name], body[f.name], f"{path}.{f.name}", ctx)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise _fail(f"{path}.{f.name}", "required field is missing")
    return cls(**kwargs)


def _decode(tp: Any, value: Any, path: str, ctx: _Ctx) -> Any:
    ctx.depth += 1
    try:
        if ctx.depth > MAX_DECODE_DEPTH:
            raise _fail(path[:200], f"nesting is deeper than {MAX_DECODE_DEPTH} levels")
        return _decode_value(tp, value, path, ctx)
    finally:
        ctx.depth -= 1


def _decode_value(tp: Any, value: Any, path: str, ctx: _Ctx) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)
    if tp is Any or tp is object:
        return freeze_json(value, path)
    if origin is Annotated:
        return _decode_annotated(args[0], args[1], value, path, ctx)
    if tp is None or tp is _NONE_TYPE:
        if value is not None:
            raise _fail(path, f"expected null, got {_type_name(value)}")
        return None
    if origin is Union or origin is types.UnionType:
        return _decode_union(args, value, path, ctx)
    if origin is Literal:
        if value not in args or isinstance(value, bool) != isinstance(
            args[args.index(value)], bool
        ):
            raise _fail(path, f"must be one of {list(args)}, got {value!r}")
        return value
    if origin is tuple:
        return _decode_tuple(args, value, path, ctx)
    if origin in (Mapping, dict, FrozenDict):
        return _decode_mapping(args, value, path, ctx)
    if isinstance(tp, type):
        if tp is str:
            if not isinstance(value, str):
                raise _fail(path, f"expected a string, got {_type_name(value)}")
            return value
        if tp is bool:
            if not isinstance(value, bool):
                raise _fail(path, f"expected a bool, got {_type_name(value)}")
            return value
        if tp is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise _fail(path, f"expected an int, got {_type_name(value)}")
            return value
        if tp is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _fail(path, f"expected a number, got {_type_name(value)}")
            return float(value)
        if issubclass(tp, Enum):
            try:
                return tp(value)
            except ValueError:
                raise _fail(
                    path, f"must be one of {[m.value for m in tp]}, got {value!r}"
                ) from None
        if issubclass(tp, Matcher):
            return _decode_matcher(value, path, ctx)
        if dataclasses.is_dataclass(tp):
            return _decode_dataclass(tp, value, path, ctx)
    raise TypeError(f"unsupported type hint {tp!r} at {path}")


def _decode_annotated(base: Any, marker: Any, value: Any, path: str, ctx: _Ctx) -> Any:
    if marker == JSON_MARKER:
        return freeze_json(value, path)
    if marker == JSON_OBJECT_MARKER:
        if not isinstance(value, Mapping):
            raise _fail(path, f"expected an object, got {_type_name(value)}")
        return freeze_json(dict(value), path)
    if marker == MATCHER_MAP_MARKER:
        if not isinstance(value, Mapping):
            raise _fail(path, f"expected an object of matchers, got {_type_name(value)}")
        return FrozenDict(
            {key: _decode_matcher(item, f"{path}.{key}", ctx) for key, item in value.items()}
        )
    if marker == MATCHER_LIST_MARKER:
        if not isinstance(value, (list, tuple)):
            raise _fail(path, f"expected a list of matchers, got {_type_name(value)}")
        return tuple(_decode_matcher(item, f"{path}[{i}]", ctx) for i, item in enumerate(value))
    if marker == MATCHER_OR_STR_MARKER:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping) and "$m" in value:
            return _decode_matcher(value, path, ctx)
        raise _fail(path, "must be a string or a {'$m': ...} matcher")
    return _decode(base, value, path, ctx)


def _decode_union(args: tuple[Any, ...], value: Any, path: str, ctx: _Ctx) -> Any:
    if value is None and _NONE_TYPE in args:
        return None
    members = [arg for arg in args if arg is not _NONE_TYPE]
    if len(members) == 1:
        return _decode(members[0], value, path, ctx)
    candidates = [member for member in members if _fits(member, value)] or members
    problems: list[str] = []
    for member in candidates:
        counter, used = ctx.call_counter, set(ctx.used_ids)
        try:
            return _decode(member, value, path, ctx)
        except ModelValidationError as exc:
            ctx.call_counter, ctx.used_ids = counter, used  # a failed attempt shifts no ids
            problems.append(str(exc))
    if len(problems) == 1:  # only one shape could apply: report its specific error
        raise ModelValidationError(problems[0])
    raise _fail(path, "does not match any allowed shape (" + "; ".join(problems) + ")")


def _fits(tp: Any, value: Any) -> bool:
    """Coarse check of whether ``value`` could be a ``tp`` (used to pick a union member)."""
    origin = get_origin(tp)
    if origin is Annotated:
        marker = get_args(tp)[1]
        if marker in (JSON_OBJECT_MARKER, MATCHER_MAP_MARKER):
            return isinstance(value, Mapping)
        if marker == MATCHER_LIST_MARKER:
            return isinstance(value, (list, tuple))
        return True
    if origin is tuple:
        return isinstance(value, (list, tuple))
    if origin in (Mapping, dict, FrozenDict):
        return isinstance(value, Mapping)
    if origin is Literal:
        return value in get_args(tp)
    if origin is Union or origin is types.UnionType:
        return any(_fits(arg, value) for arg in get_args(tp))
    if isinstance(tp, type):
        if tp is bool:
            return isinstance(value, bool)
        if tp in (int, float):
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if tp is str or issubclass(tp, Enum):
            return isinstance(value, str)
        if dataclasses.is_dataclass(tp) and not issubclass(tp, Matcher):
            return isinstance(value, Mapping)
    return True


def _decode_tuple(args: tuple[Any, ...], value: Any, path: str, ctx: _Ctx) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise _fail(path, f"expected a list, got {_type_name(value)}")
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(_decode(args[0], item, f"{path}[{i}]", ctx) for i, item in enumerate(value))
    if len(value) != len(args):
        raise _fail(path, f"expected {len(args)} items, got {len(value)}")
    return tuple(
        _decode(a, item, f"{path}[{i}]", ctx)
        for i, (a, item) in enumerate(zip(args, value, strict=True))
    )


def _decode_mapping(
    args: tuple[Any, ...], value: Any, path: str, ctx: _Ctx
) -> "FrozenDict[Any, Any]":
    if not isinstance(value, Mapping):
        raise _fail(path, f"expected an object, got {_type_name(value)}")
    key_type, value_type = args if len(args) == 2 else (str, Any)
    out: dict[Any, Any] = {}
    for key, item in value.items():
        if isinstance(key, bool) or not isinstance(key, (str, int)):
            raise _fail(path, f"key {key!r} must be a string")
        if isinstance(key, int) and key_type is str:
            key = str(key)  # YAML turns {1: ...} keys into ints
        elif isinstance(key, int) and key_type is not Any:
            raise _fail(path, f"key {key!r} must be a string")
        out[key] = _decode(value_type, item, f"{path}[{key!r}]", ctx)
    return FrozenDict(out)


def _decode_matcher(value: Any, path: str, ctx: _Ctx) -> Matcher:
    if isinstance(value, Matcher):
        return value
    if isinstance(value, Mapping) and "$m" in value:
        cls = matcher_class(str(value["$m"]))
        body = {key: item for key, item in value.items() if key != "$m"}
        return _decode_dataclass(cls, body, path, ctx)  # type: ignore[no-any-return]
    literal = freeze_json(value, path)
    if _holds_tag_key(literal):
        raise _fail(
            path,
            "a '$m' key inside a plain value is not decoded as a matcher; use "
            "{'$m': 'dict_match', 'fields': {...}} (or 'contains' / 'unordered' for lists) to "
            "match inside a value, or {'$m': 'eq', 'value': ...} if '$m' is literal data",
        )
    return Eq(literal)


# ----------------------------------------------------------------------------- hint checking


def check_supported_hints(cls: type) -> list[str]:
    """Return a message for every field of ``cls`` whose type hint the decoder cannot handle."""
    problems: list[str] = []
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        try:
            _check_hint(_hints(cls)[f.name])
        except TypeError as exc:
            problems.append(f"{cls.__name__}.{f.name}: {exc}")
    return problems


def _check_hint(tp: Any) -> None:
    origin = get_origin(tp)
    args = get_args(tp)
    if tp is Any or tp is object or tp is None or tp is _NONE_TYPE:
        return
    if origin is Annotated:
        if args[1] in (
            JSON_MARKER,
            JSON_OBJECT_MARKER,
            MATCHER_MAP_MARKER,
            MATCHER_OR_STR_MARKER,
            MATCHER_LIST_MARKER,
        ):
            return
        _check_hint(args[0])
        return
    if origin is Union or origin is types.UnionType:
        for arg in args:
            _check_hint(arg)
        return
    if origin is Literal:
        return
    if origin is tuple:
        for arg in args:
            if arg is not Ellipsis:
                _check_hint(arg)
        return
    if origin in (Mapping, dict, FrozenDict):
        for arg in args:
            _check_hint(arg)
        return
    if isinstance(tp, type) and (
        tp in (str, bool, int, float)
        or issubclass(tp, (Enum, Matcher))
        or dataclasses.is_dataclass(tp)
    ):
        return
    raise TypeError(f"unsupported type hint {tp!r}")
