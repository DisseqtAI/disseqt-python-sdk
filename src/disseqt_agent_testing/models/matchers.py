"""Matcher **data** classes (what an argument value should look like).

Matching logic lives in the assertions step; these classes only hold and validate the
declaration and serialize. JSON form: a plain JSON value is shorthand for ``Eq``; everything else
is ``{"$m": "<tag>", ...fields}`` (tags are registered in ``_registry``).
"""

import dataclasses
import re
from collections.abc import Iterator, Mapping
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime, timedelta, timezone

from ._frozen import FrozenDict, JSONValue, MatcherList, MatcherMap, freeze_json
from ._registry import matcher_tag, register_matcher
from ._validate import (
    as_tuple,
    check_bool,
    check_choice,
    check_int,
    check_number,
    check_str,
    fail,
    set_field,
    str_tuple,
)
from .errors import ModelValidationError


@dataclass(slots=True, frozen=True)
class Matcher:
    """Base marker. Not instantiated directly."""


@dataclass(slots=True, frozen=True)
class Eq(Matcher):
    """Equal to ``value`` (``1 == 1.0`` unless ``coerce_numeric=False``)."""

    value: JSONValue
    _: KW_ONLY
    coerce_numeric: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.value, Matcher):
            fail("Eq.value", "must be a JSON value, not a matcher")
        set_field(self, "value", freeze_json(self.value, "Eq.value"))
        check_bool(self.coerce_numeric, "Eq.coerce_numeric")


@dataclass(slots=True, frozen=True)
class OneOf(Matcher):
    """Equal to any of ``values`` (at least one)."""

    values: tuple[JSONValue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, (list, tuple)) or len(self.values) < 1:
            fail("OneOf.values", "must be a non-empty list or tuple")
        set_field(
            self,
            "values",
            tuple(freeze_json(v, f"OneOf.values[{i}]") for i, v in enumerate(self.values)),
        )


@dataclass(slots=True, frozen=True)
class AnyValue(Matcher):
    """The key must be present; any value (including ``null``) matches."""


@dataclass(slots=True, frozen=True)
class Absent(Matcher):
    """The key must NOT be present."""


@dataclass(slots=True, frozen=True)
class Regex(Matcher):
    """The string matches ``pattern`` (``re.search``). Compiled once here to validate."""

    pattern: str
    _: KW_ONLY
    flags: int = 0

    def __post_init__(self) -> None:
        check_str(self.pattern, "Regex.pattern", non_empty=False)
        check_int(self.flags, "Regex.flags", minimum=0)
        try:
            re.compile(self.pattern, self.flags)
        except re.error as exc:
            fail("Regex.pattern", f"invalid regular expression {self.pattern!r}: {exc}")
        except (ValueError, OverflowError) as exc:  # bad flag combinations, huge flag values
            fail("Regex.flags", f"{self.flags!r} cannot be used with this pattern: {exc}")


@dataclass(slots=True, frozen=True)
class Approx(Matcher):
    """A number within tolerance of ``value`` (``math.isclose`` semantics)."""

    value: float
    _: KW_ONLY
    abs_tol: float = 0.0
    rel_tol: float = 1e-9

    def __post_init__(self) -> None:
        set_field(self, "value", check_number(self.value, "Approx.value"))
        set_field(self, "abs_tol", check_number(self.abs_tol, "Approx.abs_tol", minimum=0))
        set_field(self, "rel_tol", check_number(self.rel_tol, "Approx.rel_tol", minimum=0))


_TURN = re.compile(r"[0-9]{1,9}")
_KEY = re.compile(r"[A-Za-z0-9_-]+")
_TARGET = re.compile(r"[A-Za-z0-9_*?-]+")
_INDEX = re.compile(r"([0-9]+)\]")


def parse_ref_source(source: str) -> tuple[str, str | None, tuple[str | int, ...]]:
    """Parse a ``Ref`` source into ``(kind, target, path)``.

    Grammar: ``call:<target>(.result|.args)(.<key>|[<int>])*``, ``input`` or ``input:<turn>``,
    ``fake:<name>(.<key>|[<int>])*``. For ``call`` the path starts with ``"result"`` or
    ``"args"``; for ``input`` the target is the turn (or ``None``). Raises
    ``ModelValidationError`` naming the position of the problem.
    """

    def error(position: int, problem: str) -> ModelValidationError:
        return ModelValidationError(
            f"Ref.source: invalid ref {source!r} at position {position}: {problem}"
        )

    def parse_path(pos: int) -> tuple[tuple[str | int, ...], int]:
        segments: list[str | int] = []
        while pos < len(source):
            if source[pos] == ".":
                key = _KEY.match(source, pos + 1)
                if key is None:
                    raise error(pos + 1, "expected a key after '.'")
                segments.append(key.group())
                pos = key.end()
            elif source[pos] == "[":
                index = _INDEX.match(source, pos + 1)
                if index is None:
                    raise error(pos + 1, "expected an index like [0]")
                segments.append(int(index.group(1)))
                pos = index.end()
            else:
                raise error(pos, f"unexpected character {source[pos]!r}")
        return tuple(segments), pos

    if not isinstance(source, str) or not source:
        raise ModelValidationError("Ref.source: must be a non-empty string")
    if source.startswith("call:"):
        pos = len("call:")
        target = _TARGET.match(source, pos)
        if target is None:
            raise error(pos, "expected a call id, expect id or tool name")
        pos = target.end()
        for part in ("result", "args"):
            marker = "." + part
            if source.startswith(marker, pos):
                path, _ = parse_path(pos + len(marker))
                return "call", target.group(), (part, *path)
        raise error(pos, "expected '.result' or '.args' after the call target")
    if source == "input" or source.startswith("input:"):
        if source == "input":
            return "input", None, ()
        turn = source[len("input:") :]
        if not _TURN.fullmatch(turn):
            raise error(len("input:"), "expected a turn number after 'input:'")
        return "input", turn, ()
    if source.startswith("fake:"):
        pos = len("fake:")
        name = _KEY.match(source, pos)
        if name is None:
            raise error(pos, "expected a fake name")
        path, _ = parse_path(name.end())
        return "fake", name.group(), path
    raise error(0, "must start with 'call:', 'input' or 'fake:'")


@dataclass(slots=True, frozen=True)
class Ref(Matcher):
    """The argument must equal a value taken from an earlier call or from the input.

    Resolved at scoring time (Step 3), never at dispatch.
    """

    source: str

    def __post_init__(self) -> None:
        parse_ref_source(self.source)


@dataclass(slots=True, frozen=True)
class Optional_(Matcher):
    """A missing key passes; a present key must match ``inner``."""

    inner: Matcher

    def __post_init__(self) -> None:
        inner = to_matcher(self.inner, "Optional_.inner")
        if isinstance(inner, (Optional_, Absent)):
            fail("Optional_.inner", f"{type(inner).__name__} cannot be wrapped in Optional_")
        set_field(self, "inner", inner)


TYPE_NAMES: tuple[str, ...] = (
    "string",
    "integer",
    "number",
    "boolean",
    "array",
    "object",
    "null",
)
DICT_MODES: tuple[str, ...] = ("subset", "exact")
PREDICATE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_TIMEZONE_NAME = re.compile(r"^[A-Za-z0-9_+/-]+$")


def _text_values(value: object, path: str) -> tuple[str, ...]:
    """A single string or a non-empty list of strings."""
    items = (value,) if isinstance(value, str) else as_tuple(value, path)
    if not items:
        fail(path, "must not be empty")
    return tuple(check_str(item, f"{path}[{i}]") for i, item in enumerate(items))


@dataclass(slots=True, frozen=True)
class Normalized(Matcher):
    """The string equals any of ``values`` after NFC, deleting ``strip_chars`` and casefolding."""

    values: tuple[str, ...]
    _: KW_ONLY
    strip_chars: str = " ,./-_*^"
    casefold: bool = True

    def __post_init__(self) -> None:
        set_field(self, "values", _text_values(self.values, "Normalized.values"))
        check_str(self.strip_chars, "Normalized.strip_chars", non_empty=False)
        check_bool(self.casefold, "Normalized.casefold")


@dataclass(slots=True, frozen=True)
class TypeIs(Matcher):
    """The value has one of the JSON types in ``types`` (``integer`` and ``number`` exclude bool)."""

    types: tuple[str, ...]

    def __post_init__(self) -> None:
        names = _text_values(self.types, "TypeIs.types")
        for name in names:
            check_choice(name, "TypeIs.types", TYPE_NAMES)
        set_field(self, "types", names)


@dataclass(slots=True, frozen=True)
class Unordered(Matcher):
    """A list with the same items as ``items`` in any order (each item a matcher or plain value)."""

    items: MatcherList

    def __post_init__(self) -> None:
        raw = as_tuple(self.items, "Unordered.items")
        set_field(
            self,
            "items",
            tuple(to_matcher(item, f"Unordered.items[{i}]") for i, item in enumerate(raw)),
        )


@dataclass(slots=True, frozen=True)
class Contains(Matcher):
    """A list containing every item (distinct elements), or a string containing every item."""

    items: MatcherList

    def __post_init__(self) -> None:
        raw = as_tuple(self.items, "Contains.items")
        if not raw:
            fail("Contains.items", "must not be empty")
        set_field(
            self,
            "items",
            tuple(to_matcher(item, f"Contains.items[{i}]") for i, item in enumerate(raw)),
        )


@dataclass(slots=True, frozen=True)
class DictMatch(Matcher):
    """An object whose listed fields match; ``exact`` also forbids unlisted keys."""

    fields: MatcherMap = field(default_factory=FrozenDict)
    _: KW_ONLY
    mode: str = "subset"
    ignore: tuple[str, ...] = ()
    defaults: MatcherMap = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        set_field(self, "fields", matcher_map(self.fields, "DictMatch.fields"))
        check_choice(self.mode, "DictMatch.mode", DICT_MODES)
        set_field(self, "ignore", str_tuple(self.ignore, "DictMatch.ignore"))
        set_field(self, "defaults", matcher_map(self.defaults, "DictMatch.defaults"))


_ISO = re.compile(
    r"(?P<y>[0-9]{4})-(?P<mo>[0-9]{2})-(?P<d>[0-9]{2})"
    r"(?:[Tt](?P<h>[0-9]{2}):(?P<mi>[0-9]{2})(?::(?P<s>[0-9]{2})(?:\.(?P<f>[0-9]{1,6}))?)?"
    r"(?P<z>[Zz]|[+-][0-9]{2}:[0-9]{2})?)?"
)


def is_date_only(value: str) -> bool:
    """True for a bare ``YYYY-MM-DD`` value."""
    return len(value) == 10 and _ISO.fullmatch(value) is not None


def parse_iso(value: str) -> datetime:
    """Parse the one ISO-8601 subset every Python version accepts identically.

    Accepted: ``YYYY-MM-DD`` and ``YYYY-MM-DDTHH:MM[:SS[.f]]`` (``f`` is 1 to 6 digits) with an
    optional ``Z`` or ``+HH:MM`` / ``-HH:MM`` offset (``T`` and ``Z`` may be lower case). Rejected:
    the basic format (``20240101``), a space instead of ``T``, week dates, ordinal dates, more
    than 6 fraction digits, 24:00 and offsets of a day or more. Raises ``ValueError``. Used by
    ``DateEq`` at construction and at match time, so both always agree.
    """
    match = _ISO.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"not an ISO-8601 date or datetime in the supported subset: {value!r}")
    tz: timezone | None = None
    zone = match["z"]
    if zone is not None:
        if zone in ("Z", "z"):
            tz = timezone.utc
        else:
            hours, minutes = int(zone[1:3]), int(zone[4:6])
            if hours > 23 or minutes > 59:
                raise ValueError(f"invalid UTC offset in {value!r}")
            delta = timedelta(hours=hours, minutes=minutes)
            tz = timezone(-delta if zone[0] == "-" else delta)
    fraction = match["f"] or ""
    return datetime(
        int(match["y"]),
        int(match["mo"]),
        int(match["d"]),
        int(match["h"] or 0),
        int(match["mi"] or 0),
        int(match["s"] or 0),
        int(fraction.ljust(6, "0") or 0),
        tzinfo=tz,
    )


@dataclass(slots=True, frozen=True)
class DateEq(Matcher):
    """The same date (date-only ``value``) or instant (datetime ``value``).

    ``value`` and the compared string use the ISO-8601 subset documented on ``parse_iso`` (same
    on every Python version).

    ``tz`` is the IANA zone used for date-only comparison and for naive datetimes; the zone is
    resolved at scoring time (zone data may be missing on some platforms).
    """

    value: str
    _: KW_ONLY
    tz: str | None = None

    def __post_init__(self) -> None:
        check_str(self.value, "DateEq.value")
        try:
            parse_iso(self.value)
        except ValueError:
            fail("DateEq.value", f"{self.value!r} is not an ISO date or datetime")
        if self.tz is not None:
            check_str(self.tz, "DateEq.tz")
            if not _TIMEZONE_NAME.fullmatch(self.tz):
                fail("DateEq.tz", f"{self.tz!r} is not an IANA time zone name")


@dataclass(slots=True, frozen=True)
class Predicate(Matcher):
    """A user predicate, referenced by **registered name** (never an import path)."""

    name: str

    def __post_init__(self) -> None:
        check_str(self.name, "Predicate.name")
        if not PREDICATE_NAME_PATTERN.fullmatch(self.name):
            fail("Predicate.name", f"{self.name!r} must match {PREDICATE_NAME_PATTERN.pattern}")


@dataclass(slots=True, frozen=True)
class Semantic(Matcher):
    """Equivalent in meaning to ``reference`` under ``rubric``; decided by a judge."""

    reference: JSONValue
    _: KW_ONLY
    rubric: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.reference, Matcher):
            fail("Semantic.reference", "must be a JSON value, not a matcher")
        set_field(self, "reference", freeze_json(self.reference, "Semantic.reference"))
        check_str(self.rubric, "Semantic.rubric", non_empty=False)


def to_matcher(value: object, path: str = "matcher") -> Matcher:
    """A ``Matcher`` is kept; any other value is wrapped in ``Eq`` (strictly frozen)."""
    if isinstance(value, Matcher):
        return value
    try:
        return Eq(value)  # type: ignore[arg-type]
    except ModelValidationError as exc:
        raise ModelValidationError(str(exc).replace("Eq.value", path, 1)) from None


def matcher_map(value: object, path: str) -> "FrozenDict[str, Matcher]":
    """Normalize ``{name: matcher-or-plain-value}`` (``None`` means empty)."""
    if value is None:
        return FrozenDict()
    if not isinstance(value, Mapping):
        fail(path, f"must be an object of argument matchers, got {type(value).__name__}")
    out: dict[str, Matcher] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            fail(path, f"argument names must be non-empty strings, got {key!r}")
        out[key] = to_matcher(item, f"{path}.{key}")
    return FrozenDict(out)


def walk_matchers(value: object) -> Iterator[Matcher]:
    """Yield every matcher inside ``value`` (itself, and any matcher nested in its fields)."""
    if isinstance(value, Matcher):
        yield value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from walk_matchers(getattr(value, f.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from walk_matchers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk_matchers(item)


UNDECIDABLE_TAGS = ("ref", "semantic", "predicate")


def reject_undecidable(value: object, path: str, where: str) -> None:
    """Refuse matchers that resolve at scoring time (``Ref``, ``Semantic``, ``Predicate``)."""
    for matcher in walk_matchers(value):
        if matcher_tag(type(matcher)) in UNDECIDABLE_TAGS:
            fail(path, f"matcher {type(matcher).__name__} is not allowed in {where}")


for _tag, _cls in (
    ("eq", Eq),
    ("one_of", OneOf),
    ("any", AnyValue),
    ("absent", Absent),
    ("regex", Regex),
    ("approx", Approx),
    ("ref", Ref),
    ("optional", Optional_),
    ("normalized", Normalized),
    ("type_is", TypeIs),
    ("unordered", Unordered),
    ("contains", Contains),
    ("dict_match", DictMatch),
    ("date_eq", DateEq),
    ("predicate", Predicate),
    ("semantic", Semantic),
):
    register_matcher(_tag, _cls, override=True)
