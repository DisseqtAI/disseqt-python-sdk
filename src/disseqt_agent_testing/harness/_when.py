"""Yes/no evaluation of a fake's ``when`` matchers against a call's arguments.

This answers only "does this rule apply to this call?" for fake dispatch. Scoring semantics
(per-argument error codes, partial credit, ``Ref`` resolution) belong to the assertions step; when
that lands it should replace this module with its own ``match_value`` so the two cannot disagree.
``Ref``, ``Semantic`` and ``Predicate`` cannot appear here (``FakeResponse`` rejects them).
"""

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date, timezone
from fnmatch import fnmatchcase
from typing import Any

from ..models import (
    Absent,
    AnyValue,
    Approx,
    Eq,
    Matcher,
    OneOf,
    Optional_,
    Regex,
)
from ..models.matchers import (
    Contains,
    DateEq,
    DictMatch,
    Normalized,
    TypeIs,
    Unordered,
    is_date_only,
    parse_iso,
)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def equal(expected: Any, actual: Any, coerce_numeric: bool = True) -> bool:
    """Equality with a bool guard (``1`` never equals ``True``) and recursive containers."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    if _is_number(expected) and _is_number(actual):
        if not coerce_numeric and type(expected) is not type(actual):
            return False
        return bool(expected == actual)  # NaN never equals anything
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, str):
        return isinstance(actual, str) and expected == actual
    if isinstance(expected, (tuple, list)):
        return (
            isinstance(actual, (tuple, list))
            and len(expected) == len(actual)
            and all(equal(e, a, coerce_numeric) for e, a in zip(expected, actual, strict=True))
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(expected) == set(actual)
            and all(equal(expected[k], actual[k], coerce_numeric) for k in expected)
        )
    return False


def _normalize(text: str, strip_chars: str, casefold: bool) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if ch not in strip_chars)
    return text.casefold() if casefold else text


def _perfect_matching(items: Sequence[Matcher], elements: Sequence[Any], *, full: bool) -> bool:
    """Every item matched by a distinct element (and, if ``full``, every element used)."""
    if full and len(items) != len(elements):
        return False
    compatible = [[j for j, el in enumerate(elements) if matches(item, el)] for item in items]
    owner: dict[int, int] = {}

    def assign(i: int, seen: set[int]) -> bool:
        for j in compatible[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or assign(owner[j], seen):
                owner[j] = i
                return True
        return False

    return all(assign(i, set()) for i in range(len(items)))


def _date_equal(expected: DateEq, actual: Any) -> bool:
    if not isinstance(actual, str):
        return False
    try:
        want = parse_iso(expected.value)
        got = parse_iso(actual)
    except ValueError:
        return False
    zone: Any = timezone.utc
    if expected.tz is not None:
        try:
            from zoneinfo import ZoneInfo

            zone = ZoneInfo(expected.tz)
        except Exception:  # unknown zone or missing tz data: the rule cannot apply
            return False
    if is_date_only(expected.value):  # a date-only expectation compares calendar dates
        got_date: date = got.astimezone(zone).date() if got.tzinfo else got.date()
        return got_date == want.date()
    if (want.tzinfo is None) != (got.tzinfo is None):
        return False
    if want.tzinfo is None:
        return want == got
    return want.astimezone(timezone.utc) == got.astimezone(timezone.utc)


def _type_ok(name: str, value: Any) -> bool:
    return {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": _is_number(value),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, (list, tuple)),
        "object": isinstance(value, Mapping),
        "null": value is None,
    }[name]


def matches(matcher: Matcher, actual: Any) -> bool:
    """Does ``actual`` (a present value) satisfy ``matcher``?"""
    if isinstance(matcher, Eq):
        return equal(matcher.value, actual, matcher.coerce_numeric)
    if isinstance(matcher, OneOf):
        return any(equal(value, actual) for value in matcher.values)
    if isinstance(matcher, AnyValue):
        return True
    if isinstance(matcher, Optional_):
        return matches(matcher.inner, actual)
    if isinstance(matcher, Regex):
        return (
            isinstance(actual, str)
            and re.search(matcher.pattern, actual, matcher.flags) is not None
        )
    if isinstance(matcher, Approx):
        return _is_number(actual) and math.isclose(
            actual, matcher.value, rel_tol=matcher.rel_tol, abs_tol=matcher.abs_tol
        )
    if isinstance(matcher, TypeIs):
        return any(_type_ok(name, actual) for name in matcher.types)
    if isinstance(matcher, Normalized):
        if not isinstance(actual, str):
            return False
        got = _normalize(actual, matcher.strip_chars, matcher.casefold)
        return any(
            got == _normalize(value, matcher.strip_chars, matcher.casefold)
            for value in matcher.values
        )
    if isinstance(matcher, Contains):
        if isinstance(actual, str):
            return all(
                isinstance(item, Eq)
                and isinstance(item.value, str)
                and item.value.casefold() in actual.casefold()
                for item in matcher.items
            )
        return isinstance(actual, (list, tuple)) and _perfect_matching(
            matcher.items, actual, full=False
        )
    if isinstance(matcher, Unordered):
        return isinstance(actual, (list, tuple)) and _perfect_matching(
            matcher.items, actual, full=True
        )
    if isinstance(matcher, DictMatch):
        return _dict_matches(matcher, actual)
    if isinstance(matcher, DateEq):
        return _date_equal(matcher, actual)
    return False  # Ref / Semantic / Predicate: not decidable at dispatch time


def _dict_matches(matcher: DictMatch, actual: Any) -> bool:
    if not isinstance(actual, Mapping):
        return False
    view = {
        key: value
        for key, value in actual.items()
        if not any(fnmatchcase(key, glob) for glob in matcher.ignore)
        and not (key in matcher.defaults and matches(matcher.defaults[key], value))
    }
    if not when_matches(matcher.fields, view):
        return False
    return matcher.mode != "exact" or set(view) <= set(matcher.fields)


def when_matches(when: Mapping[str, Matcher], args: Mapping[str, Any]) -> bool:
    """All of a rule's ``when`` matchers hold. A missing key never matches (except ``Absent``,
    ``Optional_`` and nothing else); an empty ``when`` matches any call."""
    for key, matcher in when.items():
        present = key in args
        if isinstance(matcher, Absent):
            if present:
                return False
        elif isinstance(matcher, Optional_):
            if present and not matches(matcher, args[key]):
                return False
        elif not present or not matches(matcher, args[key]):
            return False
    return True
