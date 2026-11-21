"""Yes/no evaluation of a fake's ``when`` matchers against a call's arguments.

This answers only "does this rule apply to this call?" for fake dispatch. Every value is scored by
the assertion engine's ``match_value`` and reduced to pass/fail, so dispatch and scoring cannot
disagree. ``Ref``, ``Semantic`` and ``Predicate`` cannot appear here (``FakeResponse`` rejects
them).
"""

from collections.abc import Mapping
from typing import Any

from ..assertions.config import ExpectConfigError
from ..assertions.matching import MatchContext, match_value
from ..models import Absent, Matcher, Optional_
from .errors import HarnessConfigError


def matches(matcher: Matcher, actual: Any) -> bool:
    """Does ``actual`` (a present value) satisfy ``matcher``?

    A matcher that cannot be evaluated at all (unknown time zone, no registered implementation)
    is a mistake in the test, so it raises ``HarnessConfigError`` instead of never matching.
    """
    try:
        results = match_value(matcher, actual, MatchContext())
    except ExpectConfigError as exc:
        raise HarnessConfigError(f"a fake's 'when' matcher cannot be evaluated: {exc}") from exc
    return all(result.status == "ok" and result.ok is True for result in results)


def when_matches(when: Mapping[str, Matcher], args: Mapping[str, Any]) -> bool:
    """All of a rule's ``when`` matchers hold. A missing key never matches (except ``Absent``,
    ``Optional_`` and nothing else); an empty ``when`` matches any call."""
    for key, matcher in when.items():
        if key not in args:
            if not isinstance(matcher, (Absent, Optional_)):
                return False
        elif isinstance(matcher, Absent) or not matches(matcher, args[key]):
            return False
    return True
