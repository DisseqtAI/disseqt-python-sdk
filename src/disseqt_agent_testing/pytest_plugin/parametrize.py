"""``parametrize_cases``: Python-side access to the same cases as a case file."""

from typing import Any

import pytest


def parametrize_cases(source: Any, *, filter: Any = None, argname: str = "agent_case") -> Any:
    """``@parametrize_cases("tests/cases")`` -> a ``pytest.mark.parametrize`` over the loaded cases
    (``ids=`` passed by keyword). A problem with the source raises ``ValueError`` listing every
    message, at collection time."""
    from ..runner.loader import format_errors, load_cases

    loaded = load_cases(source, filter=filter)
    if loaded.errors:
        raise ValueError("\n".join(format_errors(loaded.errors)))
    return pytest.mark.parametrize(
        argname, list(loaded.cases), ids=[c.case.id for c in loaded.cases]
    )
