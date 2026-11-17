"""The four Disseqt hooks. All keyword-only; implement them in ``conftest.py`` or a plugin."""

from typing import Any

import pytest


@pytest.hookspec(firstresult=True)
def pytest_disseqt_judge(config: pytest.Config) -> Any:  # pragma: no cover - spec only
    """Return a ``Judge`` (or ``None``) for agent cases that use judged checks."""


@pytest.hookspec(firstresult=True)
def pytest_disseqt_agent_factory(case: Any, config: pytest.Config) -> Any:  # pragma: no cover
    """Return the agent factory for ``case`` (an ``AgentTestCase``), or ``None``. Scope it by
    directory with a ``conftest.py``; a case's own ``run: {agent: ...}`` wins over this hook."""


def pytest_disseqt_case_result(result: Any) -> None:  # pragma: no cover - spec only
    """Called on the controller with each ``CaseResult`` as soon as its item finishes."""


def pytest_disseqt_suite_result(result: Any) -> None:  # pragma: no cover - spec only
    """Called once on the controller with the final ``SuiteResult``."""
