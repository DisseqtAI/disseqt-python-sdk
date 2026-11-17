"""Helpers for testing the pytest plugin with pytester."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SRC = str(Path(__file__).resolve().parents[3] / "src")
ROOT = str(Path(__file__).resolve().parents[3])
AGENTS = "tests.agent_testing._support.runner_agents"

FAKES = """    fakes:
      get_order: {effect: read, default: {returns: {order_id: A12, total: 59.99}}}
      issue_refund: {default: {returns: {amount: 59.99}}}
"""
GOOD_EXPECT = """    expect:
      decision: call
      tools:
        - {name: get_order, id: lookup}
        - name: issue_refund
          args: {order_id: {"$m": "ref", source: "call:lookup.result.order_id"}}
          after: [lookup]
      order: partial
"""


def case(
    case_id: str,
    agent: str | None = "make_good_agent",
    *,
    extra: str = "",
    expect: str = GOOD_EXPECT,
    tags: str = "",
    repeats: int | None = None,
) -> str:
    lines = [f"  - id: {case_id}", '    input: "Refund order A12"']
    if agent:
        lines.append(f'    run: {{agent: "{AGENTS}:{agent}"}}')
    if tags:
        lines.append(f"    tags: [{tags}]")
    if repeats:
        lines.append(f"    repeats: {repeats}")
    if extra:
        lines.append(extra.rstrip("\n"))
    return "\n".join(lines) + "\n" + FAKES + expect


def casefile(*cases: str, header: str = "") -> str:
    return "version: 1\n" + header + "cases:\n" + "".join(cases)


@pytest.fixture
def pt(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run an inner pytest with the plugin loaded explicitly (autoload off for isolation)."""
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    for var in (
        "DISSEQT_REPEATS",
        "DISSEQT_SEED",
        "DISSEQT_WORKERS",
        "DISSEQT_INCOMPLETE",
        "DISSEQT_OUTPUT",
        "CI",
    ):
        monkeypatch.delenv(var, raising=False)

    def run(*args: str, plugin: bool = True) -> pytest.RunResult:
        base = ["-p", "disseqt_agent_testing.pytest_plugin"] if plugin else []
        return pytester.runpytest(*base, "-p", "no:cacheprovider", "-p", "no:randomly", *args)

    run.pytester = pytester  # type: ignore[attr-defined]
    return run


def subprocess_pytest(
    cwd: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    import os

    e = {**os.environ, "PYTHONPATH": f"{SRC}{os.pathsep}{ROOT}", **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=cwd,
        env=e,
        capture_output=True,
        text=True,
        timeout=120,
    )


_ = json
