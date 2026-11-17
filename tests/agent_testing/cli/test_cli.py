"""The ``disseqt`` CLI and the first-shippable acceptance demo (CL-, CP-)."""

import pytest

pytest.importorskip("yaml")  # YAML case files need the [yaml] extra
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from disseqt_agent_testing.cli import main, register_subcommand
from disseqt_agent_testing.cli.exit import (
    exit_code_for,
    register_exit_policy,
    unregister_exit_policy,
)
from disseqt_agent_testing.cli.main import subcommands, unregister_subcommand
from disseqt_agent_testing.cli.test_cmd import build_parser, pytest_args
from disseqt_agent_testing.runner.schema import suite_result_schema

from .._support.schema_check import validate
from .demo import AGENTS, write_demo

SRC = str(Path(__file__).resolve().parents[3] / "src")
ROOT = str(Path(__file__).resolve().parents[3])
BOOT = "import sys\nfrom disseqt_agent_testing.cli.main import main\nsys.exit(main(sys.argv[1:]))\n"


def cli(
    cwd: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    e = {
        **os.environ,
        "PYTHONPATH": f"{SRC}{os.pathsep}{ROOT}",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        **(env or {}),
    }
    return subprocess.run(
        [sys.executable, "-c", BOOT, *args],
        cwd=cwd,
        env=e,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cl_exit_code_zero_for_the_passing_case(tmp_path: Path) -> None:
    write_demo(tmp_path)
    r = cli(tmp_path, "test", "tests/agent_cases", "--no-pytest", "--case", "refund_happy")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASSED     refund_happy" in r.stdout and "1 cases: 1 passed" in r.stdout


def test_cl_exit_code_one_when_a_case_fails_in_both_modes(tmp_path: Path) -> None:
    write_demo(tmp_path)
    direct = cli(tmp_path, "test", "tests/agent_cases", "--no-pytest", "--output", "direct.json")
    via_pytest = cli(tmp_path, "test", "tests/agent_cases", "--output", "pytest.json")
    assert direct.returncode == via_pytest.returncode == 1, direct.stdout + via_pytest.stdout
    assert "agent case 'refund_wrong_order' FAILED" in direct.stdout
    assert "agent case 'refund_wrong_order' FAILED" in via_pytest.stdout


def test_cl05_both_modes_produce_the_same_canonical_suite_result(tmp_path: Path) -> None:
    write_demo(tmp_path)
    cli(tmp_path, "test", "tests/agent_cases", "--no-pytest", "--output", "direct.json")
    cli(tmp_path, "test", "tests/agent_cases", "--output", "pytest.json")
    from disseqt_agent_testing.runner import SuiteResult

    a = SuiteResult.from_dict(json.loads((tmp_path / "direct.json").read_text()))
    b = SuiteResult.from_dict(json.loads((tmp_path / "pytest.json").read_text()))
    ca, cb = a.canonical_dict(), b.canonical_dict()
    for case in ca["cases"] + cb["cases"]:
        case["source"] = {k: v for k, v in case["source"].items() if k != "nodeid" and k != "path"}
    assert ca["cases"] == cb["cases"] and a.exit_code == b.exit_code == 1
    assert a.totals == b.totals and a.aggregate == b.aggregate


def test_cl06_arguments_after_double_dash_reach_pytest(tmp_path: Path) -> None:
    write_demo(tmp_path)
    r = cli(tmp_path, "test", "tests/agent_cases", "--", "-k", "happy", "-v")
    assert (
        r.returncode == 0
        and "refund_happy PASSED" in r.stdout
        and "refund_wrong_order" not in r.stdout
    )


def test_cl_exit_code_two_for_incomplete_and_zero_when_skipped(tmp_path: Path) -> None:
    text = """version: 1
cases:
  - id: undecidable
    input: "Refund order A12"
    repeats: 2
    run: {agent: "%s:make_demo_agent"}
    fakes:
      get_order: {effect: read, default: {returns: {order_id: A12, total: 59.99}}}
      check_policy: {effect: read, default: {returns: {eligible: true}}}
      issue_refund: {default: {returns: {amount: 59.99}}}
    expect: {decision: call, tools: [{name: get_order}, {name: issue_refund}], max_tokens: 100}
""".replace("%s", AGENTS)
    (tmp_path / "u.agent.yaml").write_text(text)
    assert cli(tmp_path, "test", "u.agent.yaml", "--no-pytest").returncode == 2
    assert (
        cli(tmp_path, "test", "u.agent.yaml", "--no-pytest", "--incomplete", "skip").returncode == 0
    )
    assert cli(tmp_path, "test", "u.agent.yaml").returncode == 2


def test_cl_exit_code_three_for_broken_files_and_five_for_no_cases(tmp_path: Path) -> None:
    (tmp_path / "bad.agent.yaml").write_text("version: 1\ncases:\n  - {id: 'bad id', input: x}\n")
    r = cli(tmp_path, "test", "bad.agent.yaml", "--no-pytest")
    assert r.returncode == 3 and "case id 'bad id' is invalid" in r.stderr
    (tmp_path / "empty").mkdir()
    assert cli(tmp_path, "test", "empty", "--no-pytest").returncode == 5
    assert cli(tmp_path, "test", "bad.agent.yaml").returncode == 3
    assert cli(tmp_path, "test", "empty").returncode == 5


def test_cl_a_missing_agent_factory_is_a_configuration_error(tmp_path: Path) -> None:
    write_demo(tmp_path)
    r = cli(tmp_path, "test", "tests/agent_cases", "--no-pytest", "--agent", "no_such_mod:fn")
    assert r.returncode == 3 and "cannot import agent factory 'no_such_mod:fn'" in r.stderr


def test_cl_exit_code_130_on_interrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import disseqt_agent_testing.runner as runner

    def boom(*a: Any, **k: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "run_suite", boom)
    write_demo(tmp_path)
    assert main(["test", str(tmp_path / "tests" / "agent_cases"), "--no-pytest"]) == 130


def test_exit_code_mapping_without_a_result_and_with_a_policy() -> None:
    assert [exit_code_for(None, pytest_code=c) for c in (0, 1, 2, 3, 4, 5)] == [0, 1, 3, 4, 3, 5]
    assert exit_code_for(None, pytest_code=2, interrupted=True) == 130
    assert exit_code_for(None) == 3

    class R:
        exit_code = 1

    assert exit_code_for(R()) == 1
    register_exit_policy("regression", lambda r: 6)
    try:
        assert exit_code_for(R(), policy="regression") == 6
        with pytest.raises(ValueError, match="already registered"):
            register_exit_policy("regression", lambda r: 7)
    finally:
        unregister_exit_policy("regression")


def test_the_subcommand_registry_and_help() -> None:
    seen: list[list[str]] = []
    register_subcommand("custom", lambda argv: seen.append(argv) or 0)
    try:
        assert (
            "custom" in subcommands()
            and "judges" in subcommands()
            and main(["custom", "calibrate"]) == 0
            and seen == [["calibrate"]]
        )
        with pytest.raises(ValueError, match="already registered"):
            register_subcommand("custom", lambda argv: 0)
    finally:
        unregister_subcommand("custom")
    assert main([]) == 3 and main(["--help"]) == 0 and main(["nope"]) == 3


def test_argument_translation_to_pytest_flags() -> None:
    ns = build_parser().parse_args(
        [
            "cases",
            "--repeats",
            "5",
            "--tags",
            "smoke",
            "--case",
            "a*",
            "--early-stop",
            "--seed",
            "3",
        ]
    )
    ns.passthrough = ["-n", "2"]
    args = pytest_args(ns, "out.json")
    for want in (
        "--disseqt-repeats",
        "5",
        "--disseqt-tags",
        "smoke",
        "--disseqt-case",
        "a*",
        "--disseqt-early-stop",
        "--disseqt-seed",
        "3",
        "--disseqt-output",
        "out.json",
        "-n",
        "2",
    ):
        assert want in args
    assert args[0] == "cases"


def test_the_runner_package_never_imports_pytest() -> None:
    import ast

    root = Path(SRC) / "disseqt_agent_testing" / "runner"
    offenders = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom) and node.level == 0
                    else []
                )
            )
            offenders += [
                f"{path.name}: {n}" for n in names if n.split(".")[0] in ("pytest", "_pytest")
            ]
    assert not offenders, offenders


def test_first_shippable_acceptance_demo(tmp_path: Path) -> None:
    """The plan's demo: the refund YAML (happy, missing-order, wrong-order; repeats: 5)."""
    write_demo(tmp_path)
    (tmp_path / "tests" / "agent_cases" / "test_param.py").write_text(
        "from tests.agent_testing._support.demo_agents import make_demo_agent\n"
        "from tests.agent_testing._support.runner_agents import refund_case\n"
        "def test_python_defined(disseqt_run_case):\n"
        "    disseqt_run_case(refund_case('py_case', repeats=2), agent_factory=lambda tools: __import__('tests.agent_testing._support.runner_agents', fromlist=['x']).make_good_agent(tools))\n"
    )
    out = tmp_path / "build" / "disseqt.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([SRC, ROOT]),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "disseqt_agent_testing.pytest_plugin", "-p", "no:cacheprovider", "tests/agent_cases", "--disseqt-output", str(out), "-q"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180,
    )  # fmt: skip
    assert r.returncode == 1, r.stdout + r.stderr
    assert (
        "agent case 'refund_wrong_order' FAILED" in r.stdout
        and "$.order_id  ungrounded_value" in r.stdout
    )
    data = json.loads(out.read_text())
    schema = suite_result_schema()
    assert validate(data, schema, schema) == []
    by = {c["case_id"]: c["status"] for c in data["cases"]}
    assert by == {
        "refund_happy": "passed",
        "refund_missing_order": "passed",
        "refund_wrong_order": "failed",
        "py_case": "passed",
    }
    only = cli(tmp_path, "test", "tests/agent_cases", "--case", "refund_happy")
    assert only.returncode == 0
