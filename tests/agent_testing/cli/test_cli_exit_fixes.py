"""Review fix pass: exit-code combination, stale output, sys.path, usage errors, trust."""

import pytest

pytest.importorskip("yaml")
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from disseqt_agent_testing.cli import main

from .demo import write_demo
from .test_cli import cli


def test_a_failing_plain_pytest_test_beside_passing_agent_cases_exits_nonzero(
    tmp_path: Path,
) -> None:
    cases = write_demo(tmp_path)
    (cases / "refund.agent.yaml").write_text(
        (cases / "refund.agent.yaml").read_text().split("  - id: refund_missing_order")[0]
    )  # only the passing case remains
    (cases / "test_plain.py").write_text("def test_boom():\n    assert False\n")
    r = cli(tmp_path, "test", "tests/agent_cases")
    assert r.returncode == 1, r.stdout + r.stderr


def test_a_stale_output_file_is_never_read_when_pytest_fails_early(tmp_path: Path) -> None:
    write_demo(tmp_path)
    stale = tmp_path / "stale.json"
    r0 = cli(
        tmp_path, "test", "tests/agent_cases", "--case", "refund_happy", "--output", "stale.json"
    )
    assert r0.returncode == 0 and stale.exists()
    r = cli(tmp_path, "test", "tests/agent_cases", "--output", "stale.json", "--", "--bogus-flag")
    assert r.returncode != 0, r.stdout + r.stderr
    assert not stale.exists()  # the pre-existing file was removed, not trusted


def test_the_temp_dir_is_cleaned_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_demo(tmp_path)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    before = set(Path(tempfile.gettempdir()).glob("disseqt-*"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    main(["test", "tests/agent_cases", "--case", "refund_happy"])
    assert set(Path(tempfile.gettempdir()).glob("disseqt-*")) == before


def test_output_is_copied_to_the_requested_path(tmp_path: Path) -> None:
    write_demo(tmp_path)
    r = cli(
        tmp_path, "test", "tests/agent_cases", "--case", "refund_happy", "--output", "out/s.json"
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((tmp_path / "out" / "s.json").read_text())["cases"]


def _simple_case_file(root: Path, with_run: bool = False) -> None:
    from ..plugin.conftest import case as mkcase
    from ..plugin.conftest import casefile

    (root / "c.agent.yaml").write_text(casefile(mkcase("c1", agent=None)))


def _script_run(
    cwd: Path, *args: str, pytest_mode: bool = False
) -> subprocess.CompletedProcess[str]:
    """Like a console script: sys.path[0] is the script's directory, not the cwd."""
    from .test_cli import ROOT, SRC

    script = cwd.parent / (cwd.name + "_bin") / "run.py"
    script.parent.mkdir(exist_ok=True)
    script.write_text(
        "import sys\nfrom disseqt_agent_testing.cli.main import main\nsys.exit(main(sys.argv[1:]))\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": f"{SRC}{os.pathsep}{ROOT}",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize("pytest_mode", [False, True])
def test_agent_factory_module_in_the_cwd_imports_like_python_dash_m(
    tmp_path: Path, pytest_mode: bool
) -> None:
    _simple_case_file(tmp_path)
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "refund.py").write_text(
        "from tests.agent_testing._support.runner_agents import make_good_agent as make_refund_agent\n"
    )
    args = ["test", "c.agent.yaml", "--agent", "agents.refund:make_refund_agent"]
    if not pytest_mode:
        args.append("--no-pytest")
    r = _script_run(tmp_path, *args)
    assert r.returncode == 0, r.stdout + r.stderr


def test_sys_path_is_restored_after_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _simple_case_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    before = list(sys.path)
    main(
        [
            "test",
            "c.agent.yaml",
            "--no-pytest",
            "--agent",
            "tests.agent_testing._support.runner_agents:make_good_agent",
        ]
    )
    assert sys.path == before


def test_usage_errors_exit_3_not_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["test", "--repeats", "notanint"]) == 3
    assert main(["test", "--no-such-flag"]) == 3
    assert main(["test", "--help"]) == 0
    assert "exit codes:" in capsys.readouterr().out


def test_seed_garbage_in_direct_mode_is_a_clean_exit_3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _simple_case_file(tmp_path)
    assert main(["test", str(tmp_path / "c.agent.yaml"), "--no-pytest", "--seed", "abc"]) == 3
    assert "--seed must be an integer" in capsys.readouterr().err


@pytest.mark.parametrize("no_pytest", [True, False])
def test_unwritable_output_after_a_passing_run_exits_3(
    tmp_path: Path,
    no_pytest: bool,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _simple_case_file(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    args = ["test", "c.agent.yaml", "--agent", "tests.agent_testing._support.runner_agents:make_good_agent",
            "--output", str(blocker / "s.json")]  # fmt: skip
    if no_pytest:
        args.append("--no-pytest")
    assert main(args) == 3
    assert "cannot write --output" in capsys.readouterr().err


def test_a_collection_error_in_a_user_py_file_exits_3_not_130(tmp_path: Path) -> None:
    write_demo(tmp_path)
    (tmp_path / "tests" / "agent_cases" / "test_broken.py").write_text("import not_a_module_xyz\n")
    r = cli(tmp_path, "test", "tests/agent_cases")
    assert r.returncode == 3, r.stdout + r.stderr


def test_only_repeat_replay_of_a_passing_repeat_is_not_incomplete(tmp_path: Path) -> None:
    write_demo(tmp_path)
    for extra in ((), ("--no-pytest",)):
        r = cli(
            tmp_path,
            "test",
            "tests/agent_cases",
            "--case",
            "refund_happy",
            "--only-repeat",
            "1",
            *extra,
        )
        assert r.returncode == 0, (extra, r.stdout + r.stderr)


def test_packaging_ships_py_typed_and_keeps_design_notes_out_of_the_sdist() -> None:
    root = Path(__file__).resolve().parents[3]
    assert (root / "src" / "disseqt_agent_testing" / "py.typed").is_file()
    text = (root / "pyproject.toml").read_text()
    sdist = text.split("[tool.hatch.build.targets.sdist]")[1].split("\n[")[0]
    assert '"/plan.md"' in sdist and '"/plans"' in sdist
    assert '"src/disseqt_agent_testing"' in text.split("[tool.hatch.build.targets.wheel]")[1]


def test_repeated_case_flags_all_reach_pytest_mode(tmp_path: Path) -> None:
    write_demo(tmp_path)
    for order in (("refund_wrong_order", "refund_happy"), ("refund_happy", "refund_wrong_order")):
        flags = [a for cid in order for a in ("--case", cid)]
        for extra in ((), ("--no-pytest",)):
            r = cli(tmp_path, "test", "tests/agent_cases", *flags, *extra)
            assert r.returncode == 1, (order, extra, r.stdout + r.stderr)


def test_no_agent_case_beside_a_passing_plain_test_exits_5_in_both_modes(tmp_path: Path) -> None:
    (tmp_path / "test_plain.py").write_text("def test_ok():\n    assert True\n")
    for extra in ((), ("--no-pytest",)):
        r = cli(tmp_path, "test", ".", *extra)
        assert r.returncode == 5, (extra, r.stdout + r.stderr)


def test_a_failing_plain_test_beats_an_all_skipped_suite(tmp_path: Path) -> None:
    from ..plugin.conftest import case as mkcase
    from ..plugin.conftest import casefile

    skipped = mkcase("sk").replace('"}', '", skip: "later"}', 1)
    (tmp_path / "c.agent.yaml").write_text(casefile(skipped))
    assert cli(tmp_path, "test", ".").returncode == 5
    (tmp_path / "test_plain.py").write_text("def test_boom():\n    assert False\n")
    r = cli(tmp_path, "test", ".")
    assert r.returncode == 1, r.stdout + r.stderr


def test_a_failing_plain_test_beats_an_incomplete_suite(tmp_path: Path) -> None:
    from ..plugin.conftest import case as mkcase
    from ..plugin.conftest import casefile

    slow = [mkcase(f"s{i}", "make_slow_agent", repeats=3) for i in range(3)]
    (tmp_path / "c.agent.yaml").write_text(casefile(*slow))
    args = ("test", ".", "--workers", "1", "--suite-timeout", "0.45")
    assert cli(tmp_path, *args).returncode == 2
    (tmp_path / "test_plain.py").write_text("def test_boom():\n    assert False\n")
    r = cli(tmp_path, *args)
    assert r.returncode == 1, r.stdout + r.stderr


class _Suite:
    def __init__(self, exit_code: int):
        self.exit_code = exit_code


@pytest.mark.parametrize(
    ("suite", "pytest_code", "plain_failed", "expected"),
    [
        (0, 1, True, 1),
        (2, 1, True, 1),
        (5, 1, True, 1),
        (2, 1, False, 2),  # pytest's 1 is the incomplete agent item itself
        (5, 0, False, 5),
        (1, 2, False, 3),
        (3, 1, True, 3),
        (2, 0, False, 2),
        (0, 5, False, 5),
    ],
)
def test_exit_codes_merge_by_priority(
    suite: int, pytest_code: int, plain_failed: bool, expected: int
) -> None:
    from disseqt_agent_testing.cli.exit import exit_code_for

    got = exit_code_for(_Suite(suite), pytest_code=pytest_code, plain_failed=plain_failed)
    assert got == expected


@pytest.mark.parametrize("no_pytest", [True, False])
def test_two_runs_in_one_process_do_not_share_the_agent_module(
    tmp_path: Path, no_pytest: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    codes = []
    for name, factory in (("one", "make_good_agent"), ("two", "make_wrong_agent")):
        project = tmp_path / name
        project.mkdir()
        _simple_case_file(project)
        (project / "project_agents_29.py").write_text(
            f"from tests.agent_testing._support.runner_agents import {factory} as make\n"
        )
        monkeypatch.chdir(project)
        args = ["test", "c.agent.yaml", "--agent", "project_agents_29:make"]
        codes.append(main(args + (["--no-pytest"] if no_pytest else [])))
        assert "project_agents_29" not in sys.modules
    assert codes == [0, 1]


@pytest.mark.parametrize("no_pytest", [True, False])
def test_an_xfail_case_gives_the_same_exit_code_in_both_modes(
    tmp_path: Path, no_pytest: bool
) -> None:
    from ..plugin.conftest import case as mkcase
    from ..plugin.conftest import casefile

    extra = ("--no-pytest",) if no_pytest else ()
    for agent, expected, status in (
        ("make_wrong_agent", 0, "xfailed"),
        ("make_good_agent", 1, "failed"),
    ):
        text = mkcase("xf", agent).replace(f':{agent}"}}', f':{agent}", xfail: "known bug"}}')
        (tmp_path / "x.agent.yaml").write_text(casefile(text))
        r = cli(tmp_path, "test", "x.agent.yaml", "--output", "o.json", *extra)
        assert r.returncode == expected, (agent, r.stdout + r.stderr)
        data = json.loads((tmp_path / "o.json").read_text())
        assert [c["status"] for c in data["cases"]] == [status]
