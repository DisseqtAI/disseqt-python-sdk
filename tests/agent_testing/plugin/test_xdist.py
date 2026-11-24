"""pytest-xdist support (XD-). Skipped when xdist is not installed; CI runs them with it."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("xdist")

from disseqt_agent_testing.runner import SuiteResult, derive_seed  # noqa: E402

from .conftest import AGENTS, ROOT, SRC, case, casefile, subprocess_pytest  # noqa: E402

CASES = casefile(
    case("alpha", repeats=4),
    case("bravo", "make_patterned_agent", repeats=5),
    case("charlie", repeats=3),
    case("delta", "make_patterned_agent", repeats=5),
)


def run(tmp: Path, *args: str, env: dict[str, str] | None = None) -> Any:
    return subprocess_pytest(
        tmp, "-p", "disseqt_agent_testing.pytest_plugin", "-p", "xdist.plugin", "tests_dir", "-q", *args,
        env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", **(env or {})},
    )  # fmt: skip


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "tests_dir").mkdir()
    (tmp_path / "tests_dir" / "suite.agent.yaml").write_text(CASES)
    return tmp_path


def load(path: Path) -> Any:
    return SuiteResult.from_dict(json.loads(path.read_text()))


def test_xd01_xd03_workers_give_the_same_outcomes_and_canonical_result(project: Path) -> None:
    results = {}
    for label, extra in (
        ("single", []),
        ("two", ["-n", "2"]),
        ("three", ["-n", "3", "--dist", "worksteal"]),
    ):
        out = project / f"{label}.json"
        r = run(project, "--disseqt-output", str(out), *extra)
        assert r.returncode == 1, r.stdout + r.stderr  # bravo and delta are flaky
        assert out.exists()
        results[label] = load(out)
    base = results["single"].canonical_dict()
    for label in ("two", "three"):
        other = results[label].canonical_dict()
        for c in base["cases"] + other["cases"]:
            c["source"] = {k: v for k, v in c["source"].items() if k not in ("nodeid", "path")}
        assert [c["status"] for c in other["cases"]] == [c["status"] for c in base["cases"]]
        assert [c["repeat_report"] for c in other["cases"]] == [
            c["repeat_report"] for c in base["cases"]
        ]
        assert [c["repeats"] for c in other["cases"]] == [c["repeats"] for c in base["cases"]]
    assert results["two"].totals == results["single"].totals


def test_xd02_the_result_file_is_written_once_by_the_controller(project: Path) -> None:
    out = project / "once.json"
    r = run(project, "-n", "2", "--disseqt-output", str(out))
    assert "disseqt: wrote" in r.stdout and r.stdout.count("disseqt: wrote") == 1
    assert [p.name for p in project.glob("*.json")] == ["once.json"]


def test_xd04_in_item_workers_are_divided_among_xdist_workers(project: Path) -> None:
    r = run(
        project, "-n", "2", "--disseqt-workers", "8", "--disseqt-output", str(project / "w.json")
    )
    data = json.loads((project / "w.json").read_text())
    failing = next(c for c in data["cases"] if c["status"] == "failed")
    assert failing["case_id"] in ("bravo", "delta") and "workers=4" in r.stdout


def test_xd06_a_random_seed_is_shared_by_every_worker(project: Path) -> None:
    out = project / "r.json"
    run(project, "-n", "2", "--disseqt-seed", "random", "--disseqt-output", str(out))
    result = load(out)
    assert result.seed != 0
    for c in result.cases:
        for rep in c.repeats:
            assert rep.seed == derive_seed(result.seed, c.case_id, rep.index)


def test_xd07_a_killed_worker_becomes_worker_crashed_and_others_survive(tmp_path: Path) -> None:
    (tmp_path / "tests_dir").mkdir()
    (tmp_path / "tests_dir" / "agents.py").write_text(
        "import os\n\ndef make_killer(tools):\n    def agent(text):\n        os._exit(1)\n    return agent\n"
    )
    killer = case("killer", "make_good_agent", repeats=1).replace(
        f"{AGENTS}:make_good_agent", "tests_dir.agents:make_killer"
    )
    (tmp_path / "tests_dir" / "suite.agent.yaml").write_text(
        casefile(killer, case("survivor", repeats=1))
    )
    out = tmp_path / "crash.json"
    env = {"PYTHONPATH": os.pathsep.join([str(tmp_path), SRC, ROOT])}
    r = subprocess_pytest(
        tmp_path, "-p", "disseqt_agent_testing.pytest_plugin", "-p", "xdist.plugin", "tests_dir", "-n", "2", "-q",
        "--disseqt-output", str(out), env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", **env},
    )  # fmt: skip
    assert r.returncode == 1, r.stdout + r.stderr
    by = {c.case_id: c for c in load(out).cases}
    assert by["killer"].status == "error" and by["killer"].reason_code == "worker_crashed"
    assert by["survivor"].status == "passed"


def test_xd08_collection_errors_reach_the_controller(project: Path) -> None:
    (project / "tests_dir" / "broken.agent.yaml").write_text("version: 1\ncases: [\n")
    out = project / "broken.json"
    r = run(project, "-n", "2", "--disseqt-output", str(out))
    assert r.returncode != 0, r.stdout + r.stderr
    result = load(out)
    assert result.exit_code == 3
    assert [e.path for e in result.collection_errors] == ["tests_dir/broken.agent.yaml"]


def test_xd09_duplicate_ids_across_files_reach_the_controller(project: Path) -> None:
    (project / "tests_dir" / "again.agent.yaml").write_text(casefile(case("alpha")))
    out = project / "dup.json"
    single = project / "dup_single.json"
    run(project, "--disseqt-output", str(single))
    run(project, "-n", "2", "--disseqt-output", str(out))
    assert [e.code for e in load(single).collection_errors] == ["DSQ-C03"]
    assert load(out).collection_errors == load(single).collection_errors
    assert load(out).exit_code == load(single).exit_code == 3
