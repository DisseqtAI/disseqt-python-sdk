"""CLI tests for the new ``disseqt redteam`` subcommands added in this PR:

  - ``redteam technique {create,update,delete}``
  - ``redteam prompt {create,get,update,delete}``
  - ``redteam generated-prompt mark-successful``
  - ``redteam breach {list,get}``

Follows the same shape as ``test_cli_resources.py`` — Click's ``CliRunner``
invokes commands in-process; ``requests_mock`` intercepts the HTTP.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from disseqt_sdk.cli import cli

REDTEAM_BASE = "https://api.disseqt.ai/dataset"
_JAILBREAK = f"{REDTEAM_BASE}/api/v1/jailbreak"
_TESTING = f"{REDTEAM_BASE}/api/v1/testing"


@pytest.fixture(autouse=True)
def _cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISSEQT_API_KEY", "test_key")
    monkeypatch.setenv("DISSEQT_PROJECT_ID", "test_project")
    monkeypatch.delenv("DISSEQT_REDTEAM_BASE_URL", raising=False)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Group help renders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ["redteam", "technique", "--help"],
        ["redteam", "prompt", "--help"],
        ["redteam", "generated-prompt", "--help"],
        ["redteam", "breach", "--help"],
    ],
)
def test_group_help_renders(runner: CliRunner, path: list[str]) -> None:
    result = runner.invoke(cli, path)
    assert result.exit_code == 0
    assert "Commands:" in result.output or "Usage:" in result.output


# ---------------------------------------------------------------------------
# technique CRUD
# ---------------------------------------------------------------------------


def test_technique_create(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(f"{_JAILBREAK}/techniques", json={"id": "t1"})
    result = runner.invoke(
        cli, ["redteam", "technique", "create", "--json", '{"category": "role-play"}']
    )
    assert result.exit_code == 0, result.output
    sent = json.loads(requests_mock.request_history[0].text)
    assert sent == {"category": "role-play"}
    assert '"id": "t1"' in result.output


def test_technique_update(runner: CliRunner, requests_mock) -> None:
    requests_mock.patch(f"{_JAILBREAK}/techniques/t1", json={"id": "t1"})
    result = runner.invoke(
        cli, ["redteam", "technique", "update", "t1", "--json", '{"description": "d"}']
    )
    assert result.exit_code == 0
    assert requests_mock.request_history[0].method == "PATCH"


def test_technique_delete(runner: CliRunner, requests_mock) -> None:
    requests_mock.delete(f"{_JAILBREAK}/techniques/t1", json={"status": "ok"})
    result = runner.invoke(cli, ["redteam", "technique", "delete", "t1"])
    assert result.exit_code == 0
    assert requests_mock.request_history[0].method == "DELETE"


# ---------------------------------------------------------------------------
# prompt CRUD
# ---------------------------------------------------------------------------


def test_prompt_create(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(f"{_JAILBREAK}/prompts", json={"id": "p1"})
    result = runner.invoke(cli, ["redteam", "prompt", "create", "--json", '{"content": "..."}'])
    assert result.exit_code == 0, result.output
    sent = json.loads(requests_mock.request_history[0].text)
    assert sent == {"content": "..."}


def test_prompt_get(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(f"{_JAILBREAK}/prompts/p1", json={"id": "p1"})
    result = runner.invoke(cli, ["redteam", "prompt", "get", "p1"])
    assert result.exit_code == 0
    assert '"id": "p1"' in result.output


def test_prompt_update(runner: CliRunner, requests_mock) -> None:
    requests_mock.patch(f"{_JAILBREAK}/prompts/p1", json={"id": "p1"})
    result = runner.invoke(
        cli, ["redteam", "prompt", "update", "p1", "--json", '{"content": "new"}']
    )
    assert result.exit_code == 0
    assert requests_mock.request_history[0].method == "PATCH"


def test_prompt_delete(runner: CliRunner, requests_mock) -> None:
    requests_mock.delete(f"{_JAILBREAK}/prompts/p1", json={"status": "ok"})
    result = runner.invoke(cli, ["redteam", "prompt", "delete", "p1"])
    assert result.exit_code == 0


def test_prompt_create_json_from_file(runner: CliRunner, requests_mock, tmp_path) -> None:
    body_path = tmp_path / "prompt.json"
    body_path.write_text('{"content": "from-file"}')
    requests_mock.post(f"{_JAILBREAK}/prompts", json={"id": "p1"})
    result = runner.invoke(cli, ["redteam", "prompt", "create", "--json", f"@{body_path}"])
    assert result.exit_code == 0, result.output
    sent = json.loads(requests_mock.request_history[0].text)
    assert sent == {"content": "from-file"}


# ---------------------------------------------------------------------------
# generated-prompt mark-successful
# ---------------------------------------------------------------------------


def test_generated_prompt_mark_successful(runner: CliRunner, requests_mock) -> None:
    requests_mock.patch(f"{_JAILBREAK}/generated-prompts/gp1/success", json={"ok": True})
    result = runner.invoke(cli, ["redteam", "generated-prompt", "mark-successful", "gp1"])
    assert result.exit_code == 0
    assert requests_mock.request_history[0].method == "PATCH"
    assert '"ok": true' in result.output


# ---------------------------------------------------------------------------
# breach list / get
# ---------------------------------------------------------------------------


def test_breach_list(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(
        f"{_TESTING}/runs/run-1/results/breaches",
        json=[{"id": "b1"}, {"id": "b2"}],
    )
    result = runner.invoke(cli, ["redteam", "breach", "list", "run-1"])
    assert result.exit_code == 0
    assert '"id": "b1"' in result.output


def test_breach_get_filters_by_id(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(
        f"{_TESTING}/runs/run-1/results/breaches",
        json=[{"id": "b1", "detail": "one"}, {"id": "b2", "detail": "two"}],
    )
    result = runner.invoke(cli, ["redteam", "breach", "get", "run-1", "b2"])
    assert result.exit_code == 0, result.output
    assert '"detail": "two"' in result.output


def test_breach_get_missing_errors(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(f"{_TESTING}/runs/run-1/results/breaches", json=[{"id": "b1"}])
    result = runner.invoke(cli, ["redteam", "breach", "get", "run-1", "nope"])
    assert result.exit_code != 0
    assert "not found" in result.output
