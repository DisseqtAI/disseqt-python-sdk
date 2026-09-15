"""CLI smoke tests for the new resource groups.

One command per group + a couple of body-parsing tests. Uses Click's
``CliRunner`` to invoke commands in-process; ``requests_mock`` intercepts
the underlying HTTP.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from disseqt_sdk.cli import cli

DATASET_BASE = "https://api.disseqt.ai/dataset"


@pytest.fixture(autouse=True)
def _cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide the auth env vars every _http.request() needs."""
    monkeypatch.setenv("DISSEQT_API_KEY", "test_key")
    monkeypatch.setenv("DISSEQT_PROJECT_ID", "test_project")
    # Force the default base explicitly to avoid stray env pollution.
    monkeypatch.delenv("DISSEQT_DATASET_BASE_URL", raising=False)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Help rendering — one per group
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "group",
    [
        "target",
        "rag-target",
        "mcp-target",
        "pack",
        "pp-run",
        "output-validation",
        "rag-validation",
        "session",
        "byov",
        "vulnerability",
    ],
)
def test_group_help_renders(runner: CliRunner, group: str) -> None:
    result = runner.invoke(cli, [group, "--help"])
    assert result.exit_code == 0
    assert group in result.output.lower() or "commands:" in result.output.lower()


# ---------------------------------------------------------------------------
# One live command per group — verifies http path + method
# ---------------------------------------------------------------------------


def test_target_list_hits_correct_url(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(f"{DATASET_BASE}/api/v1/llm/app-integrations", json={"data": []})
    result = runner.invoke(cli, ["target", "list"])
    assert result.exit_code == 0, result.output
    assert '"data": []' in result.output


def test_target_get_hits_id_url(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(f"{DATASET_BASE}/api/v1/llm/app-integrations/t1", json={"id": "t1"})
    result = runner.invoke(cli, ["target", "get", "t1"])
    assert result.exit_code == 0
    assert '"id": "t1"' in result.output


def test_target_create_with_json_literal(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(f"{DATASET_BASE}/api/v1/llm/app-integrations", json={"id": "t1"})
    result = runner.invoke(cli, ["target", "create", "--json", '{"name": "gpt-4"}'])
    assert result.exit_code == 0
    sent = json.loads(requests_mock.request_history[0].text)
    assert sent == {"name": "gpt-4"}


def test_target_create_with_json_file(runner: CliRunner, requests_mock, tmp_path) -> None:
    file_path = tmp_path / "body.json"
    file_path.write_text('{"name": "from-file"}')
    requests_mock.post(f"{DATASET_BASE}/api/v1/llm/app-integrations", json={"id": "t1"})
    result = runner.invoke(cli, ["target", "create", "--json", f"@{file_path}"])
    assert result.exit_code == 0
    sent = json.loads(requests_mock.request_history[0].text)
    assert sent == {"name": "from-file"}


def test_target_probe(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(
        f"{DATASET_BASE}/api/v1/llm/app-integrations/test-connection", json={"ok": True}
    )
    result = runner.invoke(cli, ["target", "probe", "--json", '{"provider": "openai"}'])
    assert result.exit_code == 0
    assert '"ok": true' in result.output


def test_target_parse_curl(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(
        f"{DATASET_BASE}/api/v1/llm/app-integrations/parse-curl", json={"preview": {}}
    )
    result = runner.invoke(cli, ["target", "parse-curl", "--curl", "curl https://x"])
    assert result.exit_code == 0
    sent = json.loads(requests_mock.request_history[0].text)
    assert sent == {"curl": "curl https://x"}


def test_target_delete(runner: CliRunner, requests_mock) -> None:
    requests_mock.delete(f"{DATASET_BASE}/api/v1/llm/app-integrations/t1", status_code=204)
    result = runner.invoke(cli, ["target", "delete", "t1"])
    assert result.exit_code == 0


def test_rag_target_list(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(f"{DATASET_BASE}/api/v1/rag-integrations", json={"data": []})
    result = runner.invoke(cli, ["rag-target", "list"])
    assert result.exit_code == 0


def test_mcp_target_list(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(f"{DATASET_BASE}/api/v1/mcp-integrations", json={"data": []})
    result = runner.invoke(cli, ["mcp-target", "list"])
    assert result.exit_code == 0


def test_pack_list(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(f"{DATASET_BASE}/api/v1/prompt-packs", json={"data": []})
    result = runner.invoke(cli, ["pack", "list"])
    assert result.exit_code == 0


def test_pack_publish(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(f"{DATASET_BASE}/api/v1/prompt-packs/p1/publish", json={"published": True})
    result = runner.invoke(cli, ["pack", "publish", "p1"])
    assert result.exit_code == 0
    assert '"published": true' in result.output


def test_pack_add_prompts_requires_prompts_key(runner: CliRunner) -> None:
    """A payload without a top-level 'prompts' array must fail fast, before the network."""
    result = runner.invoke(cli, ["pack", "add-prompts", "p1", "--json", "{}"])
    assert result.exit_code != 0
    assert "prompts" in result.output


def test_pp_run_create(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(f"{DATASET_BASE}/api/v1/prompt-packs/p1/runs", json={"id": "r1"})
    result = runner.invoke(cli, ["pp-run", "create", "p1", "--json", '{"provider": "openai"}'])
    assert result.exit_code == 0


def test_pp_run_get(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(f"{DATASET_BASE}/api/v1/prompt-packs/runs/r1", json={"id": "r1"})
    result = runner.invoke(cli, ["pp-run", "get", "r1"])
    assert result.exit_code == 0


def test_pp_run_cancel(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(
        f"{DATASET_BASE}/api/v1/prompt-packs/runs/r1/cancel", json={"cancelled": True}
    )
    result = runner.invoke(cli, ["pp-run", "cancel", "r1"])
    assert result.exit_code == 0


def test_output_validation_create(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(
        f"{DATASET_BASE}/api/v1/prompt-packs/runs/r1/validate-outputs", json={"id": "v1"}
    )
    result = runner.invoke(
        cli, ["output-validation", "create", "r1", "--json", '{"metric_evaluations": []}']
    )
    assert result.exit_code == 0


def test_output_validation_get(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(
        f"{DATASET_BASE}/api/v1/prompt-packs/output-validations/v1", json={"id": "v1"}
    )
    result = runner.invoke(cli, ["output-validation", "get", "v1"])
    assert result.exit_code == 0


def test_rag_validation_get(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(f"{DATASET_BASE}/api/v1/prompt-packs/rag-validations/v1", json={"id": "v1"})
    result = runner.invoke(cli, ["rag-validation", "get", "v1"])
    assert result.exit_code == 0


def test_session_create(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(f"{DATASET_BASE}/api/v1/testing/sessions", json={"id": "s1"})
    result = runner.invoke(cli, ["session", "create", "--json", '{"target": {"id": "t1"}}'])
    assert result.exit_code == 0


def test_session_run_cancel(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(f"{DATASET_BASE}/api/v1/testing/runs/r1/cancel", json={"cancelled": True})
    result = runner.invoke(cli, ["session", "run-cancel", "r1"])
    assert result.exit_code == 0


def test_byov_create(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(f"{DATASET_BASE}/api/v1/llm/custom-validators", json={"id": "b1"})
    result = runner.invoke(cli, ["byov", "create", "--json", '{"name": "c"}'])
    assert result.exit_code == 0


def test_byov_probe(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(
        f"{DATASET_BASE}/api/v1/llm/custom-validators/test-connection", json={"ok": True}
    )
    result = runner.invoke(cli, ["byov", "probe", "--json", '{"url": "https://x"}'])
    assert result.exit_code == 0


def test_vulnerability_list(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(f"{DATASET_BASE}/api/v1/vulnerabilities", json={"data": []})
    result = runner.invoke(cli, ["vulnerability", "list"])
    assert result.exit_code == 0


def test_vulnerability_test(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(f"{DATASET_BASE}/api/v1/vulnerabilities/v1/test", json={"job_id": "j1"})
    result = runner.invoke(cli, ["vulnerability", "test", "v1", "--json", '{"target": "t1"}'])
    assert result.exit_code == 0


def test_invalid_json_fails_fast(runner: CliRunner) -> None:
    """A malformed --json argument must exit non-zero without touching the network."""
    result = runner.invoke(cli, ["target", "create", "--json", "{not valid}"])
    assert result.exit_code != 0
    assert "invalid" in result.output.lower()


def test_dataset_base_env_override(runner: CliRunner, requests_mock, monkeypatch) -> None:
    """DISSEQT_DATASET_BASE_URL routes calls to a different host."""
    monkeypatch.setenv("DISSEQT_DATASET_BASE_URL", "https://custom.example")
    requests_mock.get("https://custom.example/api/v1/prompt-packs", json={"data": []})
    result = runner.invoke(cli, ["pack", "list"])
    assert result.exit_code == 0
