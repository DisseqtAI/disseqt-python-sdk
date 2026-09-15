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
        "plan",
        "plan-run",
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


def test_pack_export_delegates_to_sdk(runner: CliRunner, requests_mock) -> None:
    """`pack export` delegates to PacksResource.download and prints CSV text."""
    csv_body = b"id,prompt\n1,hello\n"
    requests_mock.get(
        f"{DATASET_BASE}/api/v1/prompt-packs/p1/download",
        content=csv_body,
        headers={"Content-Type": "text/csv"},
    )
    result = runner.invoke(cli, ["pack", "export", "p1"])
    assert result.exit_code == 0, result.output
    assert "id,prompt" in result.output


def test_pp_run_reveal(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(
        f"{DATASET_BASE}/api/v1/prompt-packs/runs/r1/results/o1/reveal",
        json={"revealed": True},
    )
    result = runner.invoke(cli, ["pp-run", "reveal", "r1", "o1"])
    assert result.exit_code == 0, result.output
    assert "revealed" in result.output


def test_pp_run_add_to_pack(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(f"{DATASET_BASE}/api/v1/prompt-packs/p1/prompts/add", json={"added": 2})
    result = runner.invoke(
        cli,
        [
            "pp-run",
            "add-to-pack",
            "r1",
            "--output-id",
            "o1",
            "--output-id",
            "o2",
            "--pack-id",
            "p1",
        ],
    )
    assert result.exit_code == 0, result.output
    sent = requests_mock.request_history[0].json()
    assert sent == {"run_id": "r1", "output_ids": ["o1", "o2"]}


def test_whoami_text_output(runner: CliRunner, monkeypatch) -> None:
    monkeypatch.setenv("DISSEQT_PROJECT_ID", "proj_123")
    monkeypatch.setenv("DISSEQT_API_KEY", "sk_test_abcdef")
    monkeypatch.setenv("DISSEQT_USER_EMAIL", "u@example.com")
    monkeypatch.delenv("DISSEQT_ORGANIZATION_ID", raising=False)
    result = runner.invoke(cli, ["whoami"])
    assert result.exit_code == 0, result.output
    assert "proj_123" in result.output
    assert "u@example.com" in result.output
    # API key is masked — never the full value.
    assert "sk_test_abcdef" not in result.output
    assert "sk_...def" in result.output
    assert "(unset)" in result.output  # organization_id


def test_whoami_json_output(runner: CliRunner, monkeypatch) -> None:
    monkeypatch.setenv("DISSEQT_PROJECT_ID", "proj_x")
    monkeypatch.setenv("DISSEQT_API_KEY", "sk_test_xyz")
    monkeypatch.delenv("DISSEQT_USER_EMAIL", raising=False)
    monkeypatch.delenv("DISSEQT_ORGANIZATION_ID", raising=False)
    result = runner.invoke(cli, ["whoami", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["project_id"] == "proj_x"
    assert payload["api_key"].startswith("sk_") and "..." in payload["api_key"]
    assert payload["user_email"] is None


# ---------------------------------------------------------------------------
# Test Plans + Test Plan Runs — one live call per group
# ---------------------------------------------------------------------------


def test_plan_list_hits_correct_url(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(f"{DATASET_BASE}/api/v1/test-plans", json={"data": []})
    result = runner.invoke(cli, ["plan", "list"])
    assert result.exit_code == 0, result.output


def test_plan_get(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(f"{DATASET_BASE}/api/v1/test-plans/p1", json={"id": "p1"})
    result = runner.invoke(cli, ["plan", "get", "p1"])
    assert result.exit_code == 0
    assert '"id": "p1"' in result.output


def test_plan_publish_sends_body(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(f"{DATASET_BASE}/api/v1/test-plans/p1/publish", json={"id": "p1"})
    body = '{"sharing_scope": "PROJECT", "expected_sharing_scope": "PRIVATE"}'
    result = runner.invoke(cli, ["plan", "publish", "p1", "--json", body])
    assert result.exit_code == 0
    sent = json.loads(requests_mock.request_history[0].text)
    assert sent["sharing_scope"] == "PROJECT"


def test_plan_run_create_no_wait(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(
        f"{DATASET_BASE}/api/v1/test-plans/p1/runs",
        json={"run_id": "r1", "status": "queued"},
    )
    body = '{"target": {"execution_mode": "app_integration", "app_integration_id": "a1"}}'
    result = runner.invoke(cli, ["plan-run", "create", "p1", "--json", body])
    assert result.exit_code == 0
    assert '"run_id": "r1"' in result.output


def test_plan_run_create_with_wait_polls_to_terminal(
    runner: CliRunner, requests_mock, monkeypatch
) -> None:
    """--wait polls plan-run get until a terminal status appears."""
    # Zero-sleep polling so the test stays fast.
    monkeypatch.setattr("disseqt_sdk.cli.resources.time.sleep", lambda _s: None)
    requests_mock.post(
        f"{DATASET_BASE}/api/v1/test-plans/p1/runs",
        json={"run_id": "r1", "status": "queued"},
    )
    requests_mock.get(
        f"{DATASET_BASE}/api/v1/test-plan-runs/r1",
        [
            {"json": {"run_id": "r1", "status": "running"}},
            {"json": {"run_id": "r1", "status": "completed"}},
        ],
    )
    body = '{"target": {"execution_mode": "app_integration", "app_integration_id": "a1"}}'
    result = runner.invoke(
        cli, ["plan-run", "create", "p1", "--json", body, "--wait", "--interval", "0"]
    )
    assert result.exit_code == 0, result.output
    assert '"status": "completed"' in result.output


def test_plan_run_cancel(runner: CliRunner, requests_mock) -> None:
    requests_mock.post(
        f"{DATASET_BASE}/api/v1/test-plan-runs/r1/cancel",
        json={"run_id": "r1", "status": "cancelled"},
    )
    result = runner.invoke(cli, ["plan-run", "cancel", "r1"])
    assert result.exit_code == 0
    assert '"status": "cancelled"' in result.output


def test_plan_run_trace_requires_prompt_ref(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(
        f"{DATASET_BASE}/api/v1/test-plan-runs/r1/trace",
        json={"prompt_ref": "pr1"},
    )
    result = runner.invoke(cli, ["plan-run", "trace", "r1", "--prompt-ref", "pr1"])
    assert result.exit_code == 0
    assert requests_mock.request_history[0].qs["prompt_ref"] == ["pr1"]
