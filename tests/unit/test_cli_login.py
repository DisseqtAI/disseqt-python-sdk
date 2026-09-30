"""CLI tests for ``disseqt login`` / ``disseqt logout``.

Uses Click's ``CliRunner`` in-process; ``requests_mock`` intercepts the
smoke-test HTTP call. The token store is redirected to a tmp path so no
test ever touches the real ``~/.disseqt/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from disseqt_sdk.auth import token_store
from disseqt_sdk.cli import cli

BASE_URL = "https://api.disseqt.ai/dataset"
SMOKE_URL = f"{BASE_URL}/api/v1/testing/attack-techniques"


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config_dir = tmp_path / ".disseqt"
    config_path = config_dir / "config.json"
    monkeypatch.setattr(token_store, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(token_store, "CONFIG_PATH", config_path)
    monkeypatch.delenv("DISSEQT_BASE_URL", raising=False)
    monkeypatch.delenv("DISSEQT_PROJECT_ID", raising=False)
    monkeypatch.delenv("DISSEQT_API_KEY", raising=False)
    return config_path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_login_with_flags_saves_config_on_200(runner: CliRunner, requests_mock) -> None:
    m = requests_mock.get(SMOKE_URL, json={"status": "success", "data": []})
    result = runner.invoke(
        cli,
        ["login", "--api-key", "dsq_fake_abcdef12345", "--project-id", "proj_1"],
    )
    assert result.exit_code == 0, result.output
    assert token_store.load() == {"api_key": "dsq_fake_abcdef12345", "project_id": "proj_1"}
    # Smoke test uses the user-key contract only.
    hdrs = m.last_request.headers
    assert hdrs["X-API-Key"] == "dsq_fake_abcdef12345"
    assert hdrs["X-Project-Id"] == "proj_1"
    assert "X-Service-API-Key" not in hdrs
    # Token value never printed.
    assert "dsq_fake_abcdef12345" not in result.output


def test_login_honours_base_url_env(runner: CliRunner, requests_mock, monkeypatch) -> None:
    monkeypatch.setenv("DISSEQT_BASE_URL", "https://custom.example")
    m = requests_mock.get("https://custom.example/api/v1/testing/attack-techniques", json=[])
    result = runner.invoke(cli, ["login", "--api-key", "k" * 12, "--project-id", "p"])
    assert result.exit_code == 0, result.output
    assert m.called_once


def test_login_masks_key_in_json_output(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(SMOKE_URL, json=[])
    result = runner.invoke(
        cli,
        ["login", "--api-key", "dsq_fake_abcdef12345", "--project-id", "proj_1", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "logged_in"
    assert payload["project_id"] == "proj_1"
    assert payload["api_key_prefix"] == "dsq_fake..."
    assert "abcdef12345" not in result.output


def test_login_rejects_401_and_does_not_save(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(SMOKE_URL, status_code=401, text="unauthorized")
    result = runner.invoke(cli, ["login", "--api-key", "sk_bad", "--project-id", "proj_1"])
    assert result.exit_code == 2
    assert "invalid api key" in result.output.lower()
    assert token_store.load() is None
    assert "sk_bad" not in result.output


def test_login_5xx_exits_1_and_does_not_save(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(SMOKE_URL, status_code=503, text="down")
    result = runner.invoke(cli, ["login", "--api-key", "sk_x", "--project-id", "proj_1"])
    assert result.exit_code == 1
    assert token_store.load() is None


def test_login_rejects_non_tty_without_flags(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["login", "--project-id", "proj_1"])
    assert result.exit_code == 2
    assert "non-interactive" in result.output.lower()


def test_logout_clears_config_without_network(runner: CliRunner, requests_mock) -> None:
    token_store.save({"api_key": "sk", "project_id": "p"})
    result = runner.invoke(cli, ["logout"])
    assert result.exit_code == 0
    assert token_store.load() is None
    assert not requests_mock.called


def test_logout_when_not_logged_in(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["logout"])
    assert result.exit_code == 0
    assert "not logged in" in result.output.lower()


def test_logout_has_no_local_only_flag(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["logout", "--local-only"])
    assert result.exit_code == 2


def test_client_reads_stored_auth_when_no_kwargs() -> None:
    from disseqt_sdk import Client

    token_store.save({"api_key": "sk_stored", "project_id": "p_stored"})
    client = Client()
    assert client.api_key == "sk_stored"
    assert client.project_id == "p_stored"


def test_client_explicit_kwargs_override_config() -> None:
    from disseqt_sdk import Client

    token_store.save({"api_key": "sk_stored", "project_id": "p_stored"})
    client = Client(project_id="explicit_p", api_key="explicit_k")
    assert client.api_key == "explicit_k"
    assert client.project_id == "explicit_p"


def test_stored_login_is_used_by_other_verbs(runner: CliRunner, requests_mock) -> None:
    """After `disseqt login`, verbs work with no DISSEQT_* env vars set."""
    requests_mock.get(SMOKE_URL, json=[])
    assert (
        runner.invoke(cli, ["login", "--api-key", "k" * 12, "--project-id", "proj_1"]).exit_code
        == 0
    )
    m = requests_mock.get(SMOKE_URL, json={"status": "success", "data": [{"key": "b64"}]})
    result = runner.invoke(cli, ["redteam", "list-attacks", "--kind", "single"])
    assert result.exit_code == 0, result.output
    assert m.last_request.headers["X-API-Key"] == "k" * 12
    assert m.last_request.headers["X-Project-Id"] == "proj_1"
    assert "b64" in result.output


def test_no_env_no_config_mentions_login(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["redteam", "list-attacks"])
    assert result.exit_code == 2
    assert "disseqt login" in result.output


def test_whoami_reads_stored_config(runner: CliRunner) -> None:
    token_store.save({"api_key": "sk_stored_abc", "project_id": "p_stored"})
    result = runner.invoke(cli, ["whoami", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["project_id"] == "p_stored"
    assert payload["source"] == "config"
    assert "sk_stored_abc" not in result.output


def test_whoami_refuses_insecure_config(runner: CliRunner, _isolated_config: Path) -> None:
    token_store.save({"api_key": "sk", "project_id": "p"})
    _isolated_config.chmod(0o644)
    result = runner.invoke(cli, ["whoami"])
    assert result.exit_code == 2
    assert "permissions" in result.output and "chmod 600" in result.output


def test_verbs_refuse_insecure_config(runner: CliRunner, _isolated_config: Path) -> None:
    token_store.save({"api_key": "sk", "project_id": "p"})
    _isolated_config.chmod(0o644)
    result = runner.invoke(cli, ["redteam", "list-attacks"])
    assert result.exit_code == 2
    assert "permissions" in result.output
