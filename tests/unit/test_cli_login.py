"""CLI tests for ``disseqt login`` / ``disseqt logout``.

Uses Click's ``CliRunner`` in-process; ``requests_mock`` intercepts the
smoke-test / revoke HTTP calls. The token store is redirected to a
tmp path so no test ever touches the real ``~/.disseqt/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from disseqt_sdk.auth import token_store
from disseqt_sdk.cli import cli

BASE_URL = "https://api.disseqt.ai/realtime-validations"
API_KEYS_URL = f"{BASE_URL}/api/v1/users/me/api-keys"


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config_dir = tmp_path / ".disseqt"
    config_path = config_dir / "config.json"
    monkeypatch.setattr(token_store, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(token_store, "CONFIG_PATH", config_path)
    return config_path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_login_with_flags_saves_config_on_200(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(API_KEYS_URL, json={"data": []})
    result = runner.invoke(
        cli,
        ["login", "--api-key", "dsq_fake_abcdef12345", "--project-id", "proj_1"],
    )
    assert result.exit_code == 0, result.output
    stored = token_store.load()
    assert stored == {
        "api_key": "dsq_fake_abcdef12345",
        "project_id": "proj_1",
        "base_url": BASE_URL,
    }
    # Token value never printed.
    assert "dsq_fake_abcdef12345" not in result.output


def test_login_masks_key_in_json_output(runner: CliRunner, requests_mock) -> None:
    requests_mock.get(API_KEYS_URL, json={"data": []})
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
    requests_mock.get(API_KEYS_URL, status_code=401, text="unauthorized")
    result = runner.invoke(
        cli,
        ["login", "--api-key", "sk_bad", "--project-id", "proj_1"],
    )
    assert result.exit_code == 2
    assert "invalid API key" in result.output.lower() or "invalid api key" in result.output.lower()
    assert token_store.load() is None
    assert "sk_bad" not in result.output


def test_login_rejects_non_tty_without_flags(runner: CliRunner) -> None:
    # CliRunner's default stdin is non-interactive; omit --api-key so we
    # would fall into the interactive branch.
    result = runner.invoke(cli, ["login", "--project-id", "proj_1"])
    assert result.exit_code == 2
    assert "non-interactive" in result.output.lower()


def test_logout_local_only_clears_config(runner: CliRunner) -> None:
    token_store.save({"api_key": "sk", "project_id": "p", "base_url": BASE_URL})
    result = runner.invoke(cli, ["logout", "--local-only"])
    assert result.exit_code == 0
    assert token_store.load() is None


def test_logout_revokes_matching_key_server_side(runner: CliRunner, requests_mock) -> None:
    fake_pat = "dsq_live_1234567890abcdef"
    token_store.save({"api_key": fake_pat, "project_id": "p", "base_url": BASE_URL})
    requests_mock.get(
        API_KEYS_URL,
        json={
            "data": [
                {"id": "key-a", "key_prefix": "dsq_live_123"},
                {"id": "key-b", "key_prefix": "dsq_fake_999"},
            ]
        },
    )
    delete_mock = requests_mock.delete(f"{API_KEYS_URL}/key-a", status_code=204)
    result = runner.invoke(cli, ["logout"])
    assert result.exit_code == 0, result.output
    assert delete_mock.called_once
    assert token_store.load() is None
    assert fake_pat not in result.output


def test_logout_when_not_logged_in(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["logout"])
    assert result.exit_code == 0
    assert "not logged in" in result.output.lower()


def test_logout_still_clears_local_on_revoke_failure(runner: CliRunner, requests_mock) -> None:
    token_store.save({"api_key": "sk_x", "project_id": "p", "base_url": BASE_URL})
    requests_mock.get(API_KEYS_URL, status_code=500, text="boom")
    result = runner.invoke(cli, ["logout"])
    assert result.exit_code == 0
    assert token_store.load() is None
    assert "local config cleared" in result.output.lower()


def test_client_reads_stored_auth_when_no_kwargs() -> None:
    from disseqt_sdk import Client

    token_store.save({"api_key": "sk_stored", "project_id": "p_stored", "base_url": BASE_URL})
    client = Client()
    assert client.api_key == "sk_stored"
    assert client.project_id == "p_stored"
    assert client.base_url == BASE_URL


def test_client_explicit_kwargs_override_config() -> None:
    from disseqt_sdk import Client

    token_store.save({"api_key": "sk_stored", "project_id": "p_stored"})
    client = Client(project_id="explicit_p", api_key="explicit_k")
    assert client.api_key == "explicit_k"
    assert client.project_id == "explicit_p"
