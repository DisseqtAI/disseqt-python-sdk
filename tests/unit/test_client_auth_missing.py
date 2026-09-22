"""Client raises :class:`AuthMissingError` at construction when no creds
resolve from kwargs or ``~/.disseqt/config.json`` — failing where the
misconfiguration is caused, not on the first API call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from disseqt_sdk import Client
from disseqt_sdk.auth import AuthMissingError, token_store


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the token store at an empty tmp path so no dev's real
    ``~/.disseqt/config.json`` can mask the missing-creds case."""
    config_dir = tmp_path / ".disseqt"
    config_path = config_dir / "config.json"
    monkeypatch.setattr(token_store, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(token_store, "CONFIG_PATH", config_path)
    return config_path


def test_client_raises_auth_missing_when_no_creds_anywhere() -> None:
    with pytest.raises(AuthMissingError) as exc:
        Client()
    msg = str(exc.value)
    assert "project_id" in msg or "api_key" in msg
    assert "disseqt login" in msg


def test_client_raises_when_only_project_id_provided() -> None:
    with pytest.raises(AuthMissingError):
        Client(project_id="proj_1")


def test_client_raises_when_only_api_key_provided() -> None:
    with pytest.raises(AuthMissingError):
        Client(api_key="dsq_test")


def test_client_treats_empty_string_creds_as_missing() -> None:
    with pytest.raises(AuthMissingError):
        Client(project_id="", api_key="")


def test_client_still_works_with_explicit_kwargs() -> None:
    c = Client(project_id="proj_1", api_key="dsq_test")
    assert c.project_id == "proj_1"
    assert c.api_key == "dsq_test"


def test_client_still_works_with_stored_config() -> None:
    token_store.save({"api_key": "sk_stored", "project_id": "p_stored"})
    c = Client()
    assert c.project_id == "p_stored"
    assert c.api_key == "sk_stored"


def test_login_command_does_not_construct_client_prematurely() -> None:
    """`disseqt login --help` must not trip the new construction check —
    the login path never calls :class:`Client`."""
    from click.testing import CliRunner

    from disseqt_sdk.cli import cli

    result = CliRunner().invoke(cli, ["login", "--help"])
    assert result.exit_code == 0
    assert "login" in result.output.lower()
