"""Unit tests for :mod:`disseqt_sdk.auth.token_store`.

Perm checks are POSIX-only (skipped on Windows). Uses ``monkeypatch`` to
point ``CONFIG_DIR`` / ``CONFIG_PATH`` at a per-test tmp path so the
suite never touches the real ``~/.disseqt/``.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from disseqt_sdk.auth import token_store


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config_dir = tmp_path / ".disseqt"
    config_path = config_dir / "config.json"
    monkeypatch.setattr(token_store, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(token_store, "CONFIG_PATH", config_path)
    return config_path


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode) & 0o777


def test_load_returns_none_when_no_file() -> None:
    assert token_store.load() is None


def test_save_creates_dir_and_file_with_correct_modes() -> None:
    token_store.save({"api_key": "sk_x", "project_id": "p1"})
    config_path = token_store.CONFIG_PATH
    assert config_path.exists()
    if os.name == "posix":
        assert _mode(config_path) == 0o600
        assert _mode(token_store.CONFIG_DIR) == 0o700
    data = json.loads(config_path.read_text())
    assert data == {"auth": {"api_key": "sk_x", "project_id": "p1"}}


def test_load_roundtrips_saved_auth() -> None:
    token_store.save({"api_key": "sk_x", "project_id": "p1", "base_url": "u"})
    assert token_store.load() == {"api_key": "sk_x", "project_id": "p1", "base_url": "u"}


@pytest.mark.skipif(os.name != "posix", reason="perm check is POSIX-only")
def test_load_refuses_wider_perms() -> None:
    token_store.save({"api_key": "sk", "project_id": "p"})
    os.chmod(token_store.CONFIG_PATH, 0o644)
    with pytest.raises(token_store.AuthConfigPermissionError):
        token_store.load()


def test_clear_is_idempotent() -> None:
    token_store.clear()  # missing file, no error
    token_store.save({"api_key": "sk", "project_id": "p"})
    token_store.clear()
    assert not token_store.CONFIG_PATH.exists()
    token_store.clear()  # second call still fine


def test_save_overwrites_existing() -> None:
    token_store.save({"api_key": "old", "project_id": "p"})
    token_store.save({"api_key": "new", "project_id": "p"})
    assert token_store.load() == {"api_key": "new", "project_id": "p"}
