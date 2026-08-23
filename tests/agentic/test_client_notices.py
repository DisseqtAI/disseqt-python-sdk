"""
Tests for the missing-application_id notice surfaced at client
construction time.

Uses the same channel contract as `disseqt_sdk._version` (v0.8.0):
plain stdlib logger under the package name, opt-out via env var or
raising the logger level.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from disseqt_agentic_sdk import DisseqtAgenticClient
from disseqt_agentic_sdk._notices import (
    APPLICATIONS_REGISTRY_DOCS_URL,
    _reset_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_and_stub(monkeypatch):
    """Reset one-shot flag and stub transport/buffer so tests don't hit the network."""
    _reset_for_tests()
    monkeypatch.setattr("disseqt_agentic_sdk.client.client.HTTPTransport", MagicMock())
    monkeypatch.setattr("disseqt_agentic_sdk.client.client.TraceBuffer", MagicMock())
    yield
    _reset_for_tests()


def _make_client(**overrides):
    kwargs = {
        "api_key": "test_key",
        "project_id": "test_proj",
        "service_name": "test_service",
        "endpoint": "http://localhost/v1/traces",
    }
    kwargs.update(overrides)
    return DisseqtAgenticClient(**kwargs)


def _application_id_notices(records):
    """Filter caplog records down to just our notice."""
    return [
        r
        for r in records
        if r.name == "disseqt_agentic_sdk"
        and r.levelno == logging.WARNING
        and "application_id" in r.getMessage()
    ]


class TestApplicationIdNotice:
    def test_notifies_when_application_id_missing(self, caplog):
        caplog.set_level(logging.WARNING, logger="disseqt_agentic_sdk")
        _make_client()
        matches = _application_id_notices(caplog.records)
        assert len(matches) == 1
        assert APPLICATIONS_REGISTRY_DOCS_URL in matches[0].getMessage()

    def test_silent_when_application_id_provided(self, caplog):
        caplog.set_level(logging.WARNING, logger="disseqt_agentic_sdk")
        _make_client(application_id="7ce57144-9df6-4fa4-8aad-8cbc1ffdb558")
        assert _application_id_notices(caplog.records) == []

    def test_notifies_when_application_id_is_whitespace_only(self, caplog):
        """Empty/whitespace normalises to None → should notify."""
        caplog.set_level(logging.WARNING, logger="disseqt_agentic_sdk")
        _make_client(application_id="   ")
        assert len(_application_id_notices(caplog.records)) == 1

    def test_one_shot_across_multiple_clients(self, caplog):
        """Constructing three clients without application_id should notify only once."""
        caplog.set_level(logging.WARNING, logger="disseqt_agentic_sdk")
        _make_client()
        _make_client()
        _make_client()
        assert len(_application_id_notices(caplog.records)) == 1

    def test_env_var_suppresses_notice(self, monkeypatch, caplog):
        """The env var is read at import time — patch and reload the module to test."""
        import importlib

        monkeypatch.setenv("DISSEQT_SDK_DISABLE_APPLICATION_ID_NOTICE", "1")
        # Reload so the module re-evaluates its env-cached constant.
        import disseqt_agentic_sdk._notices as notices_mod

        importlib.reload(notices_mod)
        _reset_for_tests()
        try:
            caplog.set_level(logging.WARNING, logger="disseqt_agentic_sdk")
            _make_client()
            assert _application_id_notices(caplog.records) == []
        finally:
            # Reload once more with the env var gone to restore module state.
            monkeypatch.delenv("DISSEQT_SDK_DISABLE_APPLICATION_ID_NOTICE", raising=False)
            importlib.reload(notices_mod)

    def test_raising_logger_level_silences_notice(self, caplog):
        """Users can silence by raising the logger level to ERROR."""
        caplog.set_level(logging.ERROR, logger="disseqt_agentic_sdk")
        logging.getLogger("disseqt_agentic_sdk").setLevel(logging.ERROR)
        try:
            _make_client()
            assert _application_id_notices(caplog.records) == []
        finally:
            logging.getLogger("disseqt_agentic_sdk").setLevel(logging.NOTSET)
