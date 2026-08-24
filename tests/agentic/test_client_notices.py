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


class TestApplicationIdValidation:
    """
    TP-2128 round-2 P2 #2.3: a malformed application_id (embedded line
    breaks, non-Latin-1 characters) breaks sending it as an HTTP header,
    so every send fails, never hits the CRITICAL auth-failure path from
    #1.3, and combined with retain-on-failure (#1.2) the buffer retries
    forever silently. Validate at construction so operators find the
    misconfiguration up front.

    Round-3 senior review P1 #1.1: the original check was a C0-control
    character blacklist that didn't match what `requests`/`http.client`
    actually reject — it rejected harmless characters (tab) while
    letting the real crash-risk class (non-Latin-1 characters, which
    crash HTTP header encoding) through untouched. Tightened to check
    the two real failure modes directly: embedded \\r/\\n (requests'
    InvalidHeader) and non-Latin-1 characters (http.client's
    UnicodeEncodeError at send time).
    """

    def test_newline_raises(self):
        with pytest.raises(ValueError, match="carriage return or newline"):
            _make_client(application_id="app-id-with-\nnewline")

    def test_carriage_return_raises(self):
        # CRLF injection variant — also caught locally by `requests`
        # today, but we prefer fail-fast at construction.
        with pytest.raises(ValueError, match="carriage return or newline"):
            _make_client(application_id="app\r\nX-Injected: 1")

    def test_non_latin1_character_raises(self):
        # TP-2128 round-3 P1 #1.1: this is the class of value that used
        # to sail through validation and then crash HTTP header encoding
        # (http.client.putheader's .encode("latin-1")) at actual send
        # time, uncaught, deep in the background flush path.
        with pytest.raises(ValueError, match="Latin-1"):
            _make_client(application_id="app-\U0001f525-id")

    def test_tab_accepted(self):
        # TP-2128 round-3 P1 #1.1: a tab is NOT rejected by `requests`
        # over the wire (verified against the real installed library),
        # so this must no longer raise — the old blacklist rejected it
        # incorrectly, failing fast against a value that would have
        # succeeded.
        _make_client(application_id="app\tid")

    def test_valid_uuid_accepted(self):
        # A normal UUID must construct without raising.
        _make_client(application_id="7ce57144-9df6-4fa4-8aad-8cbc1ffdb558")

    def test_none_accepted(self):
        # None (missing) still just triggers the notice, not a raise.
        _make_client(application_id=None)

    def test_whitespace_only_accepted(self):
        # Whitespace-only normalises to None (documented behavior),
        # which triggers the notice but must NOT raise.
        _make_client(application_id="   ")
