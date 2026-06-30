"""Tests for the agentic SDK logging wrapper (delegates to disseqt_logging)."""

import io
import json
import logging
from unittest.mock import patch

import pytest

import disseqt_logging
from disseqt_agentic_sdk.utils.logging import (
    DEFAULT_LOGGER_NAME,
    get_logger,
    set_log_level,
)
from disseqt_logging import Logger


@pytest.fixture
def cap():
    """Configure the shared logger to a JSON buffer; silence it again after."""
    buf = io.StringIO()
    disseqt_logging.configure(level="debug", fmt="json", stream=buf, redact=True)
    yield buf
    disseqt_logging.disable()


def _lines(buf):
    return [json.loads(line) for line in buf.getvalue().strip().splitlines() if line.strip()]


class TestAgenticLogging:
    def test_get_logger_returns_shared_logger(self):
        assert isinstance(get_logger(), Logger)
        assert isinstance(get_logger("custom"), Logger)

    def test_default_logger_name(self):
        assert DEFAULT_LOGGER_NAME == "disseqt_agentic_sdk"

    def test_emits_structured_json(self, cap):
        get_logger("disseqt_agentic_sdk.transport").info("buffer flush", span_count=3)
        line = _lines(cap)[-1]
        assert line["event"] == "buffer flush"
        assert line["span_count"] == 3
        assert line["service"] == "disseqt-ai-sdk"
        assert line["logger"] == "disseqt_agentic_sdk.transport"

    def test_extra_and_exc_info(self, cap):
        log = get_logger("disseqt_agentic_sdk.transport")
        try:
            raise ValueError("boom")
        except ValueError:
            log.error("Failed to send spans", extra={"endpoint": "https://x"}, exc_info=True)
        line = _lines(cap)[-1]
        assert line["error"] == "boom"
        assert line["error_type"] == "ValueError"
        assert "Traceback" in line["exception"]
        assert line["endpoint"] == "https://x"

    def test_warning_alias_works(self, cap):
        get_logger("disseqt_agentic_sdk.span").warning("retrying")
        assert _lines(cap)[-1]["level"] == "warning"

    def test_redacts_secrets(self, cap):
        get_logger("disseqt_agentic_sdk.client").info(
            "DisseqtAgenticClient initialized",
            extra={
                "project_id": "670bb08f-secret",
                "api_key": "dsk_secret",
                "endpoint": "https://x",
            },
        )
        line = _lines(cap)[-1]
        assert line["project_id"] == "[REDACTED]"
        assert line["api_key"] == "[REDACTED]"
        assert line["endpoint"] == "https://x"  # non-sensitive survives

    def test_set_log_level_filters(self, cap):
        set_log_level("warning")
        log = get_logger("disseqt_agentic_sdk.buffer")
        log.info("hidden")
        log.warning("shown")
        events = [line["event"] for line in _lines(cap)]
        assert "hidden" not in events
        assert "shown" in events

    def test_set_log_level_accepts_str_and_int(self, cap):
        set_log_level(logging.WARNING)
        assert disseqt_logging.current_level() == "warn"
        set_log_level("debug")
        assert disseqt_logging.current_level() == "debug"

    def test_set_log_level_invalid_falls_back_to_info(self, cap):
        set_log_level(object())  # type: ignore[arg-type]
        assert disseqt_logging.current_level() == "info"


class TestAgenticClientDoesNotLogProjectId:
    """Defense-in-depth: the client init must not emit project_id even unredacted."""

    def test_init_omits_project_id_even_without_redaction(self):
        from disseqt_agentic_sdk.client.client import DisseqtAgenticClient

        buf = io.StringIO()
        disseqt_logging.configure(level="info", fmt="json", stream=buf, redact=False)
        try:
            with (
                patch("disseqt_agentic_sdk.client.client.HTTPTransport"),
                patch("disseqt_agentic_sdk.client.client.TraceBuffer"),
            ):
                client = DisseqtAgenticClient(
                    api_key="test_key",
                    project_id="PID-SENTINEL-123",
                    service_name="svc",
                    endpoint="http://localhost:8080/v1/traces",
                )
                client.shutdown()
            text = buf.getvalue()
            assert "DisseqtAgenticClient initialized" in text
            assert "PID-SENTINEL-123" not in text  # never logged, redaction off
        finally:
            disseqt_logging.disable()
