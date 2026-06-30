"""Unit tests for the shared ``disseqt_logging`` package + Client.validate logging."""

from __future__ import annotations

import io
import json
import logging
import re

import pytest

import disseqt_logging as dl
from disseqt_logging.config import LoggerConfig, level_name, parse_level
from disseqt_logging.redaction import redact_field, redact_string, sensitive_key


@pytest.fixture
def cap():
    """Configure the shared logger to a JSON buffer; silence it again after."""
    buf = io.StringIO()
    dl.configure(level="debug", fmt="json", stream=buf, redact=True, service="disseqt-ai-sdk")
    yield buf
    dl.disable()


def _last(buf: io.StringIO) -> dict:
    return json.loads(buf.getvalue().strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# Redaction (parity with the platform rules)
# --------------------------------------------------------------------------- #


class TestRedaction:
    def test_content_shapes(self):
        assert redact_string("write to alice@example.com") == "write to [EMAIL]"
        assert redact_string("t eyJa.eyJb.cccc") == "t [JWT]"
        assert redact_string("pan 4111 1111 1111 1111 x") == "pan [CC] x"
        assert redact_string("ph +1 415 555 9876 x") == "ph [PHONE] x"
        assert redact_string("k " + "A" * 40) == "k [TOKEN]"

    def test_cc_runs_before_phone(self):
        # A 16-digit run must read as a card, not a phone number.
        assert "[CC]" in redact_string("4111111111111111")
        assert "[PHONE]" not in redact_string("4111111111111111")

    def test_sensitive_key_substring(self):
        for key in ("api_key", "API-Key", "reset_token", "password", "authorization", "project_id"):
            assert sensitive_key(key), key
        assert not sensitive_key("monkey")  # bare "key" is not a deny-substring
        assert not sensitive_key("status")

    def test_redact_field(self):
        assert redact_field("api_key", "anything") == "[REDACTED]"
        assert redact_field("note", "mail a@b.com") == "mail [EMAIL]"
        assert redact_field("count", 7) == 7  # non-strings pass through


# --------------------------------------------------------------------------- #
# digest
# --------------------------------------------------------------------------- #


class TestDigest:
    def test_known_and_empty(self):
        assert dl.digest("hello world") == "len=11 sha256=b94d27b9934d3e08"
        assert dl.digest("") == "len=0"
        assert dl.digest(b"") == "len=0"

    def test_str_vs_bytes_length(self):
        assert str(dl.digest("héllo")).startswith("len=5 ")
        assert str(dl.digest("héllo".encode())).startswith("len=6 ")

    def test_digest_field_is_not_redacted(self, cap):
        # "a"*227's hash contains a digit run that would otherwise match [PHONE].
        d = dl.digest("a" * 227)
        dl.get_logger("disseqt_sdk.client").info("x", payload_digest=d)
        assert _last(cap)["payload_digest"] == str(d)
        assert "[PHONE]" not in _last(cap)["payload_digest"]


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_parse_and_name(self):
        assert parse_level("debug") == logging.DEBUG
        assert parse_level("warn") == logging.WARNING
        assert parse_level("WARNING") == logging.WARNING
        assert parse_level(logging.ERROR) == logging.ERROR
        assert parse_level("nonsense") == logging.INFO
        assert parse_level(object()) == logging.INFO  # type: ignore[arg-type]
        assert level_name(logging.WARNING) == "warn"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("DISSEQT_LOG_LEVEL", "warn")
        monkeypatch.setenv("DISSEQT_ENV", "stage")
        monkeypatch.setenv("DISSEQT_LOG_REDACT", "0")
        cfg = LoggerConfig.from_env()
        assert cfg.level == "warn"
        assert cfg.env == "stage"
        assert cfg.redact is False

    def test_resolve_format(self):
        assert LoggerConfig(fmt="json").resolve_format() == "json"
        assert LoggerConfig(fmt="console").resolve_format() == "console"
        tty = io.StringIO()
        tty.isatty = lambda: True  # type: ignore[method-assign]
        assert LoggerConfig(fmt="auto", stream=tty).resolve_format() == "console"


# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #


class TestLogger:
    def test_base_fields_and_user_fields(self, cap):
        dl.get_logger("disseqt_sdk.client").info("evt", status=200, latency_ms=1.5)
        line = _last(cap)
        assert line["event"] == "evt"
        assert line["service"] == "disseqt-ai-sdk"
        assert "host" in line and "timestamp" in line
        assert line["status"] == 200 and line["latency_ms"] == 1.5
        assert line["logger"] == "disseqt_sdk.client"

    def test_redacts_secret_fields(self, cap):
        dl.get_logger("t").info("e", api_key="dsk_x", contact="a@b.com", ok="v")
        line = _last(cap)
        assert line["api_key"] == "[REDACTED]"
        assert line["contact"] == "[EMAIL]"
        assert line["ok"] == "v"

    def test_bind_and_with_component(self, cap):
        log = dl.get_logger("t").bind(request_id="r1").with_component("eval")
        log.info("e")
        line = _last(cap)
        assert line["request_id"] == "r1"
        assert line["component"] == "eval"

    def test_error_envelope_from_positional(self, cap):
        class Boom(Exception):
            code = "E_BOOM"

        dl.get_logger("t").error("failed", Boom("nope"))
        line = _last(cap)
        assert line["error"] == "nope"
        assert line["error_type"] == "Boom"
        assert line["error_code"] == "E_BOOM"

    def test_structural_collision_is_renamed(self, cap):
        dl.get_logger("t").info("real-msg", event="user-value", service="user-svc")
        line = _last(cap)
        assert line["event"] == "real-msg"  # message wins
        assert line["service"] == "disseqt-ai-sdk"  # base field wins
        assert line["field_event"] == "user-value"  # caller value preserved
        assert line["field_service"] == "user-svc"

    def test_level_filtering(self, cap):
        dl.set_level("warning")
        log = dl.get_logger("t")
        log.info("hidden")
        log.warning("shown")
        events = [json.loads(x)["event"] for x in cap.getvalue().strip().splitlines()]
        assert "hidden" not in events and "shown" in events
        assert dl.current_level() == "warn"

    def test_console_mode(self):
        buf = io.StringIO()
        dl.configure(level="info", fmt="console", stream=buf)
        dl.get_logger("t").info("hello", key="val", api_key="dsk_x")
        out = buf.getvalue().strip()
        assert "[INFO] hello" in out
        assert "key=val" in out
        assert "api_key=[REDACTED]" in out
        dl.disable()

    def test_get_logger_reparents_name(self, cap):
        # A foreign name is reparented under "disseqt" but rendered without prefix.
        dl.get_logger("disseqt_sdk.client").info("e")
        assert _last(cap)["logger"] == "disseqt_sdk.client"


# --------------------------------------------------------------------------- #
# Client.validate instrumentation (secrets must never leak)
# --------------------------------------------------------------------------- #


class TestClientValidateLogging:
    def _toxicity(self):
        from disseqt_sdk import SDKConfigInput
        from disseqt_sdk.models.input_validation import InputValidationRequest
        from disseqt_sdk.validators.input.safety import ToxicityValidator

        return ToxicityValidator(
            data=InputValidationRequest(prompt="you are worthless"),
            config=SDKConfigInput(threshold=0.5),
        )

    def test_logs_request_and_response_without_secrets(self, cap, requests_mock):
        from disseqt_sdk import Client

        requests_mock.post(re.compile(r"http://test/.*"), json={"ok": True}, status_code=200)
        Client(
            project_id="proj_xyz", api_key="dsk_should_never_log", base_url="http://test"
        ).validate(self._toxicity())
        text = cap.getvalue()
        assert "validation.request" in text
        assert "validation.response" in text
        assert "payload_digest" in text
        assert "latency_ms" in text
        assert "dsk_should_never_log" not in text
        assert "proj_xyz" not in text
        assert "you are worthless" not in text  # prompt never logged verbatim

    def test_logs_http_error(self, cap, requests_mock):
        from disseqt_sdk import Client
        from disseqt_sdk.client import HTTPError

        requests_mock.post(
            re.compile(r"http://test/.*"), json={"detail": "denied"}, status_code=401
        )
        with pytest.raises(HTTPError):
            Client(project_id="proj_xyz", api_key="dsk_secret", base_url="http://test").validate(
                self._toxicity()
            )
        text = cap.getvalue()
        assert "validation.http_error" in text
        assert '"status": 401' in text
        assert "dsk_secret" not in text


# --------------------------------------------------------------------------- #
# Silent by default (opt-in)
# --------------------------------------------------------------------------- #


class TestSilentByDefault:
    @staticmethod
    def _reset_unconfigured():
        """Force the next use to behave like a fresh, unconfigured import."""
        import disseqt_logging.logger as mod

        root = logging.getLogger(mod._ROOT_NAME)
        for handler in list(root.handlers):
            if getattr(handler, mod._OWNED_ATTR, False):
                root.removeHandler(handler)
        mod._configured = False
        mod._active = False
        mod._config = None

    def test_no_output_until_opted_in(self, monkeypatch, capfd):
        monkeypatch.delenv("DISSEQT_LOG_LEVEL", raising=False)
        self._reset_unconfigured()
        import disseqt_logging.logger as mod

        log = dl.get_logger("disseqt_sdk.client")  # first use -> silent NullHandler
        assert mod._active is False
        log.info("should stay silent")
        log.error("also silent")
        out, err = capfd.readouterr()
        assert "should stay silent" not in (out + err)
        assert "also silent" not in (out + err)
        dl.disable()

    def test_set_level_activates_from_silent(self, monkeypatch, capfd):
        monkeypatch.delenv("DISSEQT_LOG_LEVEL", raising=False)
        self._reset_unconfigured()
        import disseqt_logging.logger as mod

        dl.set_level("debug")
        assert mod._active is True
        dl.disable()

    def test_env_level_activates(self, monkeypatch, capfd):
        monkeypatch.setenv("DISSEQT_LOG_LEVEL", "info")
        self._reset_unconfigured()
        dl.get_logger("disseqt_sdk.client").info("env-enabled-line")
        out, err = capfd.readouterr()
        assert "env-enabled-line" in (out + err)
        dl.disable()
