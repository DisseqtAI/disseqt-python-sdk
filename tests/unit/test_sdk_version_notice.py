"""Tests for SDK version identity headers and the server-driven upgrade notice.

Covers the test matrix from the "SDK Version Notification — Detailed
Design" doc: request headers, warn-once semantics, notice text, malformed
headers, opt-out, fail-open, and the single-sourced version.
"""

import importlib
import logging

import pytest
from requests_mock import ANY

import disseqt_sdk._version as _version
from disseqt_sdk import Client, DisseqtAPIClient
from disseqt_sdk._version import SDK_VERSION
from disseqt_sdk.validators.input.safety import ToxicityValidator

_OK_BODY = {"data": {}, "status": {"code": "200"}}


@pytest.fixture(autouse=True)
def _fresh_warned_versions(monkeypatch):
    """Each test starts with a clean warn-once state."""
    monkeypatch.setattr(_version, "_warned_versions", set())


def _notice_records(caplog):
    return [r for r in caplog.records if r.name == "disseqt_sdk"]


class TestIdentityHeaders:
    """The SDK identifies itself on every request."""

    def test_identity_header_values(self):
        headers = _version.sdk_identity_headers()
        assert headers["X-SDK-Version"] == SDK_VERSION
        assert headers["User-Agent"] == f"disseqt-ai-sdk/{SDK_VERSION}"

    def test_validate_sends_version_headers(
        self, requests_mock, client, config, input_validation_request
    ):
        requests_mock.post(ANY, json=_OK_BODY)
        client.validate(ToxicityValidator(data=input_validation_request, config=config))

        sent = requests_mock.request_history[0].headers
        assert sent["X-SDK-Version"] == SDK_VERSION
        assert sent["User-Agent"] == f"disseqt-ai-sdk/{SDK_VERSION}"

    def test_policy_evaluate_sends_version_headers(self, requests_mock, input_validation_request):
        client = Client(
            project_id="p",
            api_key="k",
            base_url="https://test-api.disseqt.ai",
            realtime_policy_base_url="https://test-api.disseqt.ai",
            application_name="test-app",
        )
        requests_mock.post(ANY, json={"decision": "PASS"})
        client.validate(input_validation_request, policies=["policy-1"])

        sent = requests_mock.request_history[0].headers
        assert sent["X-SDK-Version"] == SDK_VERSION
        assert sent["User-Agent"] == f"disseqt-ai-sdk/{SDK_VERSION}"

    def test_prompt_packs_client_sends_version_headers(self, requests_mock):
        api = DisseqtAPIClient(project_id="p", api_key="k", base_url="https://pp.example")
        requests_mock.get(ANY, json={"data": []})
        api.list_runs("pack-1")

        sent = requests_mock.request_history[0].headers
        assert sent["X-SDK-Version"] == SDK_VERSION
        assert sent["User-Agent"] == f"disseqt-ai-sdk/{SDK_VERSION}"


class TestVersionNotice:
    """Server-advertised newer versions surface as a warn-once log line."""

    def test_warns_when_server_advertises_newer_version(
        self, requests_mock, client, config, input_validation_request, caplog
    ):
        caplog.set_level(logging.WARNING, logger="disseqt_sdk")
        requests_mock.post(ANY, json=_OK_BODY, headers={"X-SDK-Latest-Version": "9.9.9"})
        validator = ToxicityValidator(data=input_validation_request, config=config)

        result = client.validate(validator)

        assert result == _OK_BODY  # the notice never alters the response
        records = _notice_records(caplog)
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        message = records[0].getMessage()
        assert f"disseqt-ai-sdk {SDK_VERSION} is outdated" in message
        assert "9.9.9 is available" in message
        assert "pip install -U disseqt-ai-sdk" in message

    def test_warns_only_once_across_repeated_calls(
        self, requests_mock, client, config, input_validation_request, caplog
    ):
        caplog.set_level(logging.WARNING, logger="disseqt_sdk")
        requests_mock.post(ANY, json=_OK_BODY, headers={"X-SDK-Latest-Version": "9.9.9"})
        validator = ToxicityValidator(data=input_validation_request, config=config)

        for _ in range(5):
            client.validate(validator)

        assert len(_notice_records(caplog)) == 1

    def test_new_advertised_version_warns_again(
        self, requests_mock, client, config, input_validation_request, caplog
    ):
        caplog.set_level(logging.WARNING, logger="disseqt_sdk")
        validator = ToxicityValidator(data=input_validation_request, config=config)

        requests_mock.post(ANY, json=_OK_BODY, headers={"X-SDK-Latest-Version": "9.9.9"})
        client.validate(validator)
        requests_mock.post(ANY, json=_OK_BODY, headers={"X-SDK-Latest-Version": "10.0.0"})
        client.validate(validator)

        messages = [r.getMessage() for r in _notice_records(caplog)]
        assert len(messages) == 2
        assert "9.9.9 is available" in messages[0]
        assert "10.0.0 is available" in messages[1]

    def test_notice_header_text_is_appended(
        self, requests_mock, client, config, input_validation_request, caplog
    ):
        caplog.set_level(logging.WARNING, logger="disseqt_sdk")
        notice = "0.6.x is unsupported and will be blocked after 2026-09-01"
        requests_mock.post(
            ANY,
            json=_OK_BODY,
            headers={"X-SDK-Latest-Version": "9.9.9", "X-SDK-Notice": notice},
        )
        client.validate(ToxicityValidator(data=input_validation_request, config=config))

        records = _notice_records(caplog)
        assert len(records) == 1
        assert records[0].getMessage().endswith(notice)

    def test_policy_evaluate_path_also_warns(self, requests_mock, input_validation_request, caplog):
        caplog.set_level(logging.WARNING, logger="disseqt_sdk")
        client = Client(
            project_id="p",
            api_key="k",
            base_url="https://test-api.disseqt.ai",
            realtime_policy_base_url="https://test-api.disseqt.ai",
            application_name="test-app",
        )
        requests_mock.post(
            ANY, json={"decision": "PASS"}, headers={"X-SDK-Latest-Version": "9.9.9"}
        )
        client.validate(input_validation_request, policies=["policy-1"])

        assert len(_notice_records(caplog)) == 1

    def test_prompt_packs_client_also_warns(self, requests_mock, caplog):
        caplog.set_level(logging.WARNING, logger="disseqt_sdk")
        api = DisseqtAPIClient(project_id="p", api_key="k", base_url="https://pp.example")
        requests_mock.get(ANY, json={"data": []}, headers={"X-SDK-Latest-Version": "9.9.9"})
        api.list_runs("pack-1")

        assert len(_notice_records(caplog)) == 1


class TestNoticeSafety:
    """The notice channel is fail-open and opt-out-able."""

    def test_silent_without_version_headers(
        self, requests_mock, client, config, input_validation_request, caplog
    ):
        caplog.set_level(logging.WARNING, logger="disseqt_sdk")
        requests_mock.post(ANY, json=_OK_BODY)
        client.validate(ToxicityValidator(data=input_validation_request, config=config))

        assert _notice_records(caplog) == []

    @pytest.mark.parametrize("malformed", ["", "   "])
    def test_silent_on_malformed_header(self, malformed, caplog):
        caplog.set_level(logging.WARNING, logger="disseqt_sdk")
        _version.check_version_notice({"X-SDK-Latest-Version": malformed})

        assert _notice_records(caplog) == []

    def test_silent_on_broken_headers_object(self, caplog):
        caplog.set_level(logging.WARNING, logger="disseqt_sdk")

        class BrokenHeaders:
            def get(self, key, default=None):
                raise RuntimeError("boom")

        _version.check_version_notice(BrokenHeaders())

        assert _notice_records(caplog) == []

    def test_notice_failure_never_breaks_validation(
        self, requests_mock, client, config, input_validation_request, monkeypatch
    ):
        class ExplodingLogger:
            def warning(self, *args, **kwargs):
                raise RuntimeError("logging blew up")

        monkeypatch.setattr(_version, "_notice_logger", ExplodingLogger())
        requests_mock.post(ANY, json=_OK_BODY, headers={"X-SDK-Latest-Version": "9.9.9"})

        result = client.validate(ToxicityValidator(data=input_validation_request, config=config))

        assert result == _OK_BODY

    def test_opt_out_env_var_suppresses_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger="disseqt_sdk")
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("DISSEQT_SDK_DISABLE_VERSION_NOTICE", "1")
            importlib.reload(_version)
            _version.check_version_notice({"X-SDK-Latest-Version": "9.9.9"})
            assert _notice_records(caplog) == []
            # The identity headers are unaffected — telemetry keeps working.
            assert _version.sdk_identity_headers()["X-SDK-Version"] == SDK_VERSION
        # Env var gone again: restore module-level state for other tests.
        importlib.reload(_version)
        assert _version._NOTICE_DISABLED is False


class TestSingleSourcedVersion:
    """Both packages report the installed distribution's version."""

    def test_packages_agree_on_version(self):
        import disseqt_agentic_sdk
        import disseqt_sdk
        from disseqt_agentic_sdk.client import DisseqtAgenticClient

        assert disseqt_sdk.__version__ == SDK_VERSION
        assert disseqt_agentic_sdk.__version__ == SDK_VERSION
        assert DisseqtAgenticClient.SDK_VERSION == SDK_VERSION

    def test_agentic_version_drift_is_gone(self):
        import disseqt_agentic_sdk

        assert disseqt_agentic_sdk.__version__ != "0.1.0"
