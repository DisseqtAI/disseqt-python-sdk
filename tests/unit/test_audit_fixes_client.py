"""Regression tests for disseqt_sdk.Client audit fixes (M10, M12)."""

from __future__ import annotations

import pytest
import requests

from disseqt_sdk import Client, HTTPError
from disseqt_sdk.client import ResponseDecodeError
from disseqt_sdk.models.base import SDKConfigInput
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.validators.input.safety import ToxicityValidator

BASE = "https://audit.test"
TOX_URL = f"{BASE}/api/v1/sdk/validators/input-validation/toxicity"
P1 = "11111111-1111-4111-8111-111111111111"
P1_URL = f"{BASE}/api/v1/sdk/policies/{P1}/evaluate"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _client(**kw) -> Client:
    args = {
        "project_id": "proj",
        "api_key": "key",
        "base_url": BASE,
        "realtime_policy_base_url": BASE,
        "application_name": "app",
    }
    args.update(kw)
    return Client(**args)


def _tox() -> ToxicityValidator:
    return ToxicityValidator(
        data=InputValidationRequest(prompt="hello"), config=SDKConfigInput(threshold=0.5)
    )


class TestConstructorValidation:
    @pytest.mark.parametrize("field", ["project_id", "api_key"])
    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_empty_rejected(self, field, bad):
        with pytest.raises(ValueError, match=field):
            _client(**{field: bad})

    @pytest.mark.parametrize("field", ["project_id", "api_key"])
    def test_newline_rejected(self, field):
        with pytest.raises(ValueError, match=field):
            _client(**{field: "abc\r\nX-Evil: 1"})

    @pytest.mark.parametrize("field", ["project_id", "api_key"])
    def test_non_latin1_rejected(self, field):
        with pytest.raises(ValueError, match=field):
            _client(**{field: "key’s"})

    def test_unicode_encode_error_wrapped(self, monkeypatch):
        def boom(*a, **k):
            raise UnicodeEncodeError("latin-1", "’", 0, 1, "ordinal not in range")

        monkeypatch.setattr(requests, "post", boom)
        with pytest.raises(HTTPError) as ei:
            _client().validate(_tox())
        assert ei.value.status_code == 0
        monkeypatch.setattr(requests, "post", boom)
        with pytest.raises(HTTPError) as ei2:
            _client()._post_policy_evaluate(P1, {"prompt": "x"}, "app")
        assert ei2.value.status_code == 0


class TestResponseDecoding:
    def test_non_dict_json_names_type_not_body(self, requests_mock):
        requests_mock.post(TOX_URL, json=["secret-user-content"])
        with pytest.raises(ResponseDecodeError) as ei:
            _client().validate(_tox())
        msg = str(ei.value)
        assert "list" in msg and "secret-user-content" not in msg
        assert isinstance(ei.value, HTTPError) and isinstance(ei.value, ValueError)

    def test_null_json(self, requests_mock):
        requests_mock.post(TOX_URL, text="null", headers={"Content-Type": "application/json"})
        with pytest.raises(ValueError, match="NoneType"):
            _client().validate(_tox())

    def test_undecodable_body_reports_digest_not_text(self, requests_mock):
        requests_mock.post(TOX_URL, text="not json secret-body-123")
        with pytest.raises(ValueError) as ei:
            _client().validate(_tox())
        msg = str(ei.value)
        assert "secret-body-123" not in msg and "sha256" in msg
        assert isinstance(ei.value, HTTPError)
        assert "secret-body-123" not in ei.value.response_body

    def test_policy_response_non_dict_and_undecodable(self, requests_mock):
        requests_mock.post(P1_URL, json=[1])
        with pytest.raises(ResponseDecodeError, match="list"):
            _client()._post_policy_evaluate(P1, {"prompt": "x"}, "app")
        requests_mock.post(P1_URL, text="secret-policy-body")
        with pytest.raises(ValueError) as ei:
            _client()._post_policy_evaluate(P1, {"prompt": "x"}, "app")
        assert "secret-policy-body" not in str(ei.value)
