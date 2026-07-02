"""Policy-governed validate(): the client-bound policy decides which
validators run, at what threshold, and which are skipped locally."""

from __future__ import annotations

import pytest

from disseqt_sdk import Client, is_policy_skipped
from disseqt_sdk.client import HTTPError
from disseqt_sdk.models.base import SDKConfigInput
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.models.mcp_security import McpSecurityRequest
from disseqt_sdk.validators.input.invisible_text import InvisibleTextValidator
from disseqt_sdk.validators.input.safety import ToxicityValidator

BASE = "https://policy-gate.test"
POLICY_ID = "11111111-1111-4111-8111-111111111111"

DETAIL_URL = f"{BASE}/api/v1/sdk/policies/{POLICY_ID}"

POLICY_DETAIL = {
    "status": "success",
    "code": "DSQ-2000",
    "data": {
        "policy_id": POLICY_ID,
        "name": "Gate Test Policy",
        "version": 3,
        "status": "published",
        "enforcement": "sync",
        "required_input_fields": ["llm_input_query"],
        "rulesets": [
            {
                "ruleset_id": "rs-1",
                "ruleset_name": "Security",
                "required": True,
                "validators": [
                    {
                        "validator": "invisible_text",
                        "validator_type": "input-validation",
                        "enabled": True,
                        "threshold": 0.42,
                        "polarity": "risk",
                    },
                    {
                        "validator": "prompt_injection",
                        "validator_type": "mcp-security",
                        "enabled": True,
                        "threshold": 0.6,
                        "polarity": "risk",
                    },
                    {
                        "validator": "toxicity",
                        "validator_type": "input-validation",
                        "enabled": False,
                        "threshold": 0.5,
                        "polarity": "risk",
                    },
                ],
            }
        ],
    },
}

VALIDATOR_RESPONSE = {"success": True, "result": {"data": {"metric_name": "x"}}}


def gated_client() -> Client:
    return Client(
        project_id="proj",
        api_key="key",
        base_url=BASE,
        realtime_policy_id=POLICY_ID,
        application_name="gate-test",
        realtime_policy_base_url=BASE,
    )


def toxicity() -> ToxicityValidator:
    return ToxicityValidator(
        data=InputValidationRequest(prompt="hello"),
        config=SDKConfigInput(threshold=0.9),
    )


def invisible_text() -> InvisibleTextValidator:
    return InvisibleTextValidator(
        data=InputValidationRequest(prompt="hello"),
        config=SDKConfigInput(threshold=0.9),
    )


class TestSkipSemantics:
    def test_validator_not_in_policy_is_skipped_locally(self, requests_mock):
        requests_mock.get(DETAIL_URL, json=POLICY_DETAIL)
        # No validator endpoint registered: a POST would error the test.
        from disseqt_sdk.validators.input.hate_speech import HateSpeechValidator

        result = gated_client().validate(
            HateSpeechValidator(
                data=InputValidationRequest(prompt="hello"),
                config=SDKConfigInput(threshold=0.5),
            )
        )
        assert result["skipped"] is True
        assert result["skipped_reason"] == "validator_not_in_policy"
        assert result["validator_name"] == "hate-speech"
        assert result["validator_type"] == "input-validation"
        assert result["policy"]["policy_id"] == POLICY_ID
        assert result["policy"]["policy_name"] == "Gate Test Policy"
        assert result["policy"]["policy_version"] == 3
        assert is_policy_skipped(result)
        # Only the policy detail was fetched — no validator POST happened.
        assert len(requests_mock.request_history) == 1
        assert requests_mock.request_history[0].method == "GET"

    def test_disabled_validator_is_skipped(self, requests_mock):
        requests_mock.get(DETAIL_URL, json=POLICY_DETAIL)
        result = gated_client().validate(toxicity())
        assert result["skipped"] is True
        assert result["skipped_reason"] == "validator_disabled_in_policy"
        assert is_policy_skipped(result)

    def test_domain_mismatch_is_not_in_policy(self, requests_mock):
        """input-validation prompt-injection must NOT match the policy's
        mcp-security prompt_injection entry."""
        requests_mock.get(DETAIL_URL, json=POLICY_DETAIL)
        from disseqt_sdk.validators.input.prompt_injection import (
            InputPromptInjectionValidator,
        )

        result = gated_client().validate(
            InputPromptInjectionValidator(
                data=InputValidationRequest(prompt="hello"),
                config=SDKConfigInput(threshold=0.5),
            )
        )
        assert result["skipped"] is True
        assert result["skipped_reason"] == "validator_not_in_policy"


class TestEnabledPath:
    def test_enabled_validator_runs_with_policy_threshold(self, requests_mock):
        requests_mock.get(DETAIL_URL, json=POLICY_DETAIL)
        post = requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/input-validation/invisible-text",
            json=VALIDATOR_RESPONSE,
        )
        result = gated_client().validate(invisible_text())
        # The POST went out, with the POLICY's threshold (0.42), not the
        # code-level 0.9.
        assert post.called
        sent = post.last_request.json()
        assert sent["config_input"]["threshold"] == 0.42
        # Response is stamped with the governing policy.
        assert result["policy"]["policy_id"] == POLICY_ID
        assert result["policy"]["threshold_source"] == "policy"
        assert not is_policy_skipped(result)

    def test_name_normalization_hyphen_vs_underscore(self, requests_mock):
        """SDK slug 'invisible-text' matches policy name 'invisible_text'."""
        requests_mock.get(DETAIL_URL, json=POLICY_DETAIL)
        post = requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/input-validation/invisible-text",
            json=VALIDATOR_RESPONSE,
        )
        gated_client().validate(invisible_text())
        assert post.called

    def test_mcp_prompt_injection_matches_mcp_entry(self, requests_mock):
        requests_mock.get(DETAIL_URL, json=POLICY_DETAIL)
        from disseqt_sdk.validators.mcp_security.security import (
            McpPromptInjectionValidator,
        )

        post = requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/mcp-security/prompt-injection",
            json=VALIDATOR_RESPONSE,
        )
        result = gated_client().validate(
            McpPromptInjectionValidator(
                data=McpSecurityRequest(prompt="hello"),
                config=SDKConfigInput(threshold=0.9),
            )
        )
        assert post.called
        assert post.last_request.json()["config_input"]["threshold"] == 0.6
        assert result["policy"]["threshold_source"] == "policy"


class TestCaching:
    def test_detail_fetched_once_within_ttl(self, requests_mock):
        get = requests_mock.get(DETAIL_URL, json=POLICY_DETAIL)
        requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/input-validation/invisible-text",
            json=VALIDATOR_RESPONSE,
        )
        client = gated_client()
        client.validate(invisible_text())
        client.validate(invisible_text())
        client.validate(toxicity())  # skipped, also uses the cache
        assert get.call_count == 1

    def test_ttl_expiry_refetches(self, requests_mock):
        get = requests_mock.get(DETAIL_URL, json=POLICY_DETAIL)
        requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/input-validation/invisible-text",
            json=VALIDATOR_RESPONSE,
        )
        client = gated_client()
        client.validate(invisible_text())
        client._policy_detail_fetched_at -= 61  # age the cache past the TTL
        client.validate(invisible_text())
        assert get.call_count == 2

    def test_stale_served_on_5xx(self, requests_mock):
        client = gated_client()
        requests_mock.get(DETAIL_URL, json=POLICY_DETAIL)
        requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/input-validation/invisible-text",
            json=VALIDATOR_RESPONSE,
        )
        client.validate(invisible_text())  # primes the cache
        requests_mock.get(DETAIL_URL, status_code=503, text="unavailable")
        client._policy_detail_fetched_at -= 61
        result = client.validate(invisible_text())  # stale copy keeps working
        assert result["policy"]["policy_id"] == POLICY_ID

    def test_404_with_no_cache_raises(self, requests_mock):
        requests_mock.get(DETAIL_URL, status_code=404, json={"code": "DSQ-4040"})
        with pytest.raises(HTTPError) as exc_info:
            gated_client().validate(invisible_text())
        assert exc_info.value.status_code == 404

    def test_network_error_with_no_cache_raises(self, requests_mock):
        import requests as requests_lib

        requests_mock.get(DETAIL_URL, exc=requests_lib.ConnectionError("boom"))
        with pytest.raises(HTTPError) as exc_info:
            gated_client().validate(invisible_text())
        assert exc_info.value.status_code == 0


class TestUngatedPaths:
    def test_no_policy_client_never_fetches_detail(self, requests_mock):
        post = requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/input-validation/toxicity",
            json=VALIDATOR_RESPONSE,
        )
        client = Client(project_id="proj", api_key="key", base_url=BASE)
        result = client.validate(toxicity())
        assert post.called
        assert "policy" not in result
        assert len(requests_mock.request_history) == 1  # no GET

    def test_no_policy_keeps_code_threshold(self, requests_mock):
        post = requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/input-validation/toxicity",
            json=VALIDATOR_RESPONSE,
        )
        Client(project_id="proj", api_key="key", base_url=BASE).validate(toxicity())
        assert post.last_request.json()["config_input"]["threshold"] == 0.9

    def test_non_base_validator_requests_pass_through(self, requests_mock):
        """Composite/themes-style requests (not BaseValidator subclasses)
        are never policy-gated."""

        class FakeComposite:
            domain = type("D", (), {"value": "composite-score"})()
            slug = "composite-score"
            _path_template = "/api/v1/sdk/validators/composite-score"

            def to_payload(self):
                return {"input_data": {}, "config_input": {}}

        post = requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/composite-score",
            json={"success": True},
        )
        result = gated_client().validate(FakeComposite())
        assert post.called
        assert "policy" not in result
        # No detail GET happened — gating skipped entirely.
        assert all(r.method == "POST" for r in requests_mock.request_history)


class TestEvaluatePolicyDeprecation:
    def test_evaluate_policy_warns_but_works(self, requests_mock):
        requests_mock.post(
            f"{BASE}/api/v1/sdk/policies/{POLICY_ID}/evaluate",
            json={"status": "success", "data": {"policy_id": POLICY_ID, "decision": "PASS"}},
        )
        with pytest.warns(DeprecationWarning, match="evaluate_policy"):
            result = gated_client().evaluate_policy(prompt="hello")
        assert result["data"]["decision"] == "PASS"


class TestIsPolicySkippedHelper:
    def test_true_only_for_policy_skip_markers(self):
        assert is_policy_skipped(
            {"skipped": True, "skipped_reason": "validator_not_in_policy", "policy": {}}
        )
        assert not is_policy_skipped({"skipped": True})  # no policy block
        assert not is_policy_skipped({"success": True, "policy": {}})
        assert not is_policy_skipped({})


class TestReviewFindings:
    """Regression tests for the adversarially-verified review findings."""

    def test_rebinding_policy_id_invalidates_cache(self, requests_mock):
        """Mutating client.realtime_policy_id must not serve the old
        policy's cached rules."""
        other_id = "22222222-2222-4222-8222-222222222222"
        other_detail = {
            "status": "success",
            "data": {
                "policy_id": other_id,
                "name": "Other Policy",
                "version": 1,
                "enforcement": "sync",
                "rulesets": [],
            },
        }
        requests_mock.get(DETAIL_URL, json=POLICY_DETAIL)
        requests_mock.get(f"{BASE}/api/v1/sdk/policies/{other_id}", json=other_detail)
        requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/input-validation/invisible-text",
            json=VALIDATOR_RESPONSE,
        )
        client = gated_client()
        client.validate(invisible_text())  # primes cache with POLICY_ID
        client.realtime_policy_id = other_id
        result = client.validate(invisible_text())  # must refetch, not reuse
        assert result["skipped"] is True  # empty rulesets -> not in policy
        assert result["policy"]["policy_id"] == other_id

    def test_enabled_entry_wins_over_disabled_duplicate(self, requests_mock):
        """Same validator disabled in an early ruleset, enabled in a later
        one -> it must RUN (mirrors the server, which evaluates every
        ruleset)."""
        detail = {
            "status": "success",
            "data": {
                "policy_id": POLICY_ID,
                "name": "Dup Policy",
                "version": 1,
                "enforcement": "sync",
                "rulesets": [
                    {
                        "ruleset_id": "rs-draft",
                        "ruleset_name": "Drafts",
                        "validators": [
                            {
                                "validator": "invisible_text",
                                "validator_type": "input-validation",
                                "enabled": False,
                                "threshold": 0.9,
                            }
                        ],
                    },
                    {
                        "ruleset_id": "rs-live",
                        "ruleset_name": "Live",
                        "validators": [
                            {
                                "validator": "invisible_text",
                                "validator_type": "input-validation",
                                "enabled": True,
                                "threshold": 0.33,
                            }
                        ],
                    },
                ],
            },
        }
        requests_mock.get(DETAIL_URL, json=detail)
        post = requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/input-validation/invisible-text",
            json=VALIDATOR_RESPONSE,
        )
        result = gated_client().validate(invisible_text())
        assert post.called
        assert post.last_request.json()["config_input"]["threshold"] == 0.33
        assert not is_policy_skipped(result)

    def test_output_suffix_vocabulary_matches(self, requests_mock):
        """A policy rule named 'hate_speech_output' (the catalog's output
        vocabulary) must match the SDK's output hate-speech validator."""
        detail = {
            "status": "success",
            "data": {
                "policy_id": POLICY_ID,
                "name": "Output Policy",
                "version": 1,
                "enforcement": "sync",
                "rulesets": [
                    {
                        "ruleset_id": "rs-1",
                        "ruleset_name": "Output safety",
                        "validators": [
                            {
                                "validator": "hate_speech_output",
                                "validator_type": "output-validation",
                                "enabled": True,
                                "threshold": 0.7,
                            }
                        ],
                    }
                ],
            },
        }
        requests_mock.get(DETAIL_URL, json=detail)
        from disseqt_sdk.models.output_validation import OutputValidationRequest
        from disseqt_sdk.validators.output.hate_speech import OutputHateSpeechValidator

        post = requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/output-validation/hate-speech",
            json=VALIDATOR_RESPONSE,
        )
        result = gated_client().validate(
            OutputHateSpeechValidator(
                data=OutputValidationRequest(response="some model output"),
                config=SDKConfigInput(threshold=0.2),
            )
        )
        assert post.called
        assert post.last_request.json()["config_input"]["threshold"] == 0.7
        assert not is_policy_skipped(result)

    def test_stale_serve_restamps_retry_clock(self, requests_mock):
        """During an outage, only one fetch attempt per TTL window — not
        one per gated call."""
        client = gated_client()
        requests_mock.get(DETAIL_URL, json=POLICY_DETAIL)
        requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/input-validation/invisible-text",
            json=VALIDATOR_RESPONSE,
        )
        client.validate(invisible_text())  # 1 GET — primes cache
        outage = requests_mock.get(DETAIL_URL, status_code=503, text="down")
        client._policy_detail_fetched_at -= 61
        client.validate(invisible_text())  # 1 failed GET, serves stale, re-stamps
        client.validate(invisible_text())  # within re-stamped TTL -> NO new GET
        client.validate(invisible_text())
        assert outage.call_count == 1

    def test_malformed_200_serves_stale(self, requests_mock):
        """A 200 with a garbage body falls back to the cached copy, same as
        network errors and 5xx."""
        client = gated_client()
        requests_mock.get(DETAIL_URL, json=POLICY_DETAIL)
        requests_mock.post(
            f"{BASE}/api/v1/sdk/validators/input-validation/invisible-text",
            json=VALIDATOR_RESPONSE,
        )
        client.validate(invisible_text())  # primes cache
        requests_mock.get(DETAIL_URL, text="<html>gateway error</html>")
        client._policy_detail_fetched_at -= 61
        result = client.validate(invisible_text())
        assert result["policy"]["policy_id"] == POLICY_ID

    def test_malformed_200_with_no_cache_raises(self, requests_mock):
        requests_mock.get(DETAIL_URL, text="<html>gateway error</html>")
        with pytest.raises(ValueError, match="decode|policy"):
            gated_client().validate(invisible_text())
