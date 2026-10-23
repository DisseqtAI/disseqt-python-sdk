"""Tests for realtime-policy base-URL routing + response parsing."""

import pytest

from disseqt_sdk import Client, is_async, is_blocking, parse_policy
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.policy import (
    DECISION_BLOCK,
    DECISION_PASS,
    ENFORCEMENT_ASYNC,
    ENFORCEMENT_SYNC,
)

# Test URLs — kept short here so the assertions stay readable.
TEST_VALIDATORS_URL = "https://test-api.disseqt.ai/realtime-validations"
TEST_POLICIES_URL = "https://test-api.disseqt.ai/realtime-policies"


@pytest.fixture
def client():
    return Client(
        project_id="test_project_123",
        api_key="test_key_xyz",
        base_url=TEST_VALIDATORS_URL,
        realtime_policy_base_url=TEST_POLICIES_URL,
        application_name="test-app",
    )


class TestRealtimePolicyBaseURL:
    """realtime_policy_base_url is the URL used for policy evaluation
    (validate(..., policies=[...])) and is separate from base_url (which
    is the validators endpoint)."""

    def test_default_points_at_managed_gateway(self):
        c = Client(project_id="p", api_key="k")
        # Default points at the realtime-validations gateway: the
        # evaluate endpoint is served by production-monitoring alongside
        # the validators (the /realtime-policies gateway is the policy
        # CRUD dashboard and exposes no SDK routes).
        assert c.realtime_policy_base_url == "https://api.disseqt.ai/realtime-validations"

    def test_policies_use_realtime_policy_base_url_not_base_url(self, requests_mock):
        c = Client(
            project_id="p",
            api_key="k",
            base_url="https://validators.example.com",
            realtime_policy_base_url="http://localhost:9010",
            application_name="t",
        )
        # Mock only the policy URL — the validators URL should never be hit.
        requests_mock.post(
            "http://localhost:9010/api/v1/sdk/policies/p1/evaluate",
            json={"success": True, "decision": "PASS"},
        )

        c.validate(InputValidationRequest(prompt="hi"), policies=["p1"])

        assert requests_mock.called
        assert requests_mock.last_request.url.startswith("http://localhost:9010/")

    def test_trailing_slash_on_url_is_stripped(self, requests_mock):
        c = Client(
            project_id="p",
            api_key="k",
            realtime_policy_base_url="http://localhost:9010/",  # trailing slash
            application_name="t",
        )
        # If the slash weren't stripped we'd get a double slash before /api.
        requests_mock.post(
            "http://localhost:9010/api/v1/sdk/policies/p1/evaluate",
            json={"success": True, "decision": "PASS"},
        )

        c.validate(InputValidationRequest(prompt="hi"), policies=["p1"])

        assert requests_mock.called

    def test_sends_auth_headers_and_application_name(self, requests_mock, client):
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/x/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        client.validate(InputValidationRequest(prompt="hi"), policies=["x"])

        headers = requests_mock.last_request.headers
        assert headers["X-API-Key"] == "test_key_xyz"
        assert headers["X-Project-Id"] == "test_project_123"
        body = requests_mock.last_request.json()
        assert body["input_data"] == {"llm_input_query": "hi"}
        assert body["application_name"] == "test-app"


class TestParsePolicy:
    """parse_policy turns the server response into typed objects."""

    def test_returns_none_when_no_policy_id(self):
        # Server's response field is still `policy_id` — that's the
        # canonical wire shape, distinct from the SDK arg name.
        assert parse_policy({"success": False, "error": "auth failed"}) is None

    def test_parses_blocking_decision(self):
        response = {
            "success": True,
            "policy_id": "abc",
            "policy_name": "PII Guard",
            "policy_version": 3,
            "decision": DECISION_BLOCK,
            "enforcement": ENFORCEMENT_SYNC,
            "rulesets": [
                {
                    "ruleset_id": "rs1",
                    "ruleset_name": "Safety",
                    "required": True,
                    "rules": [
                        {
                            "validator": "toxicity",
                            "validator_type": "input-validation",
                            "status": "fail",
                            "score": 0.91,
                            "has_score": True,
                            "threshold": 0.8,
                            "polarity": "risk",
                            "is_decider": True,
                        }
                    ],
                }
            ],
        }

        d = parse_policy(response)
        assert d is not None
        assert d.policy_id == "abc"
        assert d.policy_version == 3
        assert d.decision == DECISION_BLOCK
        assert d.enforcement == ENFORCEMENT_SYNC
        assert len(d.rulesets) == 1
        rs = d.rulesets[0]
        assert rs.ruleset_id == "rs1"
        assert rs.required is True
        assert len(rs.rules) == 1
        rule = rs.rules[0]
        assert rule.validator == "toxicity"
        assert rule.validator_type == "input-validation"
        assert rule.status == "fail"
        assert rule.score == 0.91
        assert rule.threshold == 0.8
        assert rule.polarity == "risk"
        assert rule.is_decider is True

    def test_score_none_when_has_score_false(self):
        response = {
            "policy_id": "x",
            "policy_name": "X",
            "policy_version": 1,
            "decision": "PASS",
            "rulesets": [
                {
                    "ruleset_id": "rs",
                    "ruleset_name": "rs",
                    "rules": [
                        {
                            "validator": "foo",
                            "status": "skipped",
                            "has_score": False,
                            "threshold": 0.5,
                        }
                    ],
                }
            ],
        }
        d = parse_policy(response)
        assert d is not None
        assert d.rulesets[0].rules[0].score is None

    def test_parses_aggregation_fields(self):
        response = {
            "policy_id": "w",
            "policy_name": "Weighted",
            "policy_version": 2,
            "decision": DECISION_BLOCK,
            "enforcement": ENFORCEMENT_SYNC,
            "aggregation": "weighted",
            "aggregate_score": 0.74,
            "aggregate_threshold": 0.7,
            "rulesets": [],
        }
        d = parse_policy(response)
        assert d is not None
        assert d.aggregation == "weighted"
        assert d.aggregate_score == 0.74
        assert d.aggregate_threshold == 0.7

    def test_aggregation_fields_default_when_absent(self):
        # Servers that predate aggregation enforcement (or non-weighted
        # strategies, which omit the score) must parse cleanly.
        response = {
            "policy_id": "old",
            "policy_name": "Legacy",
            "policy_version": 1,
            "decision": "PASS",
        }
        d = parse_policy(response)
        assert d is not None
        assert d.aggregation == ""
        assert d.aggregate_score is None
        assert d.aggregate_threshold is None


class TestIsBlocking:
    """is_blocking() is the convenience caller-check on decision only —
    independent of sync/async mode."""

    def test_true_when_decision_is_block(self):
        assert is_blocking({"decision": DECISION_BLOCK}) is True

    def test_false_when_decision_is_pass(self):
        assert is_blocking({"decision": DECISION_PASS}) is False

    def test_false_when_no_decision(self):
        assert is_blocking({"success": True}) is False


class TestIsAsync:
    """is_async() reads `enforcement` which now mirrors the policy's
    strategy.executionMode 1:1."""

    def test_true_when_enforcement_is_async(self):
        assert is_async({"enforcement": ENFORCEMENT_ASYNC}) is True

    def test_false_when_enforcement_is_sync(self):
        assert is_async({"enforcement": ENFORCEMENT_SYNC}) is False

    def test_false_when_no_enforcement(self):
        assert is_async({"success": True}) is False

    def test_is_blocking_and_is_async_are_independent(self):
        # A sync policy can BLOCK; an async policy could in principle
        # also carry a decision (today the server returns no decision
        # for async, but the helpers shouldn't get confused if it does).
        response = {"decision": DECISION_BLOCK, "enforcement": ENFORCEMENT_SYNC}
        assert is_blocking(response) is True
        assert is_async(response) is False


class TestDSQEnvelopeUnwrapping:
    """The helpers accept either the raw payload OR the full
    {status, data, messages, code, ...} DSQ envelope that
    prod-monitoring's /policies/:id/evaluate now returns."""

    @staticmethod
    def _wrap(data: dict) -> dict:
        return {
            "status": "success",
            "data": data,
            "messages": [],
            "code": "DSQ-2000",
            "standard_code": "OK",
            "request_id": "req_abc",
            "timestamp": "2026-06-27T15:30:18Z",
        }

    def test_is_blocking_reads_through_envelope(self):
        wrapped = self._wrap({"decision": DECISION_BLOCK, "enforcement": ENFORCEMENT_SYNC})
        assert is_blocking(wrapped) is True

    def test_is_blocking_false_when_envelope_decision_is_pass(self):
        wrapped = self._wrap({"decision": DECISION_PASS, "enforcement": ENFORCEMENT_SYNC})
        assert is_blocking(wrapped) is False

    def test_is_async_reads_through_envelope(self):
        wrapped = self._wrap({"enforcement": ENFORCEMENT_ASYNC, "status": "accepted"})
        assert is_async(wrapped) is True

    def test_parse_policy_reads_through_envelope(self):
        wrapped = self._wrap(
            {
                "policy_id": "abc",
                "policy_name": "Safety Guard",
                "policy_version": 3,
                "status": "completed",
                "decision": DECISION_BLOCK,
                "enforcement": ENFORCEMENT_SYNC,
                "rulesets": [],
            }
        )
        d = parse_policy(wrapped)
        assert d is not None
        assert d.policy_id == "abc"
        assert d.decision == DECISION_BLOCK
        assert d.enforcement == ENFORCEMENT_SYNC

    def test_raw_payload_still_works_for_backward_compat(self):
        # Pre-envelope wire shape — must keep working so existing
        # callers that already have the payload don't break.
        raw = {"decision": DECISION_BLOCK, "enforcement": ENFORCEMENT_SYNC}
        assert is_blocking(raw) is True

    def test_error_envelope_is_not_unwrapped(self):
        # An envelope whose status isn't "success" (or whose data isn't
        # a dict) shouldn't be treated as the payload — falls through.
        err = {"status": "error", "error": {"external": "nope"}, "code": "DSQ-4000"}
        # No "decision" anywhere → is_blocking False, parse returns None.
        assert is_blocking(err) is False
        assert parse_policy(err) is None
