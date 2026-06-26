"""Tests for the realtime-policy evaluate_policy() endpoint + response parsing."""

import pytest

from disseqt_sdk import Client, is_async, is_blocking, parse_policy
from disseqt_sdk.client import HTTPError
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


class TestEvaluatePolicy:
    """Client.evaluate_policy hits the right URL with the right body."""

    def test_posts_to_correct_url(self, requests_mock, client):
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/b1f8/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        client.evaluate_policy(realtime_policy_id="b1f8", prompt="hi")

        assert requests_mock.called
        body = requests_mock.last_request.json()
        assert body["input_data"] == {"llm_input_query": "hi"}
        # application_name is always sent (from the fixture Client default)
        assert body["application_name"] == "test-app"

    def test_sends_auth_headers(self, requests_mock, client):
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/x/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        client.evaluate_policy(realtime_policy_id="x", prompt="hi")

        headers = requests_mock.last_request.headers
        assert headers["X-API-Key"] == "test_key_xyz"
        assert headers["X-Project-Id"] == "test_project_123"

    def test_includes_optional_fields(self, requests_mock, client):
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/x/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        client.evaluate_policy(
            realtime_policy_id="x",
            prompt="hi",
            config_input={"intents": ["greeting"]},
            application_name="checkout-bot",
            request_id="req-42",
        )

        body = requests_mock.last_request.json()
        assert body["config_input"] == {"intents": ["greeting"]}
        assert body["application_name"] == "checkout-bot"
        assert requests_mock.last_request.headers["X-Request-Id"] == "req-42"

    def test_raises_http_error_on_non_2xx(self, requests_mock, client):
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/missing/evaluate"
        requests_mock.post(url, status_code=500, text='{"error":"policy not found"}')

        with pytest.raises(HTTPError) as exc:
            client.evaluate_policy(realtime_policy_id="missing", prompt="hi")

        assert exc.value.status_code == 500


class TestEvaluatePolicyTypedArgs:
    """The typed args (prompt/context/response + agentic fields) are
    renamed to the wire shape ML services expects."""

    def test_prompt_renamed_to_llm_input_query(self, requests_mock, client):
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/p/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        client.evaluate_policy(realtime_policy_id="p", prompt="Hello")

        assert requests_mock.last_request.json()["input_data"] == {
            "llm_input_query": "Hello"
        }

    def test_all_llm_fields_renamed(self, requests_mock, client):
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/p/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        client.evaluate_policy(
            realtime_policy_id="p",
            prompt="Q",
            context="C",
            response="R",
        )

        assert requests_mock.last_request.json()["input_data"] == {
            "llm_input_query": "Q",
            "llm_input_context": "C",
            "llm_output": "R",
        }

    def test_agentic_fields_kept_one_to_one(self, requests_mock, client):
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/p/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        client.evaluate_policy(
            realtime_policy_id="p",
            conversation_history=["hi", "hello"],
            tool_calls=[{"name": "lookup", "args": {"x": 1}}],
            agent_responses=["sure"],
            reference_data={"docs": ["..."]},
        )

        body = requests_mock.last_request.json()
        assert body["input_data"] == {
            "conversation_history": ["hi", "hello"],
            "tool_calls": [{"name": "lookup", "args": {"x": 1}}],
            "agent_responses": ["sure"],
            "reference_data": {"docs": ["..."]},
        }

    def test_llm_and_agentic_can_coexist(self, requests_mock, client):
        # A policy mixing factual_consistency + tool_call_accuracy needs both
        # shapes — the SDK builds the union.
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/p/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        client.evaluate_policy(
            realtime_policy_id="p",
            prompt="Q",
            response="R",
            tool_calls=[{"name": "lookup"}],
        )

        assert requests_mock.last_request.json()["input_data"] == {
            "llm_input_query": "Q",
            "llm_output": "R",
            "tool_calls": [{"name": "lookup"}],
        }

    def test_raw_input_data_escape_hatch(self, requests_mock, client):
        # For shapes the typed args don't cover (e.g. themes_classifier).
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/p/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        client.evaluate_policy(
            realtime_policy_id="p",
            input_data={"themes": ["safety", "tone"], "text": "..."},
        )

        assert requests_mock.last_request.json()["input_data"] == {
            "themes": ["safety", "tone"],
            "text": "...",
        }

    def test_raw_input_data_overrides_typed_args(self, requests_mock, client):
        # Raw dict wins on key conflict so callers can override the rename.
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/p/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        client.evaluate_policy(
            realtime_policy_id="p",
            prompt="from-typed-arg",
            input_data={"llm_input_query": "from-raw-dict"},
        )

        assert (
            requests_mock.last_request.json()["input_data"]["llm_input_query"]
            == "from-raw-dict"
        )

    def test_raises_when_no_input_fields(self, requests_mock, client):
        with pytest.raises(ValueError, match="at least one input field"):
            client.evaluate_policy(realtime_policy_id="p")


class TestRealtimePolicyBaseURL:
    """realtime_policy_base_url is the URL used for evaluate_policy and is
    separate from base_url (which is the validators endpoint)."""

    def test_default_points_at_managed_gateway(self):
        c = Client(project_id="p", api_key="k")
        # Default points at the dedicated realtime-policies gateway —
        # separate from base_url (which is production-monitoring's
        # validators endpoint).
        assert (
            c.realtime_policy_base_url == "https://api.disseqt.ai/realtime-policies"
        )

    def test_evaluate_uses_realtime_policy_base_url_not_base_url(self, requests_mock):
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

        c.evaluate_policy(prompt="hi", realtime_policy_id="p1")

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

        c.evaluate_policy(prompt="hi", realtime_policy_id="p1")

        assert requests_mock.called


class TestClientDefaultRealtimePolicyId:
    """Client(realtime_policy_id=...) sets a default that evaluate_policy() uses."""

    def test_uses_client_default(self, requests_mock):
        c = Client(
            project_id="p",
            api_key="k",
            realtime_policy_base_url=TEST_POLICIES_URL,
            realtime_policy_id="default-policy-uuid",
            application_name="my-app",
        )
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/default-policy-uuid/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        c.evaluate_policy(prompt="hi")

        assert requests_mock.called
        assert "default-policy-uuid" in requests_mock.last_request.url

    def test_per_call_wins(self, requests_mock):
        c = Client(
            project_id="p",
            api_key="k",
            realtime_policy_base_url=TEST_POLICIES_URL,
            realtime_policy_id="default-policy-uuid",
            application_name="my-app",
        )
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/override-uuid/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        c.evaluate_policy(prompt="hi", realtime_policy_id="override-uuid")

        assert "override-uuid" in requests_mock.last_request.url
        assert "default-policy-uuid" not in requests_mock.last_request.url

    def test_raises_when_no_realtime_policy_id_anywhere(self):
        c = Client(project_id="p", api_key="k")  # no default
        # New helpful error message — must mention both falbacks and
        # point at client.validate() for the no-policy path.
        with pytest.raises(ValueError) as exc:
            c.evaluate_policy(prompt="hi")
        msg = str(exc.value)
        assert "realtime_policy_id" in msg
        assert "client.validate" in msg, (
            "the error should point users at client.validate() as the no-policy path"
        )

    def test_validate_still_works_without_policy_id(self, requests_mock):
        # Constructing a Client without a policy_id is fine; you just
        # can't call evaluate_policy(). The existing validate() path is
        # unaffected.
        from disseqt_sdk.models.base import SDKConfigInput
        from disseqt_sdk.models.input_validation import InputValidationRequest
        from disseqt_sdk.validators.input.safety import ToxicityValidator

        c = Client(
            project_id="p",
            api_key="k",
            base_url="https://test-api.disseqt.ai/realtime-validations",
        )
        requests_mock.post(
            "https://test-api.disseqt.ai/realtime-validations"
            "/api/v1/sdk/validators/input-validation/toxicity",
            json={"data": {}, "status": {"code": "200"}},
        )

        validator = ToxicityValidator(
            data=InputValidationRequest(prompt="hi"),
            config=SDKConfigInput(threshold=0.5),
        )
        # Should not raise — the no-policy path is just client.validate().
        c.validate(validator)
        assert requests_mock.called


class TestApplicationNameRequiredWithPolicyId:
    """realtime_policy_id on the Client requires application_name (same as
    service_name on the agentic SDK)."""

    def test_constructor_rejects_policy_id_without_application_name(self):
        with pytest.raises(ValueError, match="application_name is required"):
            Client(project_id="p", api_key="k", realtime_policy_id="some-uuid")

    def test_constructor_accepts_policy_id_with_application_name(self):
        Client(
            project_id="p",
            api_key="k",
            realtime_policy_id="some-uuid",
            application_name="checkout-bot",
        )

    def test_constructor_allows_policy_id_none(self):
        # The pairing rule only kicks in when realtime_policy_id is set.
        Client(project_id="p", api_key="k")

    def test_evaluate_sends_application_name_in_body(self, requests_mock):
        c = Client(
            project_id="p",
            api_key="k",
            realtime_policy_base_url=TEST_POLICIES_URL,
            realtime_policy_id="pol",
            application_name="checkout-bot",
        )
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/pol/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        c.evaluate_policy(prompt="hi")

        body = requests_mock.last_request.json()
        assert body["application_name"] == "checkout-bot"

    def test_per_call_application_name_wins(self, requests_mock):
        c = Client(
            project_id="p",
            api_key="k",
            realtime_policy_base_url=TEST_POLICIES_URL,
            realtime_policy_id="pol",
            application_name="default-app",
        )
        url = f"{TEST_POLICIES_URL}/api/v1/sdk/policies/pol/evaluate"
        requests_mock.post(url, json={"success": True, "decision": "PASS"})

        c.evaluate_policy(prompt="hi", application_name="override-app")

        body = requests_mock.last_request.json()
        assert body["application_name"] == "override-app"

    def test_evaluate_raises_when_no_application_name_anywhere(self):
        # No realtime_policy_id at construction (no pairing check fires),
        # then evaluate_policy provides one but no application_name —
        # the method must reject it.
        c = Client(project_id="p", api_key="k")
        with pytest.raises(ValueError, match="application_name is required"):
            c.evaluate_policy(prompt="hi", realtime_policy_id="pol")


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
