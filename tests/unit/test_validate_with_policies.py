"""validate(request, policies=[...]) — validator optional, policies optional,
stable {"validation", "policies"} envelope when policies are passed."""

from __future__ import annotations

import pytest

from disseqt_sdk import Client, any_blocking, is_blocking, parse_policy
from disseqt_sdk.client import HTTPError
from disseqt_sdk.models.agentic_behaviour import AgenticBehaviourRequest
from disseqt_sdk.models.base import SDKConfigInput
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.models.themes_classifier import ThemesClassifierRequest
from disseqt_sdk.validators.base import ThemesClassifierValidator
from disseqt_sdk.validators.input.safety import ToxicityValidator

BASE = "https://policies.test"
P1 = "11111111-1111-4111-8111-111111111111"
P2 = "22222222-2222-4222-8222-222222222222"

TOX_URL = f"{BASE}/api/v1/sdk/validators/input-validation/toxicity"
P1_URL = f"{BASE}/api/v1/sdk/policies/{P1}/evaluate"
P2_URL = f"{BASE}/api/v1/sdk/policies/{P2}/evaluate"

VALIDATOR_RESPONSE = {"success": True, "result": {"data": {"metric_name": "toxicity_evaluation"}}}
P1_BLOCK = {
    "status": "success",
    "code": "DSQ-2000",
    "data": {"policy_id": P1, "decision": "BLOCK", "enforcement": "sync"},
}
P2_PASS = {
    "status": "success",
    "code": "DSQ-2000",
    "data": {"policy_id": P2, "decision": "PASS", "enforcement": "sync"},
}

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def client(app_name: str | None = "policies-test") -> Client:
    return Client(
        project_id="proj",
        api_key="key",
        base_url=BASE,
        realtime_policy_base_url=BASE,
        application_name=app_name,
    )


def toxicity() -> ToxicityValidator:
    return ToxicityValidator(
        data=InputValidationRequest(prompt="hello"),
        config=SDKConfigInput(threshold=0.5),
    )


class TestShape1ClassicUnchanged:
    def test_validator_only_returns_plain_response(self, requests_mock):
        post = requests_mock.post(TOX_URL, json=VALIDATOR_RESPONSE)
        result = client().validate(toxicity())
        assert post.called
        assert result == VALIDATOR_RESPONSE  # no envelope, no extra keys
        assert "policies" not in result

    def test_bare_request_without_policies_raises(self):
        with pytest.raises(ValueError, match="policies"):
            client().validate(InputValidationRequest(prompt="hello"))


class TestShape2ValidatorPlusPolicies:
    def test_runs_validator_and_each_policy_in_order(self, requests_mock):
        tox = requests_mock.post(TOX_URL, json=VALIDATOR_RESPONSE)
        p1 = requests_mock.post(P1_URL, json=P1_BLOCK)
        p2 = requests_mock.post(P2_URL, json=P2_PASS)

        result = client().validate(toxicity(), policies=[P1, P2])

        assert tox.called and p1.called and p2.called
        assert result["validation"] == VALIDATOR_RESPONSE
        assert [parse_policy(p).policy_id for p in result["policies"]] == [P1, P2]
        # Policy calls carry the SAME input as the validator's data, in
        # wire shape, plus the client's application_name.
        sent = p1.last_request.json()
        assert sent["input_data"] == {"llm_input_query": "hello"}
        assert sent["application_name"] == "policies-test"

    def test_any_blocking_over_the_envelope(self, requests_mock):
        requests_mock.post(TOX_URL, json=VALIDATOR_RESPONSE)
        requests_mock.post(P1_URL, json=P1_BLOCK)
        requests_mock.post(P2_URL, json=P2_PASS)
        result = client().validate(toxicity(), policies=[P1, P2])
        assert any_blocking(result) is True
        assert is_blocking(result["policies"][0]) is True
        assert is_blocking(result["policies"][1]) is False

    def test_all_pass_not_blocking(self, requests_mock):
        requests_mock.post(TOX_URL, json=VALIDATOR_RESPONSE)
        requests_mock.post(P2_URL, json=P2_PASS)
        result = client().validate(toxicity(), policies=[P2])
        assert any_blocking(result) is False


class TestShape3PoliciesOnly:
    def test_bare_request_evaluates_policies_no_validator_call(self, requests_mock):
        p1 = requests_mock.post(P1_URL, json=P1_BLOCK)
        result = client().validate(
            InputValidationRequest(prompt="hi", response="out"), policies=[P1]
        )
        assert result["validation"] is None
        assert parse_policy(result["policies"][0]).decision == "BLOCK"
        # Only the policy endpoint was hit — no validator POST.
        assert all("policies" in h.url for h in requests_mock.request_history)
        assert p1.last_request.json()["input_data"] == {
            "llm_input_query": "hi",
            "llm_output": "out",
        }

    def test_agentic_request_carries_agentic_fields(self, requests_mock):
        p1 = requests_mock.post(P1_URL, json=P1_BLOCK)
        client().validate(
            AgenticBehaviourRequest(
                conversation_history=["a", "b"],
                tool_calls=[{"name": "t"}],
            ),
            policies=[P1],
        )
        sent = p1.last_request.json()["input_data"]
        assert sent["conversation_history"] == ["a", "b"]
        assert sent["tool_calls"] == [{"name": "t"}]

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="no input fields"):
            client().validate(InputValidationRequest(), policies=[P1])


class TestValidationRules:
    def test_empty_policies_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            client().validate(toxicity(), policies=[])

    def test_non_string_policy_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            client().validate(toxicity(), policies=[P1, 42])  # type: ignore[list-item]

    def test_blank_policy_id_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            client().validate(toxicity(), policies=["  "])

    def test_missing_application_name_raises(self):
        with pytest.raises(ValueError, match="application_name"):
            client(app_name=None).validate(toxicity(), policies=[P1])

    def test_themes_with_policies_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            client().validate(
                ThemesClassifierValidator(data=ThemesClassifierRequest(text="x")),
                policies=[P1],
            )


class TestErrorPropagation:
    def test_unknown_policy_404_propagates(self, requests_mock):
        tox = requests_mock.post(TOX_URL, json=VALIDATOR_RESPONSE)
        requests_mock.post(P1_URL, status_code=404, json={"code": "DSQ-4040"})
        with pytest.raises(HTTPError) as exc_info:
            client().validate(toxicity(), policies=[P1])
        assert exc_info.value.status_code == 404
        assert tox.called  # validator ran before the policy error surfaced

    def test_second_policy_failure_after_first_succeeds(self, requests_mock):
        requests_mock.post(TOX_URL, json=VALIDATOR_RESPONSE)
        requests_mock.post(P1_URL, json=P1_BLOCK)
        requests_mock.post(P2_URL, status_code=500, text="boom")
        with pytest.raises(HTTPError) as exc_info:
            client().validate(toxicity(), policies=[P1, P2])
        assert exc_info.value.status_code == 500


class TestAnyBlockingHelper:
    def test_accepts_all_supported_shapes(self):
        env_block = {"status": "success", "data": {"policy_id": "x", "decision": "BLOCK"}}
        env_pass = {"status": "success", "data": {"policy_id": "x", "decision": "PASS"}}
        assert any_blocking({"validation": None, "policies": [env_pass, env_block]})
        assert not any_blocking({"validation": None, "policies": [env_pass]})
        assert any_blocking([env_block])
        assert any_blocking(env_block)
        assert not any_blocking(env_pass)
        assert not any_blocking(VALIDATOR_RESPONSE)  # classic result -> False
        assert not any_blocking(None)
        assert not any_blocking({"policies": "not-a-list"})


class TestReviewFindings:
    """Regression tests for the adversarially-verified review findings."""

    def test_empty_input_raises_before_any_network_call(self, requests_mock):
        """Shape 2 with empty input must ValueError BEFORE the validator
        POST — no billing for a doomed call."""
        requests_mock.post(TOX_URL, json=VALIDATOR_RESPONSE)
        with pytest.raises(ValueError, match="no input fields"):
            client().validate(
                ToxicityValidator(
                    data=InputValidationRequest(),  # serializes to {}
                    config=SDKConfigInput(threshold=0.5),
                ),
                policies=[P1],
            )
        assert len(requests_mock.request_history) == 0  # nothing hit the wire

    def test_bare_composite_request_rejected(self):
        from disseqt_sdk.models.composite_score import CompositeScoreRequest

        with pytest.raises(ValueError, match="not supported"):
            client().validate(
                CompositeScoreRequest(llm_input_query="q", llm_output="o"),
                policies=[P1],
            )

    def test_bare_themes_request_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            client().validate(ThemesClassifierRequest(text="x"), policies=[P1])

    def test_generator_policies_evaluates_all(self, requests_mock):
        """A one-shot iterable must be normalized, not silently exhausted."""
        p1 = requests_mock.post(P1_URL, json=P1_BLOCK)
        p2 = requests_mock.post(P2_URL, json=P2_PASS)
        result = client().validate(
            InputValidationRequest(prompt="hi"),
            policies=(p for p in [P1, P2]),  # type: ignore[arg-type]
        )
        assert p1.called and p2.called
        assert len(result["policies"]) == 2

    def test_non_iterable_policies_raises_valueerror(self):
        with pytest.raises(ValueError, match="policy-id strings"):
            client().validate(InputValidationRequest(prompt="hi"), policies=42)  # type: ignore[arg-type]

    def test_unsupported_request_object_raises_valueerror(self):
        """A plain dict must raise ValueError, not AttributeError."""
        with pytest.raises(ValueError, match="request must be a validator"):
            client().validate({"llm_input_query": "hi"}, policies=[P1])  # type: ignore[arg-type]
