"""Per-request LLM-judge flag: ``SDKConfigInput.llm_as_a_judge`` /
``SDKConfigInput.judge`` serialization, and the policy path forwarding
``config_input`` to policy evaluation."""

from __future__ import annotations

import json

import pytest
from requests_mock import ANY

from disseqt_sdk import Client
from disseqt_sdk.models.base import SDKConfigInput
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.validators.input.safety import ToxicityValidator

BASE = "https://judge-flag.test"
P1 = "11111111-1111-4111-8111-111111111111"
TOX_URL = f"{BASE}/api/v1/sdk/validators/input-validation/toxicity"
P1_URL = f"{BASE}/api/v1/sdk/policies/{P1}/evaluate"

VALIDATOR_RESPONSE = {"success": True, "result": {"data": {"metric_name": "toxicity_evaluation"}}}
P1_PASS = {
    "status": "success",
    "code": "DSQ-2000",
    "data": {"policy_id": P1, "decision": "PASS", "enforcement": "sync"},
}


def make_client() -> Client:
    return Client(
        project_id="proj",
        api_key="key",
        base_url=BASE,
        realtime_policy_base_url=BASE,
        application_name="judge-flag-test",
    )


def toxicity(config: SDKConfigInput) -> ToxicityValidator:
    return ToxicityValidator(data=InputValidationRequest(prompt="hello"), config=config)


class TestSDKConfigInputSerialization:
    """to_dict includes llm_as_a_judge / judge only when set."""

    def test_defaults_omit_flag_and_judge(self):
        out = SDKConfigInput(threshold=0.5).to_dict()
        assert out == {"threshold": 0.5}
        assert "llm_as_a_judge" not in out
        assert "judge" not in out

    def test_flag_requires_llm_id(self):
        # The judge MUST be selected explicitly — construction fails fast
        # instead of a server-side 4xx (or a silent default-integration run).
        with pytest.raises(ValueError, match="requires llm_id"):
            SDKConfigInput(threshold=0.5, llm_as_a_judge=True)

    def test_llm_id_requires_the_flag(self):
        # The mirror rule: an llm_id on a traditional ML run would be a
        # silent no-op, which is exactly the confusion this field removes.
        with pytest.raises(ValueError, match="only used with llm_as_a_judge"):
            SDKConfigInput(threshold=0.5, llm_id="cllm-1")

    def test_flag_serialized_when_true(self):
        out = SDKConfigInput(threshold=0.5, llm_as_a_judge=True, llm_id="cllm-1").to_dict()
        assert out["llm_as_a_judge"] is True

    def test_flag_false_omitted(self):
        out = SDKConfigInput(threshold=0.5, llm_as_a_judge=False).to_dict()
        assert "llm_as_a_judge" not in out

    def test_judge_dict_custom_llm_id_satisfies_the_requirement(self):
        judge = {"custom_llm_id": "cllm-1", "model": "gpt-4o", "criteria": "be strict"}
        out = SDKConfigInput(threshold=0.5, llm_as_a_judge=True, judge=judge).to_dict()
        assert out["judge"] == judge

    def test_empty_judge_dict_omitted(self):
        out = SDKConfigInput(threshold=0.5, judge={}).to_dict()
        assert "judge" not in out

    def test_flag_composes_with_custom_labels(self):
        out = SDKConfigInput(
            threshold=0.5,
            custom_labels=["OK", "Bad", "Awful", "Severe"],
            label_thresholds=[0.1, 0.5, 0.9],
            llm_as_a_judge=True,
            llm_id="cllm-1",
        ).to_dict()
        assert out == {
            "threshold": 0.5,
            "custom_labels": ["OK", "Bad", "Awful", "Severe"],
            "label_thresholds": [0.1, 0.5, 0.9],
            "llm_as_a_judge": True,
            "judge": {"custom_llm_id": "cllm-1"},
        }

    def test_llm_id_serializes_into_judge_block(self):
        out = SDKConfigInput(threshold=0.5, llm_as_a_judge=True, llm_id="cllm-1").to_dict()
        assert out["judge"] == {"custom_llm_id": "cllm-1"}

    def test_llm_id_merges_with_judge_dict(self):
        out = SDKConfigInput(
            threshold=0.5,
            llm_as_a_judge=True,
            llm_id="cllm-1",
            judge={"model": "gpt-5"},
        ).to_dict()
        assert out["judge"] == {"custom_llm_id": "cllm-1", "model": "gpt-5"}

    def test_llm_id_wins_over_dict_key_on_conflict(self):
        out = SDKConfigInput(
            threshold=0.5,
            llm_as_a_judge=True,
            llm_id="flat-wins",
            judge={"custom_llm_id": "dict-loses", "model": "gpt-5"},
        ).to_dict()
        assert out["judge"]["custom_llm_id"] == "flat-wins"
        assert out["judge"]["model"] == "gpt-5"

    def test_caller_judge_dict_is_not_mutated(self):
        judge = {"model": "gpt-5"}
        SDKConfigInput(threshold=0.5, llm_as_a_judge=True, llm_id="cllm-1", judge=judge).to_dict()
        assert judge == {"model": "gpt-5"}


class TestValidatorWirePayload:
    """The flag/judge block reach the validator endpoint verbatim."""

    def test_payload_carries_flag_and_judge_when_set(self, requests_mock):
        requests_mock.post(ANY, json=VALIDATOR_RESPONSE)
        make_client().validate(
            toxicity(
                SDKConfigInput(
                    threshold=0.5,
                    llm_as_a_judge=True,
                    llm_id="cllm-1",
                )
            )
        )
        sent = json.loads(requests_mock.request_history[0].text)
        assert sent["config_input"]["llm_as_a_judge"] is True
        assert sent["config_input"]["judge"] == {"custom_llm_id": "cllm-1"}

    def test_payload_omits_flag_and_judge_by_default(self, requests_mock):
        requests_mock.post(ANY, json=VALIDATOR_RESPONSE)
        make_client().validate(toxicity(SDKConfigInput(threshold=0.5)))
        sent = json.loads(requests_mock.request_history[0].text)
        assert "llm_as_a_judge" not in sent["config_input"]
        assert "judge" not in sent["config_input"]


class TestPolicyPathForwardsConfigInput:
    """_validate_with_policies passes the validator's config_input through
    to policy evaluation; bare models send none."""

    def test_validator_config_forwarded_to_policy_evaluate(self, requests_mock):
        requests_mock.post(TOX_URL, json=VALIDATOR_RESPONSE)
        pol = requests_mock.post(P1_URL, json=P1_PASS)
        make_client().validate(
            toxicity(SDKConfigInput(threshold=0.7, llm_as_a_judge=True, llm_id="cllm-1")),
            policies=[P1],
        )
        sent = pol.last_request.json()
        assert sent["config_input"] == {
            "threshold": 0.7,
            "llm_as_a_judge": True,
            "judge": {"custom_llm_id": "cllm-1"},
        }

    def test_plain_config_also_forwarded(self, requests_mock):
        requests_mock.post(TOX_URL, json=VALIDATOR_RESPONSE)
        pol = requests_mock.post(P1_URL, json=P1_PASS)
        make_client().validate(toxicity(SDKConfigInput(threshold=0.3)), policies=[P1])
        assert pol.last_request.json()["config_input"] == {"threshold": 0.3}

    def test_bare_model_sends_no_config_input(self, requests_mock):
        pol = requests_mock.post(P1_URL, json=P1_PASS)
        make_client().validate(InputValidationRequest(prompt="hi"), policies=[P1])
        sent = pol.last_request.json()
        assert "config_input" not in sent
        assert sent["input_data"] == {"llm_input_query": "hi"}

    def test_policies_only_shape_unaffected_by_flag_default(self, requests_mock):
        # No validator run for a bare model even though policies get input.
        pol = requests_mock.post(P1_URL, json=P1_PASS)
        result = make_client().validate(InputValidationRequest(prompt="hi"), policies=[P1])
        assert result["validation"] is None
        assert pol.called


class TestJudgeResponsePassthrough:
    """The chain has grown judge-specific response fields since this SDK was
    written: top-level ``origin_validator`` (set when the reroute renames the
    run) and ``others.scoring_path`` (which scoring formula produced the
    score). The client returns the server dict verbatim, so they must survive
    to the caller untouched — this pins that property against any future
    typed-response refactor silently dropping them."""

    REROUTED_RESPONSE = {
        "success": True,
        "validator_type": "input-validation",
        "validator_name": "llm-judge-toxicity",
        "origin_validator": "toxicity",
        "score": 0.0082,
        "threshold_validated_result": "Pass",
        "result": {
            "data": {
                "metric_name": "llm-judge-toxicity",
                "actual_value": 0.0082,
                "metric_labels": ["Not Toxic"],
                "others": {
                    "engine": "llm-judge",
                    "model": "gpt-5",
                    "rubric_version": "v17",
                    "severity": 1,
                    "scoring_path": "rating_fallback",
                    "reasoning": "Benign greeting.",
                },
            },
            "status": {"code": "200", "message": "success"},
        },
        "request_id": "req_test",
    }

    def test_rerouted_judge_fields_reach_the_caller_verbatim(self, requests_mock):
        requests_mock.post(ANY, json=self.REROUTED_RESPONSE)
        resp = make_client().validate(
            toxicity(
                SDKConfigInput(
                    threshold=0.5,
                    llm_as_a_judge=True,
                    llm_id="cllm-1",
                )
            )
        )
        # The run was renamed by the reroute — provenance must survive.
        assert resp["origin_validator"] == "toxicity"
        assert resp["validator_name"] == "llm-judge-toxicity"
        # The receipts a caller uses to verify WHICH LLM judged and HOW.
        others = resp["result"]["data"]["others"]
        assert others["model"] == "gpt-5"
        assert others["scoring_path"] == "rating_fallback"
        assert others["rubric_version"] == "v17"
        # Whole-dict fidelity: nothing stripped, nothing renamed.
        assert resp == self.REROUTED_RESPONSE
