"""Tests for the intent-guard / intent-compliance input validators."""

import json

import pytest
from requests_mock import ANY

from disseqt_sdk import SDKConfigInput
from disseqt_sdk.enums import InputValidation, ValidatorDomain
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.registry import get_validator_metadata
from disseqt_sdk.validators.input import IntentComplianceValidator, IntentGuardValidator

INTENTS = ["reset_password_other", "reset_password_other_colleague"]


@pytest.fixture
def block_config():
    """Config carrying the intent block/allow list (threshold + intents)."""
    return SDKConfigInput(threshold=0.5, intents=INTENTS)


class TestIntentValidatorPaths:
    """URL path construction for the intent validators."""

    def test_intent_guard_path(self, requests_mock, client, block_config):
        """intent-guard posts to the input-validation/intent-guard endpoint."""
        validator = IntentGuardValidator(
            data=InputValidationRequest(prompt="reset password for my colleague"),
            config=block_config,
        )
        expected_url = (
            "https://test-api.disseqt.ai/api/v1/sdk/validators/input-validation/intent-guard"
        )
        requests_mock.post(expected_url, json={"threshold_validated_result": "Fail"})

        client.validate(validator)

        assert requests_mock.called
        assert requests_mock.request_history[0].url == expected_url

    def test_intent_compliance_path(self, requests_mock, client, block_config):
        """intent-compliance posts to the input-validation/intent-compliance endpoint."""
        validator = IntentComplianceValidator(
            data=InputValidationRequest(prompt="reset my own password"),
            config=block_config,
        )
        expected_url = (
            "https://test-api.disseqt.ai/api/v1/sdk/validators/input-validation/intent-compliance"
        )
        requests_mock.post(expected_url, json={"threshold_validated_result": "Pass"})

        client.validate(validator)

        assert requests_mock.called
        assert requests_mock.request_history[0].url == expected_url


class TestIntentValidatorPayload:
    """Payload + raw-response behavior."""

    def test_intents_in_config_input(self, requests_mock, client, block_config):
        """The block/allow list rides in config_input; prompt maps to llm_input_query."""
        validator = IntentGuardValidator(
            data=InputValidationRequest(prompt="reset password for my colleague"),
            config=block_config,
        )
        requests_mock.post(ANY, json={"threshold_validated_result": "Fail"})

        client.validate(validator)
        payload = json.loads(requests_mock.request_history[0].text)

        assert payload["input_data"]["llm_input_query"] == "reset password for my colleague"
        assert payload["config_input"]["threshold"] == 0.5
        assert payload["config_input"]["intents"] == INTENTS

    def test_enforcement_available_in_raw_response(self, requests_mock, client, block_config):
        """validate() returns the raw response, so callers can read `enforcement`."""
        validator = IntentGuardValidator(
            data=InputValidationRequest(prompt="reset password for my colleague"),
            config=block_config,
        )
        requests_mock.post(
            ANY,
            json={"threshold_validated_result": "Fail", "score": 1.0, "enforcement": "blocking"},
        )

        result = client.validate(validator)

        assert result["threshold_validated_result"] == "Fail"
        assert result["enforcement"] == "blocking"


class TestSDKConfigInputIntents:
    """SDKConfigInput.intents serialization."""

    def test_intents_serialized_when_set(self):
        assert SDKConfigInput(threshold=0.5, intents=INTENTS).to_dict()["intents"] == INTENTS

    def test_intents_omitted_when_none(self):
        assert "intents" not in SDKConfigInput(threshold=0.5).to_dict()

    def test_intents_omitted_when_empty(self):
        # Empty list => defer to the project's dashboard-configured list.
        assert "intents" not in SDKConfigInput(threshold=0.5, intents=[]).to_dict()


class TestIntentValidatorRegistration:
    """Domain/slug + registry registration (decorator auto-discovery)."""

    def test_intent_guard_registered(self):
        validator = IntentGuardValidator(
            data=InputValidationRequest(prompt="x"), config=SDKConfigInput(threshold=0.5)
        )
        assert validator.domain == ValidatorDomain.INPUT_VALIDATION
        assert validator.slug == InputValidation.INTENT_GUARD.value == "intent-guard"
        assert (
            get_validator_metadata(validator.domain, validator.slug)["class"]
            is IntentGuardValidator
        )

    def test_intent_compliance_registered(self):
        validator = IntentComplianceValidator(
            data=InputValidationRequest(prompt="x"), config=SDKConfigInput(threshold=0.5)
        )
        assert validator.domain == ValidatorDomain.INPUT_VALIDATION
        assert validator.slug == InputValidation.INTENT_COMPLIANCE.value == "intent-compliance"
        assert (
            get_validator_metadata(validator.domain, validator.slug)["class"]
            is IntentComplianceValidator
        )
