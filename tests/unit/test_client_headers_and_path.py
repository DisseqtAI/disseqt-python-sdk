"""Test client headers and path construction."""

import json
import uuid
from unittest.mock import patch

import pytest
from requests_mock import ANY

from disseqt_sdk.client import HTTPError
from disseqt_sdk.validators.agentic_behavior.reliability import TopicAdherenceValidator
from disseqt_sdk.validators.input.safety import ToxicityValidator
from disseqt_sdk.validators.output.accuracy import FactualConsistencyValidator


class TestClientHeaders:
    """Test client header construction."""

    def test_build_headers_includes_required_fields(self, client):
        """Test that headers include all required fields."""
        headers = client._build_headers()

        assert "X-API-Key" in headers
        assert "X-Project-Id" in headers
        assert "Content-Type" in headers
        assert "X-Request-Id" in headers

        assert headers["X-API-Key"] == "test_key_xyz"
        assert headers["X-Project-Id"] == "test_project_123"
        assert headers["Content-Type"] == "application/json"

        # Validate X-Request-Id is a valid UUID
        request_id = headers["X-Request-Id"]
        uuid.UUID(request_id)  # Will raise ValueError if invalid

    def test_build_headers_generates_unique_request_ids(self, client):
        """Test that each call generates a unique request ID."""
        headers1 = client._build_headers()
        headers2 = client._build_headers()

        assert headers1["X-Request-Id"] != headers2["X-Request-Id"]


class TestClientPaths:
    """Test client URL path construction."""

    def test_input_validation_path(self, requests_mock, client, config, input_validation_request):
        """Test input validation URL path construction."""
        validator = ToxicityValidator(data=input_validation_request, config=config)

        # Mock the API response
        expected_url = "https://test-api.disseqt.ai/api/v1/sdk/validators/input-validation/toxicity"
        requests_mock.post(expected_url, json={"data": {}, "status": {"code": "200"}})

        client.validate(validator)

        # Verify the correct URL was called
        assert requests_mock.called
        assert requests_mock.request_history[0].url == expected_url

    def test_output_validation_path(self, requests_mock, client, config, output_validation_request):
        """Test output validation URL path construction."""
        validator = FactualConsistencyValidator(data=output_validation_request, config=config)

        expected_url = "https://test-api.disseqt.ai/api/v1/sdk/validators/output-validation/factual-consistency"
        requests_mock.post(expected_url, json={"data": {}, "status": {"code": "200"}})

        client.validate(validator)

        assert requests_mock.called
        assert requests_mock.request_history[0].url == expected_url

    def test_agentic_behavior_path(self, requests_mock, client, config, agentic_behaviour_request):
        """Test agentic behavior URL path construction."""
        validator = TopicAdherenceValidator(data=agentic_behaviour_request, config=config)

        expected_url = "https://test-api.disseqt.ai/api/v1/sdk/validators/agentic-behavior/topic-adherence"
        requests_mock.post(expected_url, json={"data": {}, "status": {"code": "200"}})

        client.validate(validator)

        assert requests_mock.called
        assert requests_mock.request_history[0].url == expected_url


class TestClientPayloads:
    """Test client payload construction."""

    def test_request_payload_structure(self, requests_mock, client, config, input_validation_request):
        """Test that request payload has correct structure."""
        validator = ToxicityValidator(data=input_validation_request, config=config)

        requests_mock.post(ANY, json={"data": {}, "status": {"code": "200"}})

        client.validate(validator)

        # Check the payload structure
        request_payload = json.loads(requests_mock.request_history[0].text)

        assert "input_data" in request_payload
        assert "config_input" in request_payload

        # Check input_data mapping
        input_data = request_payload["input_data"]
        assert input_data["llm_input_query"] == "What do you think about politics?"
        assert input_data["llm_input_context"] == "This is a political discussion"
        assert input_data["llm_output"] == "I think politics is complex"

        # Check config_input
        config_input = request_payload["config_input"]
        assert config_input["threshold"] == 0.8
        assert config_input["custom_labels"] == ["Low", "Medium", "High"]
        assert config_input["label_thresholds"] == [0.3, 0.7]

    def test_headers_sent_correctly(self, requests_mock, client, config, input_validation_request):
        """Test that all required headers are sent."""
        validator = ToxicityValidator(data=input_validation_request, config=config)

        requests_mock.post(ANY, json={"data": {}, "status": {"code": "200"}})

        client.validate(validator)

        # Check headers
        request_headers = requests_mock.request_history[0].headers

        assert request_headers["X-API-Key"] == "test_key_xyz"
        assert request_headers["X-Project-Id"] == "test_project_123"
        assert request_headers["Content-Type"] == "application/json"
        assert "X-Request-Id" in request_headers

        # Validate X-Request-Id is a valid UUID
        uuid.UUID(request_headers["X-Request-Id"])


class TestClientErrorHandling:
    """Test client error handling."""

    def test_http_error_handling(self, requests_mock, client, config, input_validation_request):
        """Test HTTP error handling."""
        validator = ToxicityValidator(data=input_validation_request, config=config)

        requests_mock.post(ANY, status_code=400, text="Bad Request")

        with pytest.raises(HTTPError) as exc_info:
            client.validate(validator)

        assert exc_info.value.status_code == 400
        assert "API request failed" in str(exc_info.value)
        assert exc_info.value.response_body == "Bad Request"

    def test_json_decode_error_handling(self, requests_mock, client, config, input_validation_request):
        """Test JSON decode error handling."""
        validator = ToxicityValidator(data=input_validation_request, config=config)

        requests_mock.post(ANY, text="Invalid JSON")

        with pytest.raises(ValueError) as exc_info:
            client.validate(validator)

        assert "Failed to decode JSON response" in str(exc_info.value)

    @patch('requests.post')
    def test_network_error_handling(self, mock_post, client, config, input_validation_request):
        """Test network error handling."""
        validator = ToxicityValidator(data=input_validation_request, config=config)

        import requests
        mock_post.side_effect = requests.RequestException("Network error")

        with pytest.raises(HTTPError) as exc_info:
            client.validate(validator)

        assert exc_info.value.status_code == 0
        assert "Network error" in str(exc_info.value)

    def test_response_body_truncation(self, requests_mock, client, config, input_validation_request):
        """Test that response body is truncated in error messages."""
        validator = ToxicityValidator(data=input_validation_request, config=config)

        # Create a long error response
        long_response = "A" * 1000
        requests_mock.post(ANY, status_code=500, text=long_response)

        with pytest.raises(HTTPError) as exc_info:
            client.validate(validator)

        # Response body should be truncated to 512 characters
        assert len(exc_info.value.response_body) == 512
        assert exc_info.value.response_body == "A" * 512
