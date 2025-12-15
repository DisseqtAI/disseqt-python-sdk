"""Test client integration with composite score evaluator."""

import pytest
import requests
import requests_mock

from disseqt_sdk import Client
from disseqt_sdk.client import HTTPError
from disseqt_sdk.models.composite_score import CompositeScoreRequest
from disseqt_sdk.validators.composite.evaluate import CompositeScoreEvaluator


class TestClientCompositeIntegration:
    """Test client integration with composite score evaluator."""

    def test_client_validate_composite_with_custom_handlers(self):
        """Test client validate method uses custom request/response handlers for composite."""
        client = Client(
            project_id="test_project",
            api_key="test_key",
            base_url="https://api.test.com",
        )

        evaluator = CompositeScoreEvaluator(
            data=CompositeScoreRequest(
                llm_input_query="Test query",
                llm_output="Test output",
            )
        )

        expected_response = {
            "overall_confidence": {
                "score": 0.85,
                "label": "High Confidence",
                "breakdown": {},
            },
            "success": True,
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://api.test.com/api/v1/validators/composite/evaluate",
                json=expected_response,
            )

            result = client.validate(evaluator)

            # Verify custom response handler was used (response returned as-is)
            assert result == expected_response
            assert result["overall_confidence"]["score"] == 0.85

            # Verify custom request handler was used
            assert m.called
            request_body = m.last_request.json()
            assert "input_data" in request_body
            assert "options" in request_body
            # Should NOT have config_input (composite doesn't use it)
            assert "config_input" not in request_body

    def test_client_validate_composite_with_full_config(self):
        """Test client validate with fully configured composite evaluator."""
        client = Client(
            project_id="proj_123",
            api_key="key_xyz",
            base_url="https://api.test.com",
        )

        evaluator = CompositeScoreEvaluator(
            data=CompositeScoreRequest(
                llm_input_query="Query",
                llm_output="Output",
                llm_input_context="Context",
                weights_override={"top_level": {"factual_semantic_alignment": 0.5}},
            )
        )

        expected_response = {
            "overall_confidence": {"score": 0.75},
            "success": True,
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://api.test.com/api/v1/validators/composite/evaluate",
                json=expected_response,
            )

            result = client.validate(evaluator)

            assert result["overall_confidence"]["score"] == 0.75

            # Verify payload structure
            request_body = m.last_request.json()
            assert request_body["input_data"]["llm_input_query"] == "Query"
            assert request_body["input_data"]["llm_input_context"] == "Context"
            assert request_body["options"]["weights_override"] is not None

    def test_client_validate_composite_http_error(self):
        """Test client handles HTTP errors for composite evaluator."""
        client = Client(
            project_id="test_project",
            api_key="test_key",
            base_url="https://api.test.com",
        )

        evaluator = CompositeScoreEvaluator(
            data=CompositeScoreRequest(
                llm_input_query="Test",
                llm_output="Output",
            )
        )

        with requests_mock.Mocker() as m:
            m.post(
                "https://api.test.com/api/v1/validators/composite/evaluate",
                status_code=400,
                text="Bad Request",
            )

            with pytest.raises(HTTPError) as exc_info:
                client.validate(evaluator)

            assert exc_info.value.status_code == 400
            assert "Bad Request" in exc_info.value.response_body

    def test_client_validate_composite_network_error(self):
        """Test client handles network errors for composite evaluator."""
        client = Client(
            project_id="test_project",
            api_key="test_key",
            base_url="https://api.test.com",
        )

        evaluator = CompositeScoreEvaluator(
            data=CompositeScoreRequest(
                llm_input_query="Test",
                llm_output="Output",
            )
        )

        with requests_mock.Mocker() as m:
            m.post(
                "https://api.test.com/api/v1/validators/composite/evaluate",
                exc=requests.exceptions.ConnectionError("Network error"),
            )

            with pytest.raises(HTTPError) as exc_info:
                client.validate(evaluator)

            assert "Network error" in str(exc_info.value)

    def test_client_headers_sent_for_composite(self):
        """Test that client sends correct headers for composite evaluator."""
        client = Client(
            project_id="proj_abc",
            api_key="key_xyz",
            base_url="https://api.test.com",
        )

        evaluator = CompositeScoreEvaluator(
            data=CompositeScoreRequest(
                llm_input_query="Test",
                llm_output="Output",
            )
        )

        with requests_mock.Mocker() as m:
            m.post(
                "https://api.test.com/api/v1/validators/composite/evaluate",
                json={"overall_confidence": {"score": 0.9}, "success": True},
            )

            client.validate(evaluator)

            # Verify headers
            headers = m.last_request.headers
            assert headers["X-API-Key"] == "key_xyz"
            assert headers["X-Project-Id"] == "proj_abc"
            assert headers["Content-Type"] == "application/json"
            assert "X-Request-Id" in headers

    def test_composite_url_construction(self):
        """Test URL is correctly constructed for composite evaluator."""
        client = Client(
            project_id="test_project",
            api_key="test_key",
            base_url="https://production-monitoring-eu.disseqt.ai",
        )

        evaluator = CompositeScoreEvaluator(
            data=CompositeScoreRequest(
                llm_input_query="Test",
                llm_output="Output",
            )
        )

        with requests_mock.Mocker() as m:
            m.post(
                "https://production-monitoring-eu.disseqt.ai/api/v1/validators/composite/evaluate",
                json={"overall_confidence": {"score": 0.8}, "success": True},
            )

            client.validate(evaluator)

            assert m.called
            assert (
                m.last_request.url
                == "https://production-monitoring-eu.disseqt.ai/api/v1/validators/composite/evaluate"
            )


class TestClientCustomHandlerCoverage:
    """Test client code paths for custom handlers."""

    def test_client_uses_custom_request_handler_when_available(self):
        """Test client uses custom request handler from registry."""
        client = Client(project_id="test", api_key="key")

        evaluator = CompositeScoreEvaluator(
            data=CompositeScoreRequest(
                llm_input_query="Test",
                llm_output="Output",
            )
        )

        with requests_mock.Mocker() as m:
            m.post(
                "https://production-monitoring-eu.disseqt.ai/api/v1/validators/composite/evaluate",
                json={"overall_confidence": {"score": 0.7}},
            )

            client.validate(evaluator)

            # Verify custom request handler was used
            # (no config_input in payload)
            request_body = m.last_request.json()
            assert "config_input" not in request_body
            assert "input_data" in request_body
            assert "options" in request_body

    def test_client_uses_custom_response_handler_when_available(self):
        """Test client uses custom response handler from registry."""
        client = Client(project_id="test", api_key="key")

        evaluator = CompositeScoreEvaluator(
            data=CompositeScoreRequest(
                llm_input_query="Test",
                llm_output="Output",
            )
        )

        server_response = {
            "overall_confidence": {
                "score": 0.65,
                "custom_field": "custom_value",
            },
            "success": True,
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://production-monitoring-eu.disseqt.ai/api/v1/validators/composite/evaluate",
                json=server_response,
            )

            result = client.validate(evaluator)

            # Verify response is returned as-is (custom handler preserves structure)
            assert result == server_response
            assert result["overall_confidence"]["custom_field"] == "custom_value"

