"""Unit tests for the Prompt Packs API client."""

import json
from unittest.mock import patch

import pytest
import requests as req_lib

from disseqt_sdk.api_client import _PROMPT_PACKS_BASE, DisseqtAPIClient
from disseqt_sdk.client import HTTPError
from disseqt_sdk.models.prompt_packs import (
    CreateOutputValidationRequest,
    CreateRunRequest,
    GeneratePromptPackRequest,
    MetricEvaluation,
    OutputValidationMetric,
    PaginationParams,
    PromptPackCategory,
    PromptPackOutputValidationCategory,
    PromptPackOutputValidationRequest,
)

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

BASE_URL = "http://localhost:8000"
PREFIX = f"{BASE_URL}{_PROMPT_PACKS_BASE}"


@pytest.fixture
def api_client():
    """Create a test DisseqtAPIClient."""
    return DisseqtAPIClient(
        project_id="test_project_123",
        api_key="test_key_xyz",
        base_url=BASE_URL,
        timeout=10,
    )


@pytest.fixture
def generate_request():
    """Create a test generate prompt pack request."""
    return GeneratePromptPackRequest(
        pack_name="Security Pack",
        pack_short_desc="AI-generated prompts for security testing",
        author="AI Generator",
        domain="Security",
        generation_type="AI",
        categories=[
            PromptPackCategory(
                main_category="reliability_and_safety",
                subcategory="hate_speech",
                prompts_count=5,
            ),
            PromptPackCategory(
                main_category="privacy_and_security",
                subcategory="pii_handling",
                prompts_count=3,
            ),
        ],
    )


@pytest.fixture
def create_run_request():
    """Create a test create run request."""
    return CreateRunRequest(
        run_name="Test Run",
        run_type="evaluation",
        api_key="llm-api-key",
        model_name="gpt-4",
        provider="openai",
    )


@pytest.fixture
def create_validation_request():
    """Create a test output validation request (PromptPackOutputValidationRequest)."""
    return PromptPackOutputValidationRequest(
        prompt_pack_output_validation_run_name="SDK Test Validation",
        metric_evaluations=[
            MetricEvaluation(
                metric_name=OutputValidationMetric.HATE_SPEECH.value,
                category=PromptPackOutputValidationCategory.OUTPUT_VALIDATION.value,
            ),
            MetricEvaluation(
                metric_name=OutputValidationMetric.TOXICITY.value,
                category=PromptPackOutputValidationCategory.OUTPUT_VALIDATION.value,
            ),
        ],
    )


# ------------------------------------------------------------------
# Model unit tests
# ------------------------------------------------------------------


class TestPromptPackCategory:
    """Test PromptPackCategory model."""

    def test_to_dict(self):
        """Test category serialization."""
        category = PromptPackCategory(
            main_category="reliability_and_safety",
            subcategory="hate_speech",
            prompts_count=5,
        )
        d = category.to_dict()
        assert d == {
            "main_category": "reliability_and_safety",
            "subcategory": "hate_speech",
            "prompts_count": 5,
        }


class TestGeneratePromptPackRequest:
    """Test GeneratePromptPackRequest model."""

    def test_to_payload(self, generate_request):
        """Test generate request serialization (organization_id comes from Kong, not body)."""
        payload = generate_request.to_payload()

        assert payload["pack_name"] == "Security Pack"
        assert payload["pack_short_desc"] == "AI-generated prompts for security testing"
        assert payload["author"] == "AI Generator"
        assert payload["domain"] == "Security"
        assert payload["generation_type"] == "AI"
        assert "organization_id" not in payload
        assert len(payload["categories"]) == 2
        assert payload["categories"][0]["main_category"] == "reliability_and_safety"
        assert payload["categories"][1]["prompts_count"] == 3


class TestCreateRunRequest:
    """Test CreateRunRequest model."""

    def test_to_payload(self, create_run_request):
        """Test run request serialization (project_id/organization_id come from Kong, not body)."""
        payload = create_run_request.to_payload()

        assert payload["run_name"] == "Test Run"
        assert payload["run_type"] == "evaluation"
        assert payload["api_key"] == "llm-api-key"
        assert payload["model_name"] == "gpt-4"
        assert payload["provider"] == "openai"
        assert "project_id" not in payload
        assert "organization_id" not in payload


class TestMetricEvaluation:
    """Test MetricEvaluation model."""

    def test_to_dict(self):
        """Test metric evaluation serialization."""
        m = MetricEvaluation(metric_name="toxicity", category="output-validation")
        assert m.to_dict() == {"metric_name": "toxicity", "category": "output-validation"}


class TestPromptPackOutputValidationRequest:
    """Test PromptPackOutputValidationRequest model."""

    def test_to_payload(self, create_validation_request):
        """Test validation request serialization (new API shape)."""
        payload = create_validation_request.to_payload()

        assert payload["prompt_pack_output_validation_run_name"] == "SDK Test Validation"
        assert len(payload["metric_evaluations"]) == 2
        assert payload["metric_evaluations"][0]["metric_name"] == "hate-speech"
        assert payload["metric_evaluations"][0]["category"] == "output-validation"
        assert payload["metric_evaluations"][1]["metric_name"] == "toxicity"
        assert payload["metric_evaluations"][1]["category"] == "output-validation"

    def test_to_payload_with_enum_values(self):
        """Test using OutputValidationMetric and PromptPackOutputValidationCategory enums."""
        req = PromptPackOutputValidationRequest(
            prompt_pack_output_validation_run_name="Run",
            metric_evaluations=[
                MetricEvaluation(
                    metric_name=OutputValidationMetric.BIAS.value,
                    category=PromptPackOutputValidationCategory.OUTPUT_VALIDATION.value,
                ),
            ],
        )
        payload = req.to_payload()
        assert payload["metric_evaluations"][0]["metric_name"] == "bias"
        assert payload["metric_evaluations"][0]["category"] == "output-validation"


class TestCreateOutputValidationRequestDeprecated:
    """Test deprecated CreateOutputValidationRequest still produces new payload shape."""

    def test_to_payload_maps_to_new_shape(self):
        """Deprecated request maps to prompt_pack_output_validation_run_name + metric_evaluations."""
        req = CreateOutputValidationRequest(
            validation_type="automated", metrics=["toxicity", "bias"]
        )
        payload = req.to_payload()
        assert "prompt_pack_output_validation_run_name" in payload
        assert payload["prompt_pack_output_validation_run_name"] == "automated"
        assert "metric_evaluations" in payload
        assert len(payload["metric_evaluations"]) == 2
        assert payload["metric_evaluations"][0]["metric_name"] == "toxicity"
        assert payload["metric_evaluations"][0]["category"] == "output-validation"


class TestPaginationParams:
    """Test PaginationParams model."""

    def test_defaults(self):
        """Test default pagination values."""
        p = PaginationParams()
        params = p.to_query_params()
        assert params == {"limit": "10", "offset": "0"}

    def test_custom_values(self):
        """Test custom pagination values."""
        p = PaginationParams(limit=25, offset=50)
        params = p.to_query_params()
        assert params == {"limit": "25", "offset": "50"}


# ------------------------------------------------------------------
# DisseqtAPIClient construction tests
# ------------------------------------------------------------------


class TestDisseqtAPIClientInit:
    """Test DisseqtAPIClient initialization."""

    def test_default_values(self):
        """Test default base URL and timeout."""
        client = DisseqtAPIClient(project_id="p", api_key="k")
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30

    def test_custom_values(self, api_client):
        """Test custom initialization values."""
        assert api_client.project_id == "test_project_123"
        assert api_client.api_key == "test_key_xyz"
        assert api_client.base_url == BASE_URL
        assert api_client.timeout == 10

    def test_trailing_slash_stripped(self):
        """Test that trailing slash is stripped from base_url."""
        client = DisseqtAPIClient(project_id="p", api_key="k", base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"


class TestDisseqtAPIClientHeaders:
    """Test header construction."""

    def test_build_headers(self, api_client):
        """Test headers include required fields."""
        headers = api_client._build_headers()
        assert headers["X-API-Key"] == "test_key_xyz"
        assert headers["X-Project-Id"] == "test_project_123"
        assert headers["Content-Type"] == "application/json"
        assert len(headers) == 3

    def test_no_request_id_header(self, api_client):
        """Test that X-Request-Id is NOT included (handled by Kong)."""
        headers = api_client._build_headers()
        assert "X-Request-Id" not in headers


class TestDisseqtAPIClientURLs:
    """Test URL construction."""

    def test_url_construction(self, api_client):
        """Test internal URL builder."""
        url = api_client._url("/generate")
        assert url == f"{PREFIX}/generate"

    def test_url_with_path_params(self, api_client):
        """Test URL with path parameters."""
        url = api_client._url("/pack-123/runs")
        assert url == f"{PREFIX}/pack-123/runs"


# ------------------------------------------------------------------
# Generation endpoint tests
# ------------------------------------------------------------------


class TestGeneratePromptPack:
    """Test generate_prompt_pack endpoint."""

    def test_generate_prompt_pack_success(self, requests_mock, api_client, generate_request):
        """Test successful prompt pack generation."""
        mock_response = {"id": "pack-abc-123", "pack_name": "Security Pack"}
        requests_mock.post(f"{PREFIX}/generate", json=mock_response)

        result = api_client.generate_prompt_pack(generate_request)

        assert result == mock_response
        assert requests_mock.called
        sent = json.loads(requests_mock.request_history[0].text)
        assert sent["pack_name"] == "Security Pack"
        assert len(sent["categories"]) == 2

    def test_generate_prompt_pack_sends_correct_headers(
        self, requests_mock, api_client, generate_request
    ):
        """Test that correct headers are sent."""
        requests_mock.post(f"{PREFIX}/generate", json={"id": "pack-1"})

        api_client.generate_prompt_pack(generate_request)

        h = requests_mock.request_history[0].headers
        assert h["X-API-Key"] == "test_key_xyz"
        assert h["X-Project-Id"] == "test_project_123"
        assert h["Content-Type"] == "application/json"


class TestDownloadPackCsv:
    """Test download_pack_csv (export prompts by pack ID)."""

    PACK_ID = "pack-export-123"

    def test_download_pack_csv_returns_csv_string(self, requests_mock, api_client):
        """When server returns text/csv, returns CSV string."""
        csv_body = "prompt_text,category,severity\nTest prompt,safety,High"
        requests_mock.get(
            f"{PREFIX}/{self.PACK_ID}/download",
            text=csv_body,
            headers={"Content-Type": "text/csv"},
        )

        result = api_client.download_pack_csv(self.PACK_ID)

        assert result == csv_body
        assert requests_mock.called
        req = requests_mock.request_history[0]
        assert req.path_url.endswith(f"/{self.PACK_ID}/download")

    def test_download_pack_csv_returns_json_when_app_json(self, requests_mock, api_client):
        """When server returns application/json (e.g. download URL), returns dict."""
        mock_response = {"download_url": "https://example.com/pack.csv"}
        requests_mock.get(
            f"{PREFIX}/{self.PACK_ID}/download",
            json=mock_response,
        )

        result = api_client.download_pack_csv(self.PACK_ID)

        assert result == mock_response

    def test_download_pack_csv_raises_on_non_2xx(self, requests_mock, api_client):
        """Raises HTTPError when API returns non-2xx."""
        requests_mock.get(
            f"{PREFIX}/{self.PACK_ID}/download",
            status_code=404,
            text="Pack not found",
        )

        with pytest.raises(HTTPError) as exc_info:
            api_client.download_pack_csv(self.PACK_ID)

        assert exc_info.value.status_code == 404
        assert "404" in str(exc_info.value) or "Pack" in exc_info.value.response_body


# ------------------------------------------------------------------
# Run endpoint tests
# ------------------------------------------------------------------


class TestRunEndpoints:
    """Test run CRUD endpoints."""

    PACK_ID = "pack-abc-123"
    RUN_ID = "run-xyz-456"

    def test_create_run(self, requests_mock, api_client, create_run_request):
        """Test creating a run."""
        mock_response = {"id": self.RUN_ID, "run_name": "Test Run"}
        requests_mock.post(f"{PREFIX}/{self.PACK_ID}/runs", json=mock_response)

        result = api_client.create_run(self.PACK_ID, create_run_request)

        assert result == mock_response
        sent = json.loads(requests_mock.request_history[0].text)
        assert sent["run_name"] == "Test Run"
        assert sent["model_name"] == "gpt-4"
        assert sent["provider"] == "openai"

    def test_list_runs_default_pagination(self, requests_mock, api_client):
        """Test listing runs with default pagination."""
        mock_response = {"data": [], "total": 0}
        requests_mock.get(f"{PREFIX}/{self.PACK_ID}/runs", json=mock_response)

        result = api_client.list_runs(self.PACK_ID)

        assert result == mock_response
        qs = requests_mock.request_history[0].qs
        assert qs["limit"] == ["10"]
        assert qs["offset"] == ["0"]

    def test_list_runs_custom_pagination(self, requests_mock, api_client):
        """Test listing runs with custom pagination."""
        mock_response = {"data": [], "total": 0}
        requests_mock.get(f"{PREFIX}/{self.PACK_ID}/runs", json=mock_response)

        api_client.list_runs(self.PACK_ID, pagination=PaginationParams(limit=25, offset=50))

        qs = requests_mock.request_history[0].qs
        assert qs["limit"] == ["25"]
        assert qs["offset"] == ["50"]

    def test_get_run(self, requests_mock, api_client):
        """Test getting a single run."""
        mock_response = {"id": self.RUN_ID, "status": "completed"}
        requests_mock.get(f"{PREFIX}/runs/{self.RUN_ID}", json=mock_response)

        result = api_client.get_run(self.RUN_ID)

        assert result == mock_response
        qs = requests_mock.request_history[0].qs
        assert qs["include_outputs"] == ["true"]

    def test_get_run_without_outputs(self, requests_mock, api_client):
        """Test getting a run without outputs."""
        mock_response = {"id": self.RUN_ID}
        requests_mock.get(f"{PREFIX}/runs/{self.RUN_ID}", json=mock_response)

        api_client.get_run(self.RUN_ID, include_outputs=False)

        qs = requests_mock.request_history[0].qs
        assert qs["include_outputs"] == ["false"]

    def test_delete_run(self, requests_mock, api_client):
        """Test deleting a run."""
        requests_mock.delete(f"{PREFIX}/runs/{self.RUN_ID}", status_code=204)

        result = api_client.delete_run(self.RUN_ID)

        assert result == {"status": "deleted"}

    def test_get_run_outputs(self, requests_mock, api_client):
        """Test getting run outputs."""
        mock_response = {"data": [{"prompt": "test", "output": "response"}]}
        requests_mock.get(f"{PREFIX}/runs/{self.RUN_ID}/outputs", json=mock_response)

        result = api_client.get_run_outputs(self.RUN_ID)

        assert result == mock_response

    def test_get_run_outputs_csv(self, requests_mock, api_client):
        """Test getting run outputs as CSV."""
        mock_response = {"csv_url": "https://example.com/outputs.csv"}
        requests_mock.get(f"{PREFIX}/runs/{self.RUN_ID}/outputs/csv", json=mock_response)

        result = api_client.get_run_outputs_csv(self.RUN_ID)

        assert result == mock_response


# ------------------------------------------------------------------
# Output validation endpoint tests
# ------------------------------------------------------------------


class TestOutputValidationEndpoints:
    """Test output validation CRUD endpoints."""

    RUN_ID = "run-xyz-456"
    PACK_ID = "pack-abc-123"
    VALIDATION_ID = "val-def-789"

    def test_create_output_validation(self, requests_mock, api_client, create_validation_request):
        """Test creating an output validation."""
        mock_response = {"id": self.VALIDATION_ID, "status": "pending"}
        requests_mock.post(
            f"{PREFIX}/runs/{self.RUN_ID}/validate-outputs",
            json=mock_response,
        )

        result = api_client.create_output_validation(self.RUN_ID, create_validation_request)

        assert result == mock_response
        sent = json.loads(requests_mock.request_history[0].text)
        assert sent["prompt_pack_output_validation_run_name"] == "SDK Test Validation"
        assert len(sent["metric_evaluations"]) == 2
        assert sent["metric_evaluations"][0]["metric_name"] == "hate-speech"
        assert sent["metric_evaluations"][0]["category"] == "output-validation"
        assert sent["metric_evaluations"][1]["metric_name"] == "toxicity"
        assert sent["metric_evaluations"][1]["category"] == "output-validation"

    def test_list_run_output_validations(self, requests_mock, api_client):
        """Test listing output validations for a run."""
        mock_response = {"data": []}
        requests_mock.get(
            f"{PREFIX}/runs/{self.RUN_ID}/output-validations",
            json=mock_response,
        )

        result = api_client.list_run_output_validations(self.RUN_ID)

        assert result == mock_response

    def test_get_output_validation(self, requests_mock, api_client):
        """Test getting a specific output validation."""
        mock_response = {"id": self.VALIDATION_ID, "status": "completed"}
        requests_mock.get(
            f"{PREFIX}/output-validations/{self.VALIDATION_ID}",
            json=mock_response,
        )

        result = api_client.get_output_validation(self.VALIDATION_ID)

        assert result == mock_response

    def test_get_output_validation_summary(self, requests_mock, api_client):
        """Test getting output validation summary."""
        mock_response = {"summary": {"toxicity": 0.2, "bias": 0.1}}
        requests_mock.get(
            f"{PREFIX}/output-validations/{self.VALIDATION_ID}/summary",
            json=mock_response,
        )

        result = api_client.get_output_validation_summary(self.VALIDATION_ID)

        assert result == mock_response

    def test_get_output_validation_results(self, requests_mock, api_client):
        """Test getting output validation results with pagination."""
        mock_response = {"data": [], "total": 0}
        requests_mock.get(
            f"{PREFIX}/output-validations/{self.VALIDATION_ID}/results",
            json=mock_response,
        )

        result = api_client.get_output_validation_results(
            self.VALIDATION_ID,
            pagination=PaginationParams(limit=5, offset=10),
        )

        assert result == mock_response
        qs = requests_mock.request_history[0].qs
        assert qs["limit"] == ["5"]
        assert qs["offset"] == ["10"]

    def test_get_output_validation_grouped_outputs(self, requests_mock, api_client):
        """Test getting grouped outputs."""
        mock_response = {"groups": []}
        requests_mock.get(
            f"{PREFIX}/output-validations/{self.VALIDATION_ID}/outputs/grouped",
            json=mock_response,
        )

        result = api_client.get_output_validation_grouped_outputs(self.VALIDATION_ID)

        assert result == mock_response

    def test_get_output_validation_results_csv(self, requests_mock, api_client):
        """Test getting validation results as CSV."""
        mock_response = {"csv_url": "https://example.com/results.csv"}
        requests_mock.get(
            f"{PREFIX}/output-validations/{self.VALIDATION_ID}/results/csv",
            json=mock_response,
        )

        result = api_client.get_output_validation_results_csv(self.VALIDATION_ID)

        assert result == mock_response

    def test_delete_output_validation(self, requests_mock, api_client):
        """Test deleting an output validation."""
        requests_mock.delete(
            f"{PREFIX}/output-validations/{self.VALIDATION_ID}",
            status_code=204,
        )

        result = api_client.delete_output_validation(self.VALIDATION_ID)

        assert result == {"status": "deleted"}

    def test_list_pack_output_validations(self, requests_mock, api_client):
        """Test listing all output validations for a pack."""
        mock_response = {"data": [], "total": 0}
        requests_mock.get(
            f"{PREFIX}/{self.PACK_ID}/output-validations",
            json=mock_response,
        )

        result = api_client.list_pack_output_validations(self.PACK_ID)

        assert result == mock_response


# ------------------------------------------------------------------
# Error handling tests
# ------------------------------------------------------------------


class TestDisseqtAPIClientErrors:
    """Test error handling in DisseqtAPIClient."""

    def test_http_error_on_400(self, requests_mock, api_client, generate_request):
        """Test HTTPError raised on 400 response."""
        requests_mock.post(f"{PREFIX}/generate", status_code=400, text="Bad Request")

        with pytest.raises(HTTPError) as exc_info:
            api_client.generate_prompt_pack(generate_request)

        assert exc_info.value.status_code == 400
        assert exc_info.value.response_body == "Bad Request"

    def test_http_error_on_401(self, requests_mock, api_client, generate_request):
        """Test HTTPError raised on 401 response."""
        requests_mock.post(
            f"{PREFIX}/generate",
            status_code=401,
            text="Unauthorized",
        )

        with pytest.raises(HTTPError) as exc_info:
            api_client.generate_prompt_pack(generate_request)

        assert exc_info.value.status_code == 401

    def test_http_error_on_500(self, requests_mock, api_client, generate_request):
        """Test HTTPError raised on 500 response."""
        requests_mock.post(
            f"{PREFIX}/generate",
            status_code=500,
            text="Internal Server Error",
        )

        with pytest.raises(HTTPError) as exc_info:
            api_client.generate_prompt_pack(generate_request)

        assert exc_info.value.status_code == 500

    def test_json_decode_error(self, requests_mock, api_client, generate_request):
        """Test ValueError raised on invalid JSON response."""
        requests_mock.post(f"{PREFIX}/generate", text="Not JSON")

        with pytest.raises(ValueError, match="Failed to decode JSON"):
            api_client.generate_prompt_pack(generate_request)

    def test_null_json_response(self, requests_mock, api_client, generate_request):
        """Test ValueError raised on null JSON response."""
        requests_mock.post(f"{PREFIX}/generate", text="null")

        with pytest.raises(ValueError, match="null/empty JSON"):
            api_client.generate_prompt_pack(generate_request)

    @patch("requests.request")
    def test_network_error(self, mock_request, api_client, generate_request):
        """Test HTTPError raised on network failure."""
        mock_request.side_effect = req_lib.RequestException("Connection refused")

        with pytest.raises(HTTPError) as exc_info:
            api_client.generate_prompt_pack(generate_request)

        assert exc_info.value.status_code == 0
        assert "Network error" in str(exc_info.value)

    def test_response_body_truncation(self, requests_mock, api_client, generate_request):
        """Test that long response bodies are truncated to 512 chars."""
        long_body = "X" * 1000
        requests_mock.post(f"{PREFIX}/generate", status_code=500, text=long_body)

        with pytest.raises(HTTPError) as exc_info:
            api_client.generate_prompt_pack(generate_request)

        assert len(exc_info.value.response_body) == 512
