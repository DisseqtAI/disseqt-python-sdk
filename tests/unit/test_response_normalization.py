"""Test response normalization."""

import pytest

from disseqt_sdk.response import normalize_server_payload, validate_actual_value_type


class TestResponseNormalization:
    """Test response normalization functionality."""

    def test_normalize_complete_response(self, mock_server_response, expected_normalized_response):
        """Test normalization of complete server response."""
        result = normalize_server_payload(mock_server_response)
        assert result == expected_normalized_response

    def test_normalize_minimal_response(self):
        """Test normalization with minimal server response."""
        server_response = {
            "data": {
                "metric_name": "test_metric",
                "actual_value": 0.5,
            }
        }

        result = normalize_server_payload(server_response)

        expected = {
            "data": {
                "metric_name": "test_metric",
                "actual_value": 0.5,
                "others": {}
            },
            "status": {
                "code": "200",
                "message": "Success"
            }
        }

        assert result == expected

    def test_normalize_response_with_unknown_fields(self):
        """Test that unknown fields go into others bag."""
        server_response = {
            "data": {
                "metric_name": "test_metric",
                "actual_value": 0.5,
                "unknown_field_1": "value1",
                "unknown_field_2": {"nested": "value"},
                "metric_labels": ["label1", "label2"],  # Known field
                "another_unknown": [1, 2, 3],
            },
            "status": {
                "code": "201",
                "message": "Created"
            }
        }

        result = normalize_server_payload(server_response)

        # Check known fields are preserved
        assert result["data"]["metric_name"] == "test_metric"
        assert result["data"]["actual_value"] == 0.5
        assert result["data"]["metric_labels"] == ["label1", "label2"]

        # Check unknown fields are in others
        others = result["data"]["others"]
        assert others["unknown_field_1"] == "value1"
        assert others["unknown_field_2"] == {"nested": "value"}
        assert others["another_unknown"] == [1, 2, 3]

        # Check status is preserved
        assert result["status"]["code"] == "201"
        assert result["status"]["message"] == "Created"

    def test_normalize_response_missing_data_section(self):
        """Test normalization when data section is missing."""
        server_response = {
            "status": {
                "code": "200",
                "message": "Success"
            }
        }

        result = normalize_server_payload(server_response)

        expected = {
            "data": {
                "others": {}
            },
            "status": {
                "code": "200",
                "message": "Success"
            }
        }

        assert result == expected

    def test_normalize_response_missing_status_section(self):
        """Test normalization when status section is missing."""
        server_response = {
            "data": {
                "metric_name": "test_metric",
                "actual_value": 0.5,
            }
        }

        result = normalize_server_payload(server_response)

        # Status should default to success
        assert result["status"]["code"] == "200"
        assert result["status"]["message"] == "Success"

    def test_normalize_empty_response(self):
        """Test normalization of empty response."""
        server_response = {}

        result = normalize_server_payload(server_response)

        expected = {
            "data": {
                "others": {}
            },
            "status": {
                "code": "200",
                "message": "Success"
            }
        }

        assert result == expected

    def test_all_known_fields_preserved(self):
        """Test that all known fields are preserved correctly."""
        server_response = {
            "data": {
                "metric_name": "comprehensive_test",
                "actual_value": 0.75,
                "actual_value_type": "float",
                "metric_labels": ["Good", "Better", "Best"],
                "threshold": ["Pass"],
                "threshold_score": 0.8,
                "extra_field": "goes_to_others"
            }
        }

        result = normalize_server_payload(server_response)

        data = result["data"]
        assert data["metric_name"] == "comprehensive_test"
        assert data["actual_value"] == 0.75
        assert data["actual_value_type"] == "float"
        assert data["metric_labels"] == ["Good", "Better", "Best"]
        assert data["threshold"] == ["Pass"]
        assert data["threshold_score"] == 0.8
        assert data["others"]["extra_field"] == "goes_to_others"


class TestActualValueTypeValidation:
    """Test actual value type validation."""

    def test_validate_float_type(self):
        """Test float type validation."""
        assert validate_actual_value_type(3.14) == "float"
        assert validate_actual_value_type(0.0) == "float"
        assert validate_actual_value_type(-1.5) == "float"

    def test_validate_int_type(self):
        """Test int type validation."""
        assert validate_actual_value_type(42) == "int"
        assert validate_actual_value_type(0) == "int"
        assert validate_actual_value_type(-10) == "int"

    def test_validate_string_type(self):
        """Test string type validation."""
        assert validate_actual_value_type("hello") == "string"
        assert validate_actual_value_type("") == "string"
        assert validate_actual_value_type("123") == "string"

    def test_validate_bool_type(self):
        """Test bool type validation."""
        assert validate_actual_value_type(True) == "bool"
        assert validate_actual_value_type(False) == "bool"

    def test_validate_unsupported_type(self):
        """Test validation of unsupported types."""
        with pytest.raises(ValueError, match="Unsupported actual_value type"):
            validate_actual_value_type([1, 2, 3])

        with pytest.raises(ValueError, match="Unsupported actual_value type"):
            validate_actual_value_type({"key": "value"})

        with pytest.raises(ValueError, match="Unsupported actual_value type"):
            validate_actual_value_type(None)


class TestNewAPIResponseFormat:
    """Test normalization of new API response format."""

    def test_normalize_new_api_response_complete(self):
        """Test normalization of complete new API response."""
        server_response = {
            "score": 0.85,
            "label": "toxic",
            "passed": False,
            "explanation": "Content failed toxicity validation with score 0.85 (threshold: 0.50). toxic",
            "details": {
                "metric_name": "toxicity",
                "threshold_score": 0.5,
                "processing_time": 0.123,
                "confidence": 0.85,
                "top_label": "toxic"
            },
            "category": "input-validation",
            "request_id": "d43426b3-bfd0-4dd8-b875-aa92cfa5f678",
            "credits_info": {
                "credits_deducted": 1.0,
                "remaining_credits": 212.0,
                "operation": "input-validation_validation_toxicity",
                "credit_cost": 1.0
            }
        }

        result = normalize_server_payload(server_response)

        # Check main structure
        assert "data" in result
        assert "status" in result

        # Check data section
        data = result["data"]
        assert data["metric_name"] == "toxicity"
        assert data["actual_value"] == 0.85
        assert data["actual_value_type"] == "float"
        assert data["metric_labels"] == ["toxic"]
        assert data["threshold"] == ["Fail"]
        assert data["threshold_score"] == 0.5

        # Check others bag
        others = data["others"]
        assert others["explanation"] == "Content failed toxicity validation with score 0.85 (threshold: 0.50). toxic"
        assert others["category"] == "input-validation"
        assert others["request_id"] == "d43426b3-bfd0-4dd8-b875-aa92cfa5f678"
        assert others["credits_info"]["credits_deducted"] == 1.0
        assert others["details_processing_time"] == 0.123
        assert others["details_confidence"] == 0.85
        assert others["details_top_label"] == "toxic"

        # Check status
        assert result["status"]["code"] == "200"
        assert result["status"]["message"] == "Success"

    def test_normalize_new_api_response_null_details(self):
        """Test normalization when details is null."""
        server_response = {
            "score": 0,
            "label": "",
            "passed": False,
            "explanation": "",
            "details": None,
            "category": "input-validation",
            "request_id": "test-request-123",
            "credits_info": {
                "credits_deducted": 1,
                "remaining_credits": 212,
                "operation": "input-validation_validation_toxicity",
                "credit_cost": 1
            }
        }

        result = normalize_server_payload(server_response)

        # Check data section
        data = result["data"]
        assert data["metric_name"] == "unknown"  # Default when details is null
        assert data["actual_value"] == 0
        assert data["actual_value_type"] == "float"
        assert data["metric_labels"] == [""]
        assert data["threshold"] == ["Fail"]
        assert data["threshold_score"] == 0.0  # Default when details is null

        # Check others bag
        others = data["others"]
        assert others["details"] is None
        assert others["category"] == "input-validation"

    def test_normalize_new_api_response_passed_true(self):
        """Test normalization when passed is True."""
        server_response = {
            "score": 0.92,
            "label": "relevant",
            "passed": True,
            "explanation": "Content passed context-relevance validation",
            "details": {
                "metric_name": "context-relevance",
                "threshold_score": 0.7
            },
            "category": "rag-grounding",
            "request_id": "test-request-456"
        }

        result = normalize_server_payload(server_response)

        # Check threshold shows "Pass" when passed is True
        data = result["data"]
        assert data["threshold"] == ["Pass"]
        assert data["metric_name"] == "context-relevance"
        assert data["actual_value"] == 0.92
