"""Test composite score evaluation feature."""

import pytest

from disseqt_sdk.enums import Composite, ValidatorDomain
from disseqt_sdk.models.composite_score import CompositeScoreRequest
from disseqt_sdk.validators.composite.evaluate import (
    CompositeScoreEvaluator,
    composite_request_handler,
    composite_response_handler,
)


class TestCompositeScoreRequest:
    """Test CompositeScoreRequest model."""

    def test_minimal_request_with_required_fields(self):
        """Test minimal composite score request with only required fields."""
        request = CompositeScoreRequest(
            llm_input_query="What is AI?",
            llm_output="AI is artificial intelligence.",
        )

        assert request.llm_input_query == "What is AI?"
        assert request.llm_output == "AI is artificial intelligence."
        assert request.llm_input_context is None
        assert request.evaluation_mode == "binary_threshold"
        assert request.weights_override is None
        assert request.labels_thresholds_override is None
        assert request.overall_confidence is None

    def test_request_with_context(self):
        """Test composite score request with context."""
        request = CompositeScoreRequest(
            llm_input_query="What is AI?",
            llm_output="AI is artificial intelligence.",
            llm_input_context="Context about AI technology",
        )

        assert request.llm_input_context == "Context about AI technology"

    def test_request_with_custom_evaluation_mode(self):
        """Test composite score request with custom evaluation mode."""
        request = CompositeScoreRequest(
            llm_input_query="Test query",
            llm_output="Test output",
            evaluation_mode="weighted_average",
        )

        assert request.evaluation_mode == "weighted_average"

    def test_request_with_weights_override(self):
        """Test composite score request with weights override."""
        weights = {
            "top_level": {
                "factual_semantic_alignment": 0.50,
                "language": 0.25,
                "safety_security_integrity": 0.25,
            }
        }

        request = CompositeScoreRequest(
            llm_input_query="Test",
            llm_output="Output",
            weights_override=weights,
        )

        assert request.weights_override == weights
        assert request.weights_override["top_level"]["factual_semantic_alignment"] == 0.50

    def test_request_with_labels_thresholds_override(self):
        """Test composite score request with labels thresholds override."""
        labels_thresholds = {
            "factual_semantic_alignment": {
                "custom_labels": ["Low", "Medium", "High"],
                "label_thresholds": [0.3, 0.7],
            }
        }

        request = CompositeScoreRequest(
            llm_input_query="Test",
            llm_output="Output",
            labels_thresholds_override=labels_thresholds,
        )

        assert request.labels_thresholds_override == labels_thresholds

    def test_request_with_overall_confidence(self):
        """Test composite score request with overall confidence config."""
        overall_confidence = {
            "custom_labels": ["Low", "Medium", "High", "Very High"],
            "label_thresholds": [0.4, 0.6, 0.8],
        }

        request = CompositeScoreRequest(
            llm_input_query="Test",
            llm_output="Output",
            overall_confidence=overall_confidence,
        )

        assert request.overall_confidence == overall_confidence

    def test_to_input_data_minimal(self):
        """Test to_input_data with minimal configuration."""
        request = CompositeScoreRequest(
            llm_input_query="What is Python?",
            llm_output="Python is a programming language.",
        )

        data = request.to_input_data()

        assert "input_data" in data
        assert "options" in data

        # Check input_data
        input_data = data["input_data"]
        assert input_data["llm_input_query"] == "What is Python?"
        assert input_data["llm_output"] == "Python is a programming language."
        assert "llm_input_context" not in input_data

        # Check options
        options = data["options"]
        assert options["evaluation_mode"] == "binary_threshold"
        assert "weights_override" not in options
        assert "labels_thresholds_override" not in options
        assert "overall_confidence" not in options

    def test_to_input_data_with_context(self):
        """Test to_input_data includes context when provided."""
        request = CompositeScoreRequest(
            llm_input_query="Query",
            llm_output="Output",
            llm_input_context="Some context here",
        )

        data = request.to_input_data()
        assert data["input_data"]["llm_input_context"] == "Some context here"

    def test_to_input_data_complete(self):
        """Test to_input_data with all fields populated."""
        weights = {"top_level": {"factual_semantic_alignment": 0.5}}
        labels = {"factual_semantic_alignment": {"custom_labels": ["Low", "High"]}}
        confidence = {"custom_labels": ["Low", "High"]}

        request = CompositeScoreRequest(
            llm_input_query="Query",
            llm_output="Output",
            llm_input_context="Context",
            evaluation_mode="weighted",
            weights_override=weights,
            labels_thresholds_override=labels,
            overall_confidence=confidence,
        )

        data = request.to_input_data()

        assert data["input_data"]["llm_input_query"] == "Query"
        assert data["input_data"]["llm_output"] == "Output"
        assert data["input_data"]["llm_input_context"] == "Context"
        assert data["options"]["evaluation_mode"] == "weighted"
        assert data["options"]["weights_override"] == weights
        assert data["options"]["labels_thresholds_override"] == labels
        assert data["options"]["overall_confidence"] == confidence


class TestCompositeScoreEvaluator:
    """Test CompositeScoreEvaluator validator."""

    def test_evaluator_initialization(self):
        """Test CompositeScoreEvaluator initialization."""
        request = CompositeScoreRequest(
            llm_input_query="Test query",
            llm_output="Test output",
        )

        evaluator = CompositeScoreEvaluator(data=request)

        assert evaluator.data == request
        assert evaluator.domain == ValidatorDomain.COMPOSITE
        assert evaluator.slug == Composite.EVALUATE.value
        assert evaluator._path_template == "/api/v1/validators/{domain}/{validator}"

    def test_evaluator_domain_property(self):
        """Test domain property returns correct value."""
        request = CompositeScoreRequest(
            llm_input_query="Test",
            llm_output="Output",
        )
        evaluator = CompositeScoreEvaluator(data=request)

        assert evaluator.domain == ValidatorDomain.COMPOSITE
        assert evaluator.domain.value == "composite"

    def test_evaluator_slug_property(self):
        """Test slug property returns correct value."""
        request = CompositeScoreRequest(
            llm_input_query="Test",
            llm_output="Output",
        )
        evaluator = CompositeScoreEvaluator(data=request)

        assert evaluator.slug == "evaluate"
        assert evaluator.slug == Composite.EVALUATE.value

    def test_evaluator_to_payload(self):
        """Test evaluator to_payload method."""
        request = CompositeScoreRequest(
            llm_input_query="Query",
            llm_output="Output",
            llm_input_context="Context",
        )
        evaluator = CompositeScoreEvaluator(data=request)

        payload = evaluator.to_payload()

        assert "input_data" in payload
        assert "options" in payload
        assert payload["input_data"]["llm_input_query"] == "Query"
        assert payload["input_data"]["llm_output"] == "Output"
        assert payload["input_data"]["llm_input_context"] == "Context"

    def test_evaluator_with_full_configuration(self):
        """Test evaluator with complete configuration."""
        weights = {
            "top_level": {
                "factual_semantic_alignment": 0.50,
                "language": 0.25,
                "safety_security_integrity": 0.25,
            },
            "submetrics": {
                "factual_semantic_alignment": {
                    "factual_consistency": 0.70,
                    "answer_relevance": 0.05,
                }
            },
        }

        labels = {
            "factual_semantic_alignment": {
                "custom_labels": ["Low", "Medium", "High", "Excellent"],
                "label_thresholds": [0.4, 0.65, 0.8],
            }
        }

        confidence = {
            "custom_labels": ["Low", "Medium", "High", "Very High"],
            "label_thresholds": [0.4, 0.55, 0.8],
        }

        request = CompositeScoreRequest(
            llm_input_query="What are the differences between men and women?",
            llm_input_context="Research shows individual differences matter more.",
            llm_output="Women are naturally better at nurturing.",
            evaluation_mode="binary_threshold",
            weights_override=weights,
            labels_thresholds_override=labels,
            overall_confidence=confidence,
        )

        evaluator = CompositeScoreEvaluator(data=request)
        payload = evaluator.to_payload()

        # Verify complete payload structure
        assert payload["options"]["weights_override"] == weights
        assert payload["options"]["labels_thresholds_override"] == labels
        assert payload["options"]["overall_confidence"] == confidence


class TestCompositeRequestHandler:
    """Test composite score custom request handler."""

    def test_request_handler_returns_correct_structure(self):
        """Test request handler returns input_data and options."""
        request = CompositeScoreRequest(
            llm_input_query="Test",
            llm_output="Output",
        )
        evaluator = CompositeScoreEvaluator(data=request)

        payload = composite_request_handler(evaluator)

        assert "input_data" in payload
        assert "options" in payload

    def test_request_handler_with_minimal_config(self):
        """Test request handler with minimal configuration."""
        request = CompositeScoreRequest(
            llm_input_query="Query",
            llm_output="Response",
        )
        evaluator = CompositeScoreEvaluator(data=request)

        payload = composite_request_handler(evaluator)

        assert payload["input_data"]["llm_input_query"] == "Query"
        assert payload["input_data"]["llm_output"] == "Response"
        assert payload["options"]["evaluation_mode"] == "binary_threshold"

    def test_request_handler_with_full_config(self):
        """Test request handler with full configuration."""
        weights = {"top_level": {"factual_semantic_alignment": 0.5}}

        request = CompositeScoreRequest(
            llm_input_query="Q",
            llm_output="A",
            llm_input_context="C",
            weights_override=weights,
        )
        evaluator = CompositeScoreEvaluator(data=request)

        payload = composite_request_handler(evaluator)

        assert payload["input_data"]["llm_input_context"] == "C"
        assert payload["options"]["weights_override"] == weights


class TestCompositeResponseHandler:
    """Test composite score custom response handler."""

    def test_response_handler_returns_response_as_is(self):
        """Test response handler returns server response unchanged."""
        server_response = {
            "overall_confidence": {
                "score": 0.75,
                "label": "High Confidence",
                "breakdown": {},
            },
            "success": True,
        }

        result = composite_response_handler(server_response)

        assert result == server_response
        assert result["overall_confidence"]["score"] == 0.75

    def test_response_handler_with_complete_response(self):
        """Test response handler with complete API response."""
        server_response = {
            "overall_confidence": {
                "score": 0.5575,
                "label": "High Confidence",
                "scoring_type": "weighted_binary",
                "breakdown": {
                    "factual_semantic_alignment": {
                        "score": 0.19,
                        "label": "Low Accuracy",
                        "passed_metrics": 4,
                        "total_metrics": 9,
                    },
                    "language": {
                        "score": 1.0,
                        "label": "Excellent Quality",
                        "passed_metrics": 3,
                        "total_metrics": 3,
                    },
                },
                "processing_time_ms": 2075,
                "total_metrics_evaluated": 18,
            },
            "success": True,
            "credit_details": {
                "credits_deducted": 18,
                "credits_remaining": 2426,
            },
        }

        result = composite_response_handler(server_response)

        assert result == server_response
        assert result["overall_confidence"]["score"] == 0.5575
        assert result["credit_details"]["credits_deducted"] == 18

    def test_response_handler_preserves_all_fields(self):
        """Test response handler preserves all response fields."""
        server_response = {
            "overall_confidence": {
                "score": 0.8,
                "label": "High",
                "breakdown": {
                    "category1": {"score": 0.7},
                    "category2": {"score": 0.9},
                },
            },
            "success": True,
            "credit_details": {"credits_deducted": 10},
            "custom_field": "custom_value",
        }

        result = composite_response_handler(server_response)

        assert "overall_confidence" in result
        assert "success" in result
        assert "credit_details" in result
        assert "custom_field" in result
        assert result["custom_field"] == "custom_value"


class TestCompositeScoreIntegration:
    """Test composite score integration with SDK."""

    def test_composite_evaluator_registered_in_registry(self):
        """Test that composite evaluator is registered."""
        from disseqt_sdk.registry import get_validator_metadata

        metadata = get_validator_metadata(ValidatorDomain.COMPOSITE, "evaluate")

        assert metadata is not None
        assert metadata["domain"] == ValidatorDomain.COMPOSITE
        assert metadata["slug"] == "evaluate"
        assert metadata["request_handler"] is not None
        assert metadata["response_handler"] is not None

    def test_composite_evaluator_has_custom_handlers(self):
        """Test that composite evaluator uses custom handlers."""
        from disseqt_sdk.registry import get_validator_metadata

        metadata = get_validator_metadata(ValidatorDomain.COMPOSITE, "evaluate")

        assert metadata["request_handler"] == composite_request_handler
        assert metadata["response_handler"] == composite_response_handler

    def test_composite_domain_in_enum(self):
        """Test that COMPOSITE domain exists in ValidatorDomain enum."""
        assert hasattr(ValidatorDomain, "COMPOSITE")
        assert ValidatorDomain.COMPOSITE.value == "composite"

    def test_evaluate_slug_in_enum(self):
        """Test that EVALUATE slug exists in Composite enum."""
        assert hasattr(Composite, "EVALUATE")
        assert Composite.EVALUATE.value == "evaluate"

    def test_composite_evaluator_in_validators_package(self):
        """Test that composite evaluator can be imported."""
        from disseqt_sdk.validators.composite import CompositeScoreEvaluator as Evaluator

        assert Evaluator is not None
        assert Evaluator == CompositeScoreEvaluator

    def test_composite_score_request_in_models_package(self):
        """Test that composite score request can be imported."""
        from disseqt_sdk.models import CompositeScoreRequest as Request

        assert Request is not None
        assert Request == CompositeScoreRequest


class TestCompositeScoreEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_inputs(self):
        """Test composite score with empty string inputs."""
        request = CompositeScoreRequest(
            llm_input_query="",
            llm_output="",
        )

        assert request.llm_input_query == ""
        assert request.llm_output == ""

        data = request.to_input_data()
        assert data["input_data"]["llm_input_query"] == ""
        assert data["input_data"]["llm_output"] == ""

    def test_very_long_text_inputs(self):
        """Test composite score with very long text inputs."""
        long_text = "A" * 10000

        request = CompositeScoreRequest(
            llm_input_query=long_text,
            llm_output=long_text,
            llm_input_context=long_text,
        )

        data = request.to_input_data()
        assert len(data["input_data"]["llm_input_query"]) == 10000
        assert len(data["input_data"]["llm_output"]) == 10000
        assert len(data["input_data"]["llm_input_context"]) == 10000

    def test_special_characters_in_text(self):
        """Test composite score with special characters."""
        special_text = 'Special chars: @#$%^&*()_+-={}[]|\\:";\'<>?,./~`'

        request = CompositeScoreRequest(
            llm_input_query=special_text,
            llm_output=special_text,
        )

        data = request.to_input_data()
        assert data["input_data"]["llm_input_query"] == special_text

    def test_unicode_text(self):
        """Test composite score with unicode characters."""
        unicode_text = "Hello 世界 🌍 مرحبا привет"

        request = CompositeScoreRequest(
            llm_input_query=unicode_text,
            llm_output=unicode_text,
        )

        data = request.to_input_data()
        assert data["input_data"]["llm_input_query"] == unicode_text

    def test_empty_weights_override(self):
        """Test composite score with empty weights override."""
        request = CompositeScoreRequest(
            llm_input_query="Test",
            llm_output="Output",
            weights_override={},
        )

        data = request.to_input_data()
        # Empty dict is falsy in Python, so it won't be included in options
        assert "weights_override" not in data["options"]

    def test_nested_weights_structure(self):
        """Test composite score with deeply nested weights."""
        weights = {
            "top_level": {
                "cat1": 0.33,
                "cat2": 0.33,
                "cat3": 0.34,
            },
            "submetrics": {
                "cat1": {
                    "metric1": 0.5,
                    "metric2": 0.3,
                    "metric3": 0.2,
                }
            },
        }

        request = CompositeScoreRequest(
            llm_input_query="Test",
            llm_output="Output",
            weights_override=weights,
        )

        data = request.to_input_data()
        assert data["options"]["weights_override"]["top_level"]["cat1"] == 0.33
        assert (
            data["options"]["weights_override"]["submetrics"]["cat1"]["metric1"]
            == 0.5
        )

