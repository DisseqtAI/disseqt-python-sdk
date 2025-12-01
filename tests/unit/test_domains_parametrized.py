"""Parametrized tests across all domains and validators."""

import pytest

from disseqt_sdk.enums import ValidatorDomain
from disseqt_sdk.models.agentic_behaviour import AgenticBehaviourRequest
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.models.mcp_security import McpSecurityRequest
from disseqt_sdk.models.output_validation import OutputValidationRequest
from disseqt_sdk.models.rag_grounding import RagGroundingRequest
from disseqt_sdk.validators.agentic_behavior.reliability import TopicAdherenceValidator
from disseqt_sdk.validators.input.bias import BiasValidator
from disseqt_sdk.validators.input.safety import ToxicityValidator
from disseqt_sdk.validators.mcp_security.security import McpPromptInjectionValidator
from disseqt_sdk.validators.output.accuracy import FactualConsistencyValidator
from disseqt_sdk.validators.rag_grounding.grounding import ContextRelevanceValidator


class TestValidatorDomainProperties:
    """Test domain properties across all validators."""

    @pytest.mark.parametrize("validator_class,expected_domain,expected_slug", [
        (ToxicityValidator, ValidatorDomain.INPUT_VALIDATION, "toxicity"),
        (BiasValidator, ValidatorDomain.INPUT_VALIDATION, "bias"),
        (FactualConsistencyValidator, ValidatorDomain.OUTPUT_VALIDATION, "factual-consistency"),
        (TopicAdherenceValidator, ValidatorDomain.AGENTIC_BEHAVIOR, "topic-adherence"),
        (McpPromptInjectionValidator, ValidatorDomain.MCP_SECURITY, "prompt-injection"),
        (ContextRelevanceValidator, ValidatorDomain.RAG_GROUNDING, "context-relevance"),
    ])
    def test_validator_domain_and_slug_properties(
        self,
        validator_class,
        expected_domain,
        expected_slug,
        config
    ):
        """Test that validators have correct domain and slug properties."""
        # Create appropriate request data based on validator type
        if validator_class in [ToxicityValidator, BiasValidator]:
            data = InputValidationRequest(prompt="test prompt")
        elif validator_class == FactualConsistencyValidator:
            data = OutputValidationRequest(response="test response")
        elif validator_class == TopicAdherenceValidator:
            data = AgenticBehaviourRequest(conversation_history=["test"])
        elif validator_class == McpPromptInjectionValidator:
            data = McpSecurityRequest(prompt="test prompt")
        elif validator_class == ContextRelevanceValidator:
            data = RagGroundingRequest(prompt="test prompt")
        else:
            pytest.fail(f"Unknown validator class: {validator_class}")

        validator = validator_class(data=data, config=config)

        assert validator.domain == expected_domain
        assert validator.slug == expected_slug

    @pytest.mark.parametrize("validator_class", [
        ToxicityValidator,
        BiasValidator,
        FactualConsistencyValidator,
        TopicAdherenceValidator,
        McpPromptInjectionValidator,
        ContextRelevanceValidator,
    ])
    def test_validator_payload_structure(self, validator_class, config):
        """Test that all validators produce correct payload structure."""
        # Create appropriate request data
        if validator_class in [ToxicityValidator, BiasValidator]:
            data = InputValidationRequest(prompt="test prompt")
        elif validator_class == FactualConsistencyValidator:
            data = OutputValidationRequest(response="test response")
        elif validator_class == TopicAdherenceValidator:
            data = AgenticBehaviourRequest(conversation_history=["test"])
        elif validator_class == McpPromptInjectionValidator:
            data = McpSecurityRequest(prompt="test prompt")
        elif validator_class == ContextRelevanceValidator:
            data = RagGroundingRequest(prompt="test prompt")

        validator = validator_class(data=data, config=config)
        payload = validator.to_payload()

        # All validators should have these top-level keys
        assert "input_data" in payload
        assert "config_input" in payload

        # Config input should always have threshold
        assert "threshold" in payload["config_input"]
        assert payload["config_input"]["threshold"] == config.threshold

    @pytest.mark.parametrize("validator_class,expected_path_segment", [
        (ToxicityValidator, "input-validation/toxicity"),
        (BiasValidator, "input-validation/bias"),
        (FactualConsistencyValidator, "output-validation/factual-consistency"),
        (TopicAdherenceValidator, "agentic-behavior/topic-adherence"),
        (McpPromptInjectionValidator, "mcp-security/prompt-injection"),
        (ContextRelevanceValidator, "rag-grounding/context-relevance"),
    ])
    def test_validator_url_path_construction(
        self,
        validator_class,
        expected_path_segment,
        config
    ):
        """Test URL path construction for all validators."""
        # Create appropriate request data
        if validator_class in [ToxicityValidator, BiasValidator]:
            data = InputValidationRequest(prompt="test prompt")
        elif validator_class == FactualConsistencyValidator:
            data = OutputValidationRequest(response="test response")
        elif validator_class == TopicAdherenceValidator:
            data = AgenticBehaviourRequest(conversation_history=["test"])
        elif validator_class == McpPromptInjectionValidator:
            data = McpSecurityRequest(prompt="test prompt")
        elif validator_class == ContextRelevanceValidator:
            data = RagGroundingRequest(prompt="test prompt")

        validator = validator_class(data=data, config=config)

        # Test path template formatting
        path = validator._path_template.format(
            domain=validator.domain.value,
            validator=validator.slug
        )

        expected_path = f"/api/v1/sdk/validators/{expected_path_segment}"
        assert path == expected_path


class TestRequestModelFieldMapping:
    """Test field mapping across different request models."""

    def test_llm_text_fields_mixin_mapping(self):
        """Test LLM text fields mapping across different models."""
        test_cases = [
            (InputValidationRequest, "prompt", "context", "response"),
            (McpSecurityRequest, "prompt", "context", "response"),
            (RagGroundingRequest, "prompt", "context", "response"),
        ]

        for request_class, _prompt_field, _context_field, _response_field in test_cases:
            # Test with all fields
            if request_class == InputValidationRequest:
                request = request_class(
                    prompt="test prompt",
                    context="test context",
                    response="test response"
                )
            elif request_class == McpSecurityRequest:
                request = request_class(
                    prompt="test prompt",
                    context="test context",
                    response="test response"
                )
            elif request_class == RagGroundingRequest:
                request = request_class(
                    prompt="test prompt",
                    context="test context",
                    response="test response"
                )

            input_data = request.to_input_data()

            assert input_data["llm_input_query"] == "test prompt"
            assert input_data["llm_input_context"] == "test context"
            assert input_data["llm_output"] == "test response"

    def test_agentic_fields_mixin_mapping(self):
        """Test agentic fields mapping."""
        request = AgenticBehaviourRequest(
            conversation_history=["user: hello", "agent: hi"],
            tool_calls=[{"name": "search", "args": {}}],
            agent_responses=["hi"],
            reference_data={"topics": ["greeting"]}
        )

        input_data = request.to_input_data()

        assert input_data["conversation_history"] == ["user: hello", "agent: hi"]
        assert input_data["tool_calls"] == [{"name": "search", "args": {}}]
        assert input_data["agent_responses"] == ["hi"]
        assert input_data["reference_data"] == {"topics": ["greeting"]}

    def test_output_validation_specific_mapping(self):
        """Test output validation specific mapping."""
        request = OutputValidationRequest(response="test output")
        input_data = request.to_input_data()

        assert input_data["llm_output"] == "test output"
        assert "llm_input_query" not in input_data
        assert "llm_input_context" not in input_data


class TestValidatorInitialization:
    """Test validator initialization across all types."""

    @pytest.mark.parametrize("validator_class", [
        ToxicityValidator,
        BiasValidator,
        FactualConsistencyValidator,
        TopicAdherenceValidator,
        McpPromptInjectionValidator,
        ContextRelevanceValidator,
    ])
    def test_validator_post_init_sets_properties(self, validator_class, config):
        """Test that __post_init__ correctly sets domain and slug."""
        # Create appropriate request data
        if validator_class in [ToxicityValidator, BiasValidator]:
            data = InputValidationRequest(prompt="test")
        elif validator_class == FactualConsistencyValidator:
            data = OutputValidationRequest(response="test")
        elif validator_class == TopicAdherenceValidator:
            data = AgenticBehaviourRequest()
        elif validator_class == McpPromptInjectionValidator:
            data = McpSecurityRequest(prompt="test")
        elif validator_class == ContextRelevanceValidator:
            data = RagGroundingRequest()

        validator = validator_class(data=data, config=config)

        # Verify that domain and slug are set (not None or empty)
        assert validator._domain is not None
        assert validator._slug is not None
        assert validator._slug != ""

        # Verify they match the expected enum values
        assert isinstance(validator._domain, ValidatorDomain)
        assert isinstance(validator._slug, str)

    def test_all_validators_have_unique_domain_slug_combinations(self, config):
        """Test that all validators have unique domain:slug combinations."""
        validators = [
            ToxicityValidator(InputValidationRequest(prompt="test"), config),
            BiasValidator(InputValidationRequest(prompt="test"), config),
            FactualConsistencyValidator(OutputValidationRequest(response="test"), config),
            TopicAdherenceValidator(AgenticBehaviourRequest(), config),
            McpPromptInjectionValidator(McpSecurityRequest(prompt="test"), config),
            ContextRelevanceValidator(RagGroundingRequest(), config),
        ]

        domain_slug_pairs = set()
        for validator in validators:
            pair = (validator.domain, validator.slug)
            assert pair not in domain_slug_pairs, f"Duplicate domain:slug pair: {pair}"
            domain_slug_pairs.add(pair)

        # Should have 6 unique combinations
        assert len(domain_slug_pairs) == 6
