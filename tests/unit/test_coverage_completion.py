"""Tests to complete coverage >95%."""

import pytest

from disseqt_sdk.enums import (
    AgenticBehavior,
    Composite,
    InputValidation,
    McpSecurity,
    OutputValidation,
    RagGrounding,
    ThemesClassifier,
    ValidatorDomain,
)
from disseqt_sdk.models.agentic_behaviour import AgenticBehaviourRequest
from disseqt_sdk.models.base import SDKConfigInput
from disseqt_sdk.models.composite_score import CompositeScoreRequest
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.models.mcp_security import McpSecurityRequest
from disseqt_sdk.models.output_validation import OutputValidationRequest
from disseqt_sdk.models.rag_grounding import RagGroundingRequest
from disseqt_sdk.models.themes_classifier import ThemesClassifierRequest


class TestValidatorPostInit:
    """Test __post_init__ methods for all validators."""

    def test_agentic_behavior_validators_post_init(self):
        """Test all agentic behavior validators initialize correctly."""
        from disseqt_sdk.validators.agentic_behavior.agent_goal_accuracy import (
            AgentGoalAccuracyValidator,
        )
        from disseqt_sdk.validators.agentic_behavior.fallback_rate import (
            FallbackRateValidator,
        )
        from disseqt_sdk.validators.agentic_behavior.intent_resolution import (
            IntentResolutionValidator,
        )
        from disseqt_sdk.validators.agentic_behavior.plan_coherence import (
            PlanCoherenceValidator,
        )
        from disseqt_sdk.validators.agentic_behavior.plan_optimality import (
            PlanOptimalityValidator,
        )
        from disseqt_sdk.validators.agentic_behavior.tool_failure_rate import (
            ToolFailureRateValidator,
        )

        data = AgenticBehaviourRequest(
            conversation_history=["test"],
            tool_calls=[],
            agent_responses=["response"],
        )
        config = SDKConfigInput(threshold=0.5)

        validators = [
            AgentGoalAccuracyValidator(data=data, config=config),
            FallbackRateValidator(data=data, config=config),
            IntentResolutionValidator(data=data, config=config),
            PlanCoherenceValidator(data=data, config=config),
            PlanOptimalityValidator(data=data, config=config),
            ToolFailureRateValidator(data=data, config=config),
        ]

        for validator in validators:
            assert validator.domain == ValidatorDomain.AGENTIC_BEHAVIOR
            assert validator.slug in [e.value for e in AgenticBehavior]

    def test_input_validation_validators_post_init(self):
        """Test all input validation validators initialize correctly."""
        from disseqt_sdk.validators.input.gender_bias import GenderBiasValidator
        from disseqt_sdk.validators.input.hate_speech import HateSpeechValidator
        from disseqt_sdk.validators.input.intersectionality import (
            IntersectionalityValidator,
        )
        from disseqt_sdk.validators.input.invisible_text import InvisibleTextValidator
        from disseqt_sdk.validators.input.nsfw import NSFWValidator
        from disseqt_sdk.validators.input.prompt_injection import (
            InputPromptInjectionValidator,
        )
        from disseqt_sdk.validators.input.racial_bias import RacialBiasValidator
        from disseqt_sdk.validators.input.self_harm import SelfHarmValidator
        from disseqt_sdk.validators.input.sexual_content import SexualContentValidator
        from disseqt_sdk.validators.input.terrorism import TerrorismValidator
        from disseqt_sdk.validators.input.violence import ViolenceValidator

        data = InputValidationRequest(prompt="test prompt")
        config = SDKConfigInput(threshold=0.5)

        validators = [
            GenderBiasValidator(data=data, config=config),
            HateSpeechValidator(data=data, config=config),
            IntersectionalityValidator(data=data, config=config),
            InvisibleTextValidator(data=data, config=config),
            NSFWValidator(data=data, config=config),
            InputPromptInjectionValidator(data=data, config=config),
            RacialBiasValidator(data=data, config=config),
            SelfHarmValidator(data=data, config=config),
            SexualContentValidator(data=data, config=config),
            TerrorismValidator(data=data, config=config),
            ViolenceValidator(data=data, config=config),
        ]

        for validator in validators:
            assert validator.domain == ValidatorDomain.INPUT_VALIDATION
            assert validator.slug in [e.value for e in InputValidation]

    def test_output_validation_validators_post_init(self):
        """Test all output validation validators initialize correctly."""
        from disseqt_sdk.validators.output.answer_relevance import (
            AnswerRelevanceValidator,
        )
        from disseqt_sdk.validators.output.bias import OutputBiasValidator
        from disseqt_sdk.validators.output.bleu_score import BleuScoreValidator
        from disseqt_sdk.validators.output.clarity import ClarityValidator
        from disseqt_sdk.validators.output.coherence import CoherenceValidator
        from disseqt_sdk.validators.output.compression_score import (
            CompressionScoreValidator,
        )
        from disseqt_sdk.validators.output.cosine_similarity import (
            CosineSimilarityValidator,
        )
        from disseqt_sdk.validators.output.data_leakage import OutputDataLeakageValidator
        from disseqt_sdk.validators.output.fuzzy_score import FuzzyScoreValidator
        from disseqt_sdk.validators.output.insecure_output import (
            OutputInsecureOutputValidator,
        )
        from disseqt_sdk.validators.output.meteor_score import MeteorScoreValidator
        from disseqt_sdk.validators.output.rouge_score import RougeScoreValidator
        from disseqt_sdk.validators.output.toxicity import OutputToxicityValidator

        data = OutputValidationRequest(response="test response")
        config = SDKConfigInput(threshold=0.5)

        validators = [
            AnswerRelevanceValidator(data=data, config=config),
            OutputBiasValidator(data=data, config=config),
            BleuScoreValidator(data=data, config=config),
            ClarityValidator(data=data, config=config),
            CoherenceValidator(data=data, config=config),
            CompressionScoreValidator(data=data, config=config),
            CosineSimilarityValidator(data=data, config=config),
            OutputDataLeakageValidator(data=data, config=config),
            FuzzyScoreValidator(data=data, config=config),
            OutputInsecureOutputValidator(data=data, config=config),
            MeteorScoreValidator(data=data, config=config),
            RougeScoreValidator(data=data, config=config),
            OutputToxicityValidator(data=data, config=config),
        ]

        for validator in validators:
            assert validator.domain == ValidatorDomain.OUTPUT_VALIDATION
            assert validator.slug in [e.value for e in OutputValidation]

    def test_rag_grounding_validators_post_init(self):
        """Test all RAG grounding validators initialize correctly."""
        from disseqt_sdk.validators.rag_grounding.context_entities_recall import (
            ContextEntitiesRecallValidator,
        )
        from disseqt_sdk.validators.rag_grounding.context_precision import (
            ContextPrecisionValidator,
        )
        from disseqt_sdk.validators.rag_grounding.context_recall import (
            ContextRecallValidator,
        )
        from disseqt_sdk.validators.rag_grounding.faithfulness import (
            FaithfulnessValidator,
        )
        from disseqt_sdk.validators.rag_grounding.noise_sensitivity import (
            NoiseSensitivityValidator,
        )
        from disseqt_sdk.validators.rag_grounding.response_relevancy import (
            ResponseRelevancyValidator,
        )

        data = RagGroundingRequest(
            prompt="test prompt",
            context="test context",
            response="test response",
        )
        config = SDKConfigInput(threshold=0.5)

        validators = [
            ContextEntitiesRecallValidator(data=data, config=config),
            ContextPrecisionValidator(data=data, config=config),
            ContextRecallValidator(data=data, config=config),
            FaithfulnessValidator(data=data, config=config),
            NoiseSensitivityValidator(data=data, config=config),
            ResponseRelevancyValidator(data=data, config=config),
        ]

        for validator in validators:
            assert validator.domain == ValidatorDomain.RAG_GROUNDING
            assert validator.slug in [e.value for e in RagGrounding]

    def test_mcp_security_validators_post_init(self):
        """Test all MCP security validators initialize correctly."""
        from disseqt_sdk.validators.mcp_security.data_leakage import DataLeakageValidator
        from disseqt_sdk.validators.mcp_security.insecure_output import InsecureOutputValidator

        data = McpSecurityRequest(prompt="test prompt")
        config = SDKConfigInput(threshold=0.5)

        validators = [
            DataLeakageValidator(data=data, config=config),
            InsecureOutputValidator(data=data, config=config),
        ]

        for validator in validators:
            assert validator.domain == ValidatorDomain.MCP_SECURITY
            assert validator.slug in [e.value for e in McpSecurity]


class TestRoutesModule:
    """Test routes module for URL construction."""

    def test_build_validator_url_with_enum_domain(self):
        """Test build_validator_url handles enum domain."""
        from disseqt_sdk.routes import build_validator_url

        # Test with domain as enum
        url = build_validator_url(
            base_url="https://api.test.com",
            domain=ValidatorDomain.COMPOSITE,
            slug=Composite.EVALUATE.value,
            path_template="/api/v1/validators/{domain}/{validator}",
        )

        assert url == "https://api.test.com/api/v1/validators/composite/evaluate"

    def test_build_validator_url_all_domains(self):
        """Test build_validator_url with all domain types."""
        from disseqt_sdk.routes import build_validator_url

        base_url = "https://api.test.com"
        template = "/api/v1/validators/{domain}/{validator}"

        # Test each domain
        test_cases = [
            (ValidatorDomain.INPUT_VALIDATION, "toxicity", "input-validation/toxicity"),
            (ValidatorDomain.OUTPUT_VALIDATION, "accuracy", "output-validation/accuracy"),
            (ValidatorDomain.AGENTIC_BEHAVIOR, "topic", "agentic-behavior/topic"),
            (ValidatorDomain.RAG_GROUNDING, "context", "rag-grounding/context"),
            (ValidatorDomain.MCP_SECURITY, "injection", "mcp-security/injection"),
            (ValidatorDomain.THEMES_CLASSIFIER, "classify", "themes-classifier/classify"),
            (ValidatorDomain.COMPOSITE, "evaluate", "composite/evaluate"),
        ]

        for domain, slug, expected_path in test_cases:
            url = build_validator_url(base_url, domain, slug, template)
            assert url == f"{base_url}/api/v1/validators/{expected_path}"


class TestThemesClassifierHandlers:
    """Test themes classifier custom handlers."""

    def test_themes_request_handler(self):
        """Test themes classifier request handler."""
        from disseqt_sdk.validators.themes_classifier.classify import (
            ClassifyValidator,
            themes_request_handler,
        )

        data = ThemesClassifierRequest(text="Test text")
        validator = ClassifyValidator(data=data)

        payload = themes_request_handler(validator)

        assert "text" in payload
        assert payload["text"] == "Test text"

    def test_themes_response_handler(self):
        """Test themes classifier response handler."""
        from disseqt_sdk.validators.themes_classifier.classify import (
            themes_response_handler,
        )

        server_response = {
            "themes": ["technology", "science"],
            "confidence": 0.95,
            "sub_themes": ["AI", "ML"],
        }

        result = themes_response_handler(server_response)

        assert result == server_response
        assert result["themes"] == ["technology", "science"]


class TestEnumCompleteness:
    """Test all enums are properly defined."""

    def test_all_validator_domains_defined(self):
        """Test all validator domains are defined."""
        domains = [
            ValidatorDomain.INPUT_VALIDATION,
            ValidatorDomain.OUTPUT_VALIDATION,
            ValidatorDomain.RAG_GROUNDING,
            ValidatorDomain.AGENTIC_BEHAVIOR,
            ValidatorDomain.MCP_SECURITY,
            ValidatorDomain.THEMES_CLASSIFIER,
            ValidatorDomain.COMPOSITE,
        ]

        assert len(domains) == 7
        assert all(isinstance(d, ValidatorDomain) for d in domains)

    def test_composite_enum_complete(self):
        """Test Composite enum is properly defined."""
        assert Composite.EVALUATE.value == "evaluate"
        assert len(list(Composite)) == 1

