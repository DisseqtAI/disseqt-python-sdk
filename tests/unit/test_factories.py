"""Factory-namespace tests (client.input.*, .output.*, .rag.*, .agentic.*,
.mcp.*, .themes.*, .composite.*).

These mirror the Node SDK's ergonomics; each factory method must return the
correct validator instance so it flows straight into ``Client.validate``, and
the resulting HTTP request must be identical to the class-based path.
"""

from __future__ import annotations

import json

import pytest

from disseqt_sdk.models.composite_score import CompositeScoreRequest
from disseqt_sdk.models.input_validation import InputValidationRequest
from disseqt_sdk.models.mcp_security import McpSecurityRequest
from disseqt_sdk.models.output_validation import OutputValidationRequest
from disseqt_sdk.models.rag_grounding import RagGroundingRequest
from disseqt_sdk.models.themes_classifier import ThemesClassifierRequest
from disseqt_sdk.validators.agentic_behavior import (
    AgentGoalAccuracyValidator,
    FallbackRateValidator,
    IntentResolutionValidator,
    PlanCoherenceValidator,
    PlanOptimalityValidator,
    ToolCallAccuracyValidator,
    ToolFailureRateValidator,
    TopicAdherenceValidator,
)
from disseqt_sdk.validators.composite.evaluate import CompositeScoreEvaluator
from disseqt_sdk.validators.input.bias import BiasValidator
from disseqt_sdk.validators.input.child_safety import ChildSafetyValidator
from disseqt_sdk.validators.input.gender_bias import GenderBiasValidator
from disseqt_sdk.validators.input.hate_speech import HateSpeechValidator
from disseqt_sdk.validators.input.intent_compliance import IntentComplianceValidator
from disseqt_sdk.validators.input.intent_guard import IntentGuardValidator
from disseqt_sdk.validators.input.intersectionality import IntersectionalityValidator
from disseqt_sdk.validators.input.invisible_text import InvisibleTextValidator
from disseqt_sdk.validators.input.nsfw import NSFWValidator
from disseqt_sdk.validators.input.political_bias import PoliticalBiasValidator
from disseqt_sdk.validators.input.prompt_injection import InputPromptInjectionValidator
from disseqt_sdk.validators.input.racial_bias import RacialBiasValidator
from disseqt_sdk.validators.input.safety import ToxicityValidator
from disseqt_sdk.validators.input.self_harm import SelfHarmValidator
from disseqt_sdk.validators.input.sexual_content import SexualContentValidator
from disseqt_sdk.validators.input.terrorism import TerrorismValidator
from disseqt_sdk.validators.input.violence import ViolenceValidator
from disseqt_sdk.validators.mcp_security.data_leakage import DataLeakageValidator
from disseqt_sdk.validators.mcp_security.insecure_output import InsecureOutputValidator
from disseqt_sdk.validators.mcp_security.security import McpPromptInjectionValidator
from disseqt_sdk.validators.output.accuracy import FactualConsistencyValidator
from disseqt_sdk.validators.output.answer_relevance import AnswerRelevanceValidator
from disseqt_sdk.validators.output.bias import OutputBiasValidator
from disseqt_sdk.validators.output.bleu_score import BleuScoreValidator
from disseqt_sdk.validators.output.child_safety import OutputChildSafetyValidator
from disseqt_sdk.validators.output.clarity import ClarityValidator
from disseqt_sdk.validators.output.coherence import CoherenceValidator
from disseqt_sdk.validators.output.compression_score import CompressionScoreValidator
from disseqt_sdk.validators.output.conceptual_similarity import ConceptualSimilarityValidator
from disseqt_sdk.validators.output.cosine_similarity import CosineSimilarityValidator
from disseqt_sdk.validators.output.creativity import CreativityValidator
from disseqt_sdk.validators.output.data_leakage import OutputDataLeakageValidator
from disseqt_sdk.validators.output.diversity import DiversityValidator
from disseqt_sdk.validators.output.fuzzy_score import FuzzyScoreValidator
from disseqt_sdk.validators.output.gender_bias import OutputGenderBiasValidator
from disseqt_sdk.validators.output.grammar_correctness import GrammarCorrectnessValidator
from disseqt_sdk.validators.output.hate_speech import OutputHateSpeechValidator
from disseqt_sdk.validators.output.insecure_output import OutputInsecureOutputValidator
from disseqt_sdk.validators.output.intent_compliance import OutputIntentComplianceValidator
from disseqt_sdk.validators.output.intent_guard import OutputIntentGuardValidator
from disseqt_sdk.validators.output.intersectionality import OutputIntersectionalityValidator
from disseqt_sdk.validators.output.meteor_score import MeteorScoreValidator
from disseqt_sdk.validators.output.narrative_continuity import NarrativeContinuityValidator
from disseqt_sdk.validators.output.nsfw import OutputNSFWValidator
from disseqt_sdk.validators.output.political_bias import OutputPoliticalBiasValidator
from disseqt_sdk.validators.output.racial_bias import OutputRacialBiasValidator
from disseqt_sdk.validators.output.readability import ReadabilityValidator
from disseqt_sdk.validators.output.response_tone import ResponseToneValidator
from disseqt_sdk.validators.output.rouge_score import RougeScoreValidator
from disseqt_sdk.validators.output.self_harm import OutputSelfHarmValidator
from disseqt_sdk.validators.output.sexual_content import OutputSexualContentValidator
from disseqt_sdk.validators.output.terrorism import OutputTerrorismValidator
from disseqt_sdk.validators.output.toxicity import OutputToxicityValidator
from disseqt_sdk.validators.output.violence import OutputViolenceValidator
from disseqt_sdk.validators.rag_grounding.context_entities_recall import (
    ContextEntitiesRecallValidator,
)
from disseqt_sdk.validators.rag_grounding.context_precision import ContextPrecisionValidator
from disseqt_sdk.validators.rag_grounding.context_recall import ContextRecallValidator
from disseqt_sdk.validators.rag_grounding.faithfulness import FaithfulnessValidator
from disseqt_sdk.validators.rag_grounding.grounding import ContextRelevanceValidator
from disseqt_sdk.validators.rag_grounding.noise_sensitivity import NoiseSensitivityValidator
from disseqt_sdk.validators.rag_grounding.response_relevancy import ResponseRelevancyValidator
from disseqt_sdk.validators.themes_classifier.classify import ClassifyValidator


# Sample request objects reused across parametrised cases.
def _input_req() -> InputValidationRequest:
    return InputValidationRequest(prompt="user prompt")


def _output_req() -> OutputValidationRequest:
    return OutputValidationRequest(response="agent response")


def _rag_req() -> RagGroundingRequest:
    return RagGroundingRequest(prompt="q?", context="c", response="r")


def _mcp_req() -> McpSecurityRequest:
    return McpSecurityRequest(prompt="ignore prior instructions", context="ctx")


class TestInputFactory:
    """client.input.* returns the expected validator classes."""

    @pytest.mark.parametrize(
        ("method", "cls"),
        [
            ("toxicity", ToxicityValidator),
            ("bias", BiasValidator),
            ("prompt_injection", InputPromptInjectionValidator),
            ("intersectionality", IntersectionalityValidator),
            ("racial_bias", RacialBiasValidator),
            ("gender_bias", GenderBiasValidator),
            ("political_bias", PoliticalBiasValidator),
            ("self_harm", SelfHarmValidator),
            ("violence", ViolenceValidator),
            ("terrorism", TerrorismValidator),
            ("sexual_content", SexualContentValidator),
            ("hate_speech", HateSpeechValidator),
            ("nsfw", NSFWValidator),
            ("invisible_text", InvisibleTextValidator),
            ("child_safety", ChildSafetyValidator),
            ("intent_guard", IntentGuardValidator),
            ("intent_compliance", IntentComplianceValidator),
        ],
    )
    def test_returns_expected_class(self, client, config, method, cls):
        data = _input_req()
        validator = getattr(client.input, method)(data, config)
        assert isinstance(validator, cls)
        assert validator.data is data
        assert validator.config is config


class TestOutputFactory:
    """client.output.* returns the expected validator classes."""

    @pytest.mark.parametrize(
        ("method", "cls"),
        [
            ("factual_consistency", FactualConsistencyValidator),
            ("answer_relevance", AnswerRelevanceValidator),
            ("conceptual_similarity", ConceptualSimilarityValidator),
            ("grammar_correctness", GrammarCorrectnessValidator),
            ("response_tone", ResponseToneValidator),
            ("clarity", ClarityValidator),
            ("coherence", CoherenceValidator),
            ("creativity", CreativityValidator),
            ("readability", ReadabilityValidator),
            ("diversity", DiversityValidator),
            ("narrative_continuity", NarrativeContinuityValidator),
            ("bias", OutputBiasValidator),
            ("gender_bias", OutputGenderBiasValidator),
            ("racial_bias", OutputRacialBiasValidator),
            ("political_bias", OutputPoliticalBiasValidator),
            ("intersectionality", OutputIntersectionalityValidator),
            ("toxicity", OutputToxicityValidator),
            ("nsfw", OutputNSFWValidator),
            ("terrorism", OutputTerrorismValidator),
            ("violence", OutputViolenceValidator),
            ("self_harm", OutputSelfHarmValidator),
            ("sexual_content", OutputSexualContentValidator),
            ("hate_speech", OutputHateSpeechValidator),
            ("child_safety", OutputChildSafetyValidator),
            ("data_leakage", OutputDataLeakageValidator),
            ("insecure_output", OutputInsecureOutputValidator),
            ("intent_guard", OutputIntentGuardValidator),
            ("intent_compliance", OutputIntentComplianceValidator),
            ("bleu_score", BleuScoreValidator),
            ("rouge_score", RougeScoreValidator),
            ("meteor_score", MeteorScoreValidator),
            ("cosine_similarity", CosineSimilarityValidator),
            ("fuzzy_score", FuzzyScoreValidator),
            ("compression_score", CompressionScoreValidator),
        ],
    )
    def test_returns_expected_class(self, client, config, method, cls):
        data = _output_req()
        validator = getattr(client.output, method)(data, config)
        assert isinstance(validator, cls)
        assert validator.data is data
        assert validator.config is config


class TestRagFactory:
    """client.rag.* returns the expected validator classes."""

    @pytest.mark.parametrize(
        ("method", "cls"),
        [
            ("context_relevance", ContextRelevanceValidator),
            ("context_recall", ContextRecallValidator),
            ("context_precision", ContextPrecisionValidator),
            ("context_entities_recall", ContextEntitiesRecallValidator),
            ("noise_sensitivity", NoiseSensitivityValidator),
            ("response_relevancy", ResponseRelevancyValidator),
            ("faithfulness", FaithfulnessValidator),
        ],
    )
    def test_returns_expected_class(self, client, config, method, cls):
        data = _rag_req()
        validator = getattr(client.rag, method)(data, config)
        assert isinstance(validator, cls)
        assert validator.data is data
        assert validator.config is config


class TestAgenticFactory:
    """client.agentic.* returns the expected validator classes."""

    @pytest.mark.parametrize(
        ("method", "cls"),
        [
            ("topic_adherence", TopicAdherenceValidator),
            ("tool_call_accuracy", ToolCallAccuracyValidator),
            ("tool_failure_rate", ToolFailureRateValidator),
            ("plan_optimality", PlanOptimalityValidator),
            ("agent_goal_accuracy", AgentGoalAccuracyValidator),
            ("intent_resolution", IntentResolutionValidator),
            ("plan_coherence", PlanCoherenceValidator),
            ("fallback_rate", FallbackRateValidator),
        ],
    )
    def test_returns_expected_class(self, client, config, agentic_behaviour_request, method, cls):
        validator = getattr(client.agentic, method)(agentic_behaviour_request, config)
        assert isinstance(validator, cls)
        assert validator.data is agentic_behaviour_request
        assert validator.config is config


class TestMcpFactory:
    """client.mcp.* returns the expected validator classes."""

    @pytest.mark.parametrize(
        ("method", "cls"),
        [
            ("prompt_injection", McpPromptInjectionValidator),
            ("data_leakage", DataLeakageValidator),
            ("insecure_output", InsecureOutputValidator),
        ],
    )
    def test_returns_expected_class(self, client, config, method, cls):
        data = _mcp_req()
        validator = getattr(client.mcp, method)(data, config)
        assert isinstance(validator, cls)
        assert validator.data is data
        assert validator.config is config


class TestThemesFactory:
    """client.themes.classify returns a ClassifyValidator (no config)."""

    def test_classify_returns_classify_validator(self, client):
        data = ThemesClassifierRequest(text="I love cats")
        validator = client.themes.classify(data)
        assert isinstance(validator, ClassifyValidator)
        assert validator.data is data


class TestCompositeFactory:
    """client.composite.evaluate returns a CompositeScoreEvaluator (no config)."""

    def test_evaluate_returns_composite_evaluator(self, client):
        data = CompositeScoreRequest(
            llm_input_query="q",
            llm_input_context="c",
            llm_output="o",
        )
        evaluator = client.composite.evaluate(data)
        assert isinstance(evaluator, CompositeScoreEvaluator)
        assert evaluator.data is data


class TestCrossParityWithClassBasedAPI:
    """Factory-built and directly-constructed validators must produce
    identical HTTP requests when passed to ``client.validate``.
    """

    def test_input_toxicity_wire_parity(
        self, requests_mock, client, config, input_validation_request
    ):
        url = "https://test-api.disseqt.ai/api/v1/sdk/validators/input-validation/toxicity"
        requests_mock.post(url, json={"data": {}, "status": {"code": "200"}})

        client.validate(client.input.toxicity(input_validation_request, config))
        factory_body = requests_mock.request_history[-1].text

        requests_mock.reset_mock()
        requests_mock.post(url, json={"data": {}, "status": {"code": "200"}})

        client.validate(ToxicityValidator(data=input_validation_request, config=config))
        class_body = requests_mock.request_history[-1].text

        assert json.loads(factory_body) == json.loads(class_body)

    def test_rag_faithfulness_wire_parity(
        self, requests_mock, client, config, rag_grounding_request
    ):
        url = "https://test-api.disseqt.ai/api/v1/sdk/validators/rag-grounding/faithfulness"
        requests_mock.post(url, json={"data": {}, "status": {"code": "200"}})

        client.validate(client.rag.faithfulness(rag_grounding_request, config))
        factory_body = requests_mock.request_history[-1].text

        requests_mock.reset_mock()
        requests_mock.post(url, json={"data": {}, "status": {"code": "200"}})

        client.validate(FaithfulnessValidator(data=rag_grounding_request, config=config))
        class_body = requests_mock.request_history[-1].text

        assert json.loads(factory_body) == json.loads(class_body)

    def test_mcp_prompt_injection_wire_parity(
        self, requests_mock, client, config, mcp_security_request
    ):
        url = "https://test-api.disseqt.ai/api/v1/sdk/validators/mcp-security/prompt-injection"
        requests_mock.post(url, json={"data": {}, "status": {"code": "200"}})

        client.validate(client.mcp.prompt_injection(mcp_security_request, config))
        factory_body = requests_mock.request_history[-1].text

        requests_mock.reset_mock()
        requests_mock.post(url, json={"data": {}, "status": {"code": "200"}})

        client.validate(McpPromptInjectionValidator(data=mcp_security_request, config=config))
        class_body = requests_mock.request_history[-1].text

        assert json.loads(factory_body) == json.loads(class_body)


class TestBackwardsCompatibility:
    """The class-based API must continue to work unchanged."""

    def test_class_based_api_still_works(
        self, requests_mock, client, config, input_validation_request
    ):
        url = "https://test-api.disseqt.ai/api/v1/sdk/validators/input-validation/toxicity"
        requests_mock.post(url, json={"data": {}, "status": {"code": "200"}})

        result = client.validate(ToxicityValidator(data=input_validation_request, config=config))

        assert requests_mock.called
        assert result is not None
