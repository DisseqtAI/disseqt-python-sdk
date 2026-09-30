"""Factory namespaces for the Disseqt SDK ``Client``.

These mirror the Node SDK's ergonomics (``client.input.toxicity(data, config)``,
``client.rag.faithfulness(...)``, etc.) while staying additive over the Python
class-based validator API. Each factory method builds and returns the
appropriate validator instance so the caller can pass it to
:meth:`Client.validate` (or reuse it, inspect it, etc.).

Two equivalent call styles now produce identical HTTP requests::

    client.validate(client.input.toxicity(data, config))
    client.validate(ToxicityValidator(data=data, config=config))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models.base import SDKConfigInput
from .validators.agentic_behavior import (
    AgentGoalAccuracyValidator,
    FallbackRateValidator,
    IntentResolutionValidator,
    PlanCoherenceValidator,
    PlanOptimalityValidator,
    ToolCallAccuracyValidator,
    ToolFailureRateValidator,
    TopicAdherenceValidator,
)
from .validators.composite.evaluate import CompositeScoreEvaluator
from .validators.input.bias import BiasValidator
from .validators.input.child_safety import ChildSafetyValidator
from .validators.input.gender_bias import GenderBiasValidator
from .validators.input.hate_speech import HateSpeechValidator
from .validators.input.intent_compliance import IntentComplianceValidator
from .validators.input.intent_guard import IntentGuardValidator
from .validators.input.intersectionality import IntersectionalityValidator
from .validators.input.invisible_text import InvisibleTextValidator
from .validators.input.nsfw import NSFWValidator
from .validators.input.political_bias import PoliticalBiasValidator
from .validators.input.prompt_injection import InputPromptInjectionValidator
from .validators.input.racial_bias import RacialBiasValidator
from .validators.input.safety import ToxicityValidator
from .validators.input.self_harm import SelfHarmValidator
from .validators.input.sexual_content import SexualContentValidator
from .validators.input.terrorism import TerrorismValidator
from .validators.input.violence import ViolenceValidator
from .validators.mcp_security.data_leakage import DataLeakageValidator
from .validators.mcp_security.insecure_output import InsecureOutputValidator
from .validators.mcp_security.security import McpPromptInjectionValidator
from .validators.output.accuracy import FactualConsistencyValidator
from .validators.output.answer_relevance import AnswerRelevanceValidator
from .validators.output.bias import OutputBiasValidator
from .validators.output.bleu_score import BleuScoreValidator
from .validators.output.child_safety import OutputChildSafetyValidator
from .validators.output.clarity import ClarityValidator
from .validators.output.coherence import CoherenceValidator
from .validators.output.compression_score import CompressionScoreValidator
from .validators.output.conceptual_similarity import ConceptualSimilarityValidator
from .validators.output.cosine_similarity import CosineSimilarityValidator
from .validators.output.creativity import CreativityValidator
from .validators.output.data_leakage import OutputDataLeakageValidator
from .validators.output.diversity import DiversityValidator
from .validators.output.fuzzy_score import FuzzyScoreValidator
from .validators.output.gender_bias import OutputGenderBiasValidator
from .validators.output.grammar_correctness import GrammarCorrectnessValidator
from .validators.output.hate_speech import OutputHateSpeechValidator
from .validators.output.insecure_output import OutputInsecureOutputValidator
from .validators.output.intent_compliance import OutputIntentComplianceValidator
from .validators.output.intent_guard import OutputIntentGuardValidator
from .validators.output.intersectionality import OutputIntersectionalityValidator
from .validators.output.meteor_score import MeteorScoreValidator
from .validators.output.narrative_continuity import NarrativeContinuityValidator
from .validators.output.nsfw import OutputNSFWValidator
from .validators.output.political_bias import OutputPoliticalBiasValidator
from .validators.output.racial_bias import OutputRacialBiasValidator
from .validators.output.readability import ReadabilityValidator
from .validators.output.response_tone import ResponseToneValidator
from .validators.output.rouge_score import RougeScoreValidator
from .validators.output.self_harm import OutputSelfHarmValidator
from .validators.output.sexual_content import OutputSexualContentValidator
from .validators.output.terrorism import OutputTerrorismValidator
from .validators.output.toxicity import OutputToxicityValidator
from .validators.output.violence import OutputViolenceValidator
from .validators.rag_grounding.context_entities_recall import ContextEntitiesRecallValidator
from .validators.rag_grounding.context_precision import ContextPrecisionValidator
from .validators.rag_grounding.context_recall import ContextRecallValidator
from .validators.rag_grounding.faithfulness import FaithfulnessValidator
from .validators.rag_grounding.grounding import ContextRelevanceValidator
from .validators.rag_grounding.noise_sensitivity import NoiseSensitivityValidator
from .validators.rag_grounding.response_relevancy import ResponseRelevancyValidator
from .validators.themes_classifier.classify import ClassifyValidator

if TYPE_CHECKING:
    from .models.agentic_behaviour import AgenticBehaviourRequest
    from .models.composite_score import CompositeScoreRequest
    from .models.input_validation import InputValidationRequest
    from .models.mcp_security import McpSecurityRequest
    from .models.output_validation import OutputValidationRequest
    from .models.rag_grounding import RagGroundingRequest
    from .models.themes_classifier import ThemesClassifierRequest


class _InputValidatorFactory:
    """Factory namespace for input validators (``client.input.*``)."""

    def toxicity(self, data: InputValidationRequest, config: SDKConfigInput) -> ToxicityValidator:
        return ToxicityValidator(data=data, config=config)

    def bias(self, data: InputValidationRequest, config: SDKConfigInput) -> BiasValidator:
        return BiasValidator(data=data, config=config)

    def prompt_injection(
        self, data: InputValidationRequest, config: SDKConfigInput
    ) -> InputPromptInjectionValidator:
        return InputPromptInjectionValidator(data=data, config=config)

    def intersectionality(
        self, data: InputValidationRequest, config: SDKConfigInput
    ) -> IntersectionalityValidator:
        return IntersectionalityValidator(data=data, config=config)

    def racial_bias(
        self, data: InputValidationRequest, config: SDKConfigInput
    ) -> RacialBiasValidator:
        return RacialBiasValidator(data=data, config=config)

    def gender_bias(
        self, data: InputValidationRequest, config: SDKConfigInput
    ) -> GenderBiasValidator:
        return GenderBiasValidator(data=data, config=config)

    def political_bias(
        self, data: InputValidationRequest, config: SDKConfigInput
    ) -> PoliticalBiasValidator:
        return PoliticalBiasValidator(data=data, config=config)

    def self_harm(self, data: InputValidationRequest, config: SDKConfigInput) -> SelfHarmValidator:
        return SelfHarmValidator(data=data, config=config)

    def violence(self, data: InputValidationRequest, config: SDKConfigInput) -> ViolenceValidator:
        return ViolenceValidator(data=data, config=config)

    def terrorism(self, data: InputValidationRequest, config: SDKConfigInput) -> TerrorismValidator:
        return TerrorismValidator(data=data, config=config)

    def sexual_content(
        self, data: InputValidationRequest, config: SDKConfigInput
    ) -> SexualContentValidator:
        return SexualContentValidator(data=data, config=config)

    def hate_speech(
        self, data: InputValidationRequest, config: SDKConfigInput
    ) -> HateSpeechValidator:
        return HateSpeechValidator(data=data, config=config)

    def nsfw(self, data: InputValidationRequest, config: SDKConfigInput) -> NSFWValidator:
        return NSFWValidator(data=data, config=config)

    def invisible_text(
        self, data: InputValidationRequest, config: SDKConfigInput
    ) -> InvisibleTextValidator:
        return InvisibleTextValidator(data=data, config=config)

    def child_safety(
        self, data: InputValidationRequest, config: SDKConfigInput
    ) -> ChildSafetyValidator:
        return ChildSafetyValidator(data=data, config=config)

    def intent_guard(
        self, data: InputValidationRequest, config: SDKConfigInput
    ) -> IntentGuardValidator:
        return IntentGuardValidator(data=data, config=config)

    def intent_compliance(
        self, data: InputValidationRequest, config: SDKConfigInput
    ) -> IntentComplianceValidator:
        return IntentComplianceValidator(data=data, config=config)


class _OutputValidatorFactory:
    """Factory namespace for output validators (``client.output.*``)."""

    def factual_consistency(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> FactualConsistencyValidator:
        return FactualConsistencyValidator(data=data, config=config)

    def answer_relevance(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> AnswerRelevanceValidator:
        return AnswerRelevanceValidator(data=data, config=config)

    def conceptual_similarity(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> ConceptualSimilarityValidator:
        return ConceptualSimilarityValidator(data=data, config=config)

    def grammar_correctness(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> GrammarCorrectnessValidator:
        return GrammarCorrectnessValidator(data=data, config=config)

    def response_tone(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> ResponseToneValidator:
        return ResponseToneValidator(data=data, config=config)

    def clarity(self, data: OutputValidationRequest, config: SDKConfigInput) -> ClarityValidator:
        return ClarityValidator(data=data, config=config)

    def coherence(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> CoherenceValidator:
        return CoherenceValidator(data=data, config=config)

    def creativity(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> CreativityValidator:
        return CreativityValidator(data=data, config=config)

    def readability(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> ReadabilityValidator:
        return ReadabilityValidator(data=data, config=config)

    def diversity(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> DiversityValidator:
        return DiversityValidator(data=data, config=config)

    def narrative_continuity(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> NarrativeContinuityValidator:
        return NarrativeContinuityValidator(data=data, config=config)

    def bias(self, data: OutputValidationRequest, config: SDKConfigInput) -> OutputBiasValidator:
        return OutputBiasValidator(data=data, config=config)

    def gender_bias(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputGenderBiasValidator:
        return OutputGenderBiasValidator(data=data, config=config)

    def racial_bias(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputRacialBiasValidator:
        return OutputRacialBiasValidator(data=data, config=config)

    def political_bias(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputPoliticalBiasValidator:
        return OutputPoliticalBiasValidator(data=data, config=config)

    def intersectionality(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputIntersectionalityValidator:
        return OutputIntersectionalityValidator(data=data, config=config)

    def toxicity(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputToxicityValidator:
        return OutputToxicityValidator(data=data, config=config)

    def nsfw(self, data: OutputValidationRequest, config: SDKConfigInput) -> OutputNSFWValidator:
        return OutputNSFWValidator(data=data, config=config)

    def terrorism(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputTerrorismValidator:
        return OutputTerrorismValidator(data=data, config=config)

    def violence(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputViolenceValidator:
        return OutputViolenceValidator(data=data, config=config)

    def self_harm(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputSelfHarmValidator:
        return OutputSelfHarmValidator(data=data, config=config)

    def sexual_content(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputSexualContentValidator:
        return OutputSexualContentValidator(data=data, config=config)

    def hate_speech(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputHateSpeechValidator:
        return OutputHateSpeechValidator(data=data, config=config)

    def child_safety(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputChildSafetyValidator:
        return OutputChildSafetyValidator(data=data, config=config)

    def data_leakage(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputDataLeakageValidator:
        return OutputDataLeakageValidator(data=data, config=config)

    def insecure_output(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputInsecureOutputValidator:
        return OutputInsecureOutputValidator(data=data, config=config)

    def intent_guard(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputIntentGuardValidator:
        return OutputIntentGuardValidator(data=data, config=config)

    def intent_compliance(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> OutputIntentComplianceValidator:
        return OutputIntentComplianceValidator(data=data, config=config)

    def bleu_score(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> BleuScoreValidator:
        return BleuScoreValidator(data=data, config=config)

    def rouge_score(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> RougeScoreValidator:
        return RougeScoreValidator(data=data, config=config)

    def meteor_score(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> MeteorScoreValidator:
        return MeteorScoreValidator(data=data, config=config)

    def cosine_similarity(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> CosineSimilarityValidator:
        return CosineSimilarityValidator(data=data, config=config)

    def fuzzy_score(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> FuzzyScoreValidator:
        return FuzzyScoreValidator(data=data, config=config)

    def compression_score(
        self, data: OutputValidationRequest, config: SDKConfigInput
    ) -> CompressionScoreValidator:
        return CompressionScoreValidator(data=data, config=config)


class _RagFactory:
    """Factory namespace for RAG grounding validators (``client.rag.*``)."""

    def context_relevance(
        self, data: RagGroundingRequest, config: SDKConfigInput
    ) -> ContextRelevanceValidator:
        return ContextRelevanceValidator(data=data, config=config)

    def context_recall(
        self, data: RagGroundingRequest, config: SDKConfigInput
    ) -> ContextRecallValidator:
        return ContextRecallValidator(data=data, config=config)

    def context_precision(
        self, data: RagGroundingRequest, config: SDKConfigInput
    ) -> ContextPrecisionValidator:
        return ContextPrecisionValidator(data=data, config=config)

    def context_entities_recall(
        self, data: RagGroundingRequest, config: SDKConfigInput
    ) -> ContextEntitiesRecallValidator:
        return ContextEntitiesRecallValidator(data=data, config=config)

    def noise_sensitivity(
        self, data: RagGroundingRequest, config: SDKConfigInput
    ) -> NoiseSensitivityValidator:
        return NoiseSensitivityValidator(data=data, config=config)

    def response_relevancy(
        self, data: RagGroundingRequest, config: SDKConfigInput
    ) -> ResponseRelevancyValidator:
        return ResponseRelevancyValidator(data=data, config=config)

    def faithfulness(
        self, data: RagGroundingRequest, config: SDKConfigInput
    ) -> FaithfulnessValidator:
        return FaithfulnessValidator(data=data, config=config)


class _AgenticFactory:
    """Factory namespace for agentic behaviour validators (``client.agentic.*``)."""

    def topic_adherence(
        self, data: AgenticBehaviourRequest, config: SDKConfigInput
    ) -> TopicAdherenceValidator:
        return TopicAdherenceValidator(data=data, config=config)

    def tool_call_accuracy(
        self, data: AgenticBehaviourRequest, config: SDKConfigInput
    ) -> ToolCallAccuracyValidator:
        return ToolCallAccuracyValidator(data=data, config=config)

    def tool_failure_rate(
        self, data: AgenticBehaviourRequest, config: SDKConfigInput
    ) -> ToolFailureRateValidator:
        return ToolFailureRateValidator(data=data, config=config)

    def plan_optimality(
        self, data: AgenticBehaviourRequest, config: SDKConfigInput
    ) -> PlanOptimalityValidator:
        return PlanOptimalityValidator(data=data, config=config)

    def agent_goal_accuracy(
        self, data: AgenticBehaviourRequest, config: SDKConfigInput
    ) -> AgentGoalAccuracyValidator:
        return AgentGoalAccuracyValidator(data=data, config=config)

    def intent_resolution(
        self, data: AgenticBehaviourRequest, config: SDKConfigInput
    ) -> IntentResolutionValidator:
        return IntentResolutionValidator(data=data, config=config)

    def plan_coherence(
        self, data: AgenticBehaviourRequest, config: SDKConfigInput
    ) -> PlanCoherenceValidator:
        return PlanCoherenceValidator(data=data, config=config)

    def fallback_rate(
        self, data: AgenticBehaviourRequest, config: SDKConfigInput
    ) -> FallbackRateValidator:
        return FallbackRateValidator(data=data, config=config)


class _McpFactory:
    """Factory namespace for MCP security validators (``client.mcp.*``)."""

    def prompt_injection(
        self, data: McpSecurityRequest, config: SDKConfigInput
    ) -> McpPromptInjectionValidator:
        return McpPromptInjectionValidator(data=data, config=config)

    def data_leakage(
        self, data: McpSecurityRequest, config: SDKConfigInput
    ) -> DataLeakageValidator:
        return DataLeakageValidator(data=data, config=config)

    def insecure_output(
        self, data: McpSecurityRequest, config: SDKConfigInput
    ) -> InsecureOutputValidator:
        return InsecureOutputValidator(data=data, config=config)


class _ThemesFactory:
    """Factory namespace for the themes classifier (``client.themes.*``).

    The themes endpoint does not take a ``config`` — matches Node's
    ``ThemesClassifierHelpers``.
    """

    def classify(self, data: ThemesClassifierRequest) -> ClassifyValidator:
        return ClassifyValidator(data=data)


class _CompositeFactory:
    """Factory namespace for the composite-score evaluator (``client.composite.*``).

    The composite endpoint does not take a ``config`` — matches Node's
    ``CompositeHelpers``.
    """

    def evaluate(self, data: CompositeScoreRequest) -> CompositeScoreEvaluator:
        return CompositeScoreEvaluator(data=data)


__all__ = [
    "_AgenticFactory",
    "_CompositeFactory",
    "_InputValidatorFactory",
    "_McpFactory",
    "_OutputValidatorFactory",
    "_RagFactory",
    "_ThemesFactory",
]
