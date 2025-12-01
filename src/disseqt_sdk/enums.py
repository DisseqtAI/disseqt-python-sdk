"""Enums for Disseqt SDK domains and validator slugs."""

from __future__ import annotations

from enum import Enum


class ValidatorDomain(Enum):
    """Validator domains matching the Dataset API routes."""

    INPUT_VALIDATION = "input-validation"
    OUTPUT_VALIDATION = "output-validation"
    RAG_GROUNDING = "rag-grounding"
    AGENTIC_BEHAVIOR = "agentic-behavior"
    MCP_SECURITY = "mcp-security"
    THEMES_CLASSIFIER = "themes-classifier"
    COMPOSITE = "composite"


class InputValidation(Enum):
    """Input validation validator slugs."""

    TOXICITY = "toxicity"
    BIAS = "bias"
    PROMPT_INJECTION = "prompt-injection"
    INTERSECTIONALITY = "intersectionality"
    RACIAL_BIAS = "racial-bias"
    GENDER_BIAS = "gender-bias"
    SELF_HARM = "self-harm"
    VIOLENCE = "violence"
    TERRORISM = "terrorism"
    SEXUAL_CONTENT = "sexual-content"
    HATE_SPEECH = "hate-speech"
    NSFW = "nsfw"
    INVISIBLE_TEXT = "invisible-text"


class OutputValidation(Enum):
    """Output validation validator slugs."""

    FACTUAL_CONSISTENCY = "factual-consistency"
    ANSWER_RELEVANCE = "answer-relevance"
    BIAS = "bias"
    TOXICITY = "toxicity"
    CLARITY = "clarity"
    COHERENCE = "coherence"
    DATA_LEAKAGE = "data-leakage"
    INSECURE_OUTPUT = "insecure-output"
    BLEU_SCORE = "bleu-score"
    ROUGE_SCORE = "rouge-score"
    METEOR_SCORE = "meteor-score"
    COSINE_SIMILARITY = "cosine-similarity"
    FUZZY_SCORE = "fuzzy-score"
    COMPRESSION_SCORE = "compression-score"


class RagGrounding(Enum):
    """RAG grounding validator slugs."""

    CONTEXT_RELEVANCE = "context-relevance"
    CONTEXT_RECALL = "context-recall"
    CONTEXT_PRECISION = "context-precision"
    CONTEXT_ENTITIES_RECALL = "context-entities-recall"
    NOISE_SENSITIVITY = "noise-sensitivity"
    RESPONSE_RELEVANCY = "response-relevancy"
    FAITHFULNESS = "faithfulness"


class AgenticBehavior(Enum):
    """Agentic behavior validator slugs."""

    TOPIC_ADHERENCE = "topic-adherence"
    TOOL_CALL_ACCURACY = "tool-call-accuracy"
    TOOL_FAILURE_RATE = "tool-failure-rate"
    PLAN_OPTIMALITY = "plan-optimality"
    AGENT_GOAL_ACCURACY = "agent-goal-accuracy"
    INTENT_RESOLUTION = "intent-resolution"
    PLAN_COHERENCE = "plan-coherence"
    FALLBACK_RATE = "fallback-rate"
    CONTEXT_SWITCHING = "context-switching"


class McpSecurity(Enum):
    """MCP security validator slugs."""

    PROMPT_INJECTION = "prompt-injection"
    DATA_LEAKAGE = "data-leakage"
    INSECURE_OUTPUT = "insecure-output"


class ThemesClassifier(Enum):
    """Themes classifier validator slugs."""

    CLASSIFY = "classify"


class Composite(Enum):
    """Composite score evaluation slugs."""

    EVALUATE = "evaluate"
