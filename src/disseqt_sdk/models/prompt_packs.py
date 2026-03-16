"""Prompt Packs request models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(slots=True)
class PromptPackCategory:
    """A category within a prompt pack generation request."""

    main_category: str
    subcategory: str
    prompts_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API payload."""
        return {
            "main_category": self.main_category,
            "subcategory": self.subcategory,
            "prompts_count": self.prompts_count,
        }


@dataclass(slots=True)
class GeneratePromptPackRequest:
    """Request model for generating a prompt pack.

    organization_id is supplied by Kong gateway (from URL/headers), not in the request body.

    Attributes:
        pack_name: Name of the prompt pack
        pack_short_desc: Short description of the pack
        author: Author of the pack
        domain: Domain/area the pack covers
        generation_type: Type of generation (e.g. "AI")
        categories: List of categories with prompt counts
    """

    pack_name: str
    pack_short_desc: str
    author: str
    domain: str
    generation_type: str
    categories: list[PromptPackCategory]

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload."""
        return {
            "pack_name": self.pack_name,
            "pack_short_desc": self.pack_short_desc,
            "author": self.author,
            "domain": self.domain,
            "generation_type": self.generation_type,
            "categories": [c.to_dict() for c in self.categories],
        }


@dataclass(slots=True)
class CreateRunRequest:
    """Request model for creating a prompt pack run.

    project_id and organization_id are supplied by Kong gateway (from URL/headers), not in the request body.

    Attributes:
        run_name: Name for this run
        run_type: Type of run (e.g. "evaluation")
        api_key: LLM provider API key
        model_name: Model to use (e.g. "gpt-4")
        provider: LLM provider (e.g. "openai")
    """

    run_name: str
    run_type: str
    api_key: str
    model_name: str
    provider: str

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload."""
        return {
            "run_name": self.run_name,
            "run_type": self.run_type,
            "api_key": self.api_key,
            "model_name": self.model_name,
            "provider": self.provider,
        }


class PromptPackOutputValidationCategory(str, Enum):
    """Category for prompt pack output validation (aligns with API and canonical layers)."""

    OUTPUT_VALIDATION = "output-validation"
    INPUT_VALIDATION = "input-validation"
    RAG_GROUNDING = "rag-grounding"
    AGENTIC_BEHAVIOR = "agentic-behavior"
    MCP_SECURITY = "mcp-security"


class OutputValidationMetric(str, Enum):
    """Output validation metric slugs supported by the prompt pack output validation API.

    Aligns with the metrics list: Coherence, Diversity, Toxicity, Data Leakage, etc.
    Use these with category PromptPackOutputValidationCategory.OUTPUT_VALIDATION.
    """

    COHERENCE = "coherence"
    DIVERSITY = "diversity"
    TOXICITY = "toxicity"
    DATA_LEAKAGE = "data-leakage"
    NARRATIVE_CONTINUITY = "narrative-continuity"
    INSECURE_OUTPUT = "insecure-output"
    RESPONSE_TONE = "response-tone"
    CLARITY = "clarity"
    READABILITY = "readability"
    GRAMMAR_CORRECTNESS = "grammar-correctness"
    NSFW = "nsfw"
    BIAS = "bias"
    GENDER_BIAS = "gender-bias"
    RACIAL_BIAS = "racial-bias"
    POLITICAL_BIAS = "political-bias"
    INTERSECTIONALITY = "intersectionality"
    HATE_SPEECH = "hate-speech"
    SEXUAL_CONTENT = "sexual-content"
    TERRORISM = "terrorism"
    VIOLENCE = "violence"
    SELF_HARM = "self-harm"


@dataclass(slots=True)
class MetricEvaluation:
    """A single metric to evaluate in a prompt pack output validation run."""

    metric_name: str
    category: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to API payload item."""
        return {
            "metric_name": self.metric_name,
            "category": self.category,
        }


@dataclass(slots=True)
class PromptPackOutputValidationRequest:
    """Request model for creating a prompt pack output validation on a run.

    Request body shape:
        {
          "prompt_pack_output_validation_run_name": "SDK Test Validation",
          "metric_evaluations": [
            {"metric_name": "hate-speech", "category": "output-validation"},
            {"metric_name": "toxicity", "category": "output-validation"}
          ]
        }

    Use OutputValidationMetric and PromptPackOutputValidationCategory for type-safe metric/category.
    """

    prompt_pack_output_validation_run_name: str
    metric_evaluations: list[MetricEvaluation]

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload."""
        return {
            "prompt_pack_output_validation_run_name": self.prompt_pack_output_validation_run_name,
            "metric_evaluations": [m.to_dict() for m in self.metric_evaluations],
        }


@dataclass(slots=True)
class CreateOutputValidationRequest:
    """Deprecated: use PromptPackOutputValidationRequest with metric_evaluations instead."""

    validation_type: str
    metrics: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """Convert to new API payload shape for backward compatibility."""
        run_name = self.validation_type or "Validation"
        metric_evaluations = [
            MetricEvaluation(metric_name=m, category=PromptPackOutputValidationCategory.OUTPUT_VALIDATION.value)
            for m in self.metrics
        ]
        return PromptPackOutputValidationRequest(
            prompt_pack_output_validation_run_name=run_name,
            metric_evaluations=metric_evaluations,
        ).to_payload()


@dataclass(slots=True)
class PaginationParams:
    """Pagination parameters for list endpoints.

    Attributes:
        limit: Maximum number of results to return
        offset: Number of results to skip
    """

    limit: int = 10
    offset: int = 0

    def to_query_params(self) -> dict[str, str]:
        """Convert to query parameters."""
        return {
            "limit": str(self.limit),
            "offset": str(self.offset),
        }
