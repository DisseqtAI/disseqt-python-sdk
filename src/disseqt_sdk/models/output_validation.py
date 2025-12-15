"""Output validation request model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import _LLMTextFieldsMixin


@dataclass(slots=True)
class OutputValidationRequest(_LLMTextFieldsMixin):
    """Request model for output validation validators.

    Different validators have different field requirements:
    - Response only: grammar_correctness, response_tone, bias, toxicity, clarity,
      coherence, data_leakage, insecure_output, narrative_continuity, diversity,
      readability, nsfw, gender_bias, racial_bias, political_bias, self_harm,
      terrorism, violence, sexual_content, hate_speech
    - Prompt + Response: answer_relevance, conceptual_similarity
    - Context + Response: factual_consistency, creativity, compression_score,
      fuzzy_score, rouge_score, bleu_score, meteor_score, cosine_similarity
    """

    # All fields are optional since different validators need different combinations
    response: str | None = None  # llm_output - most common required field
    prompt: str | None = None  # llm_input_query
    context: str | None = None  # llm_input_context

    def to_input_data(self) -> dict[str, Any]:
        """Convert to input_data format for API payload."""
        return self._to_llm_dict()
