"""Composite score evaluation validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...enums import Composite, ValidatorDomain
from ...registry import register_validator

if TYPE_CHECKING:
    from ...models.composite_score import CompositeScoreRequest


def composite_request_handler(validator: CompositeScoreEvaluator) -> dict[str, Any]:
    """Custom request handler for composite score evaluation.

    Composite score sends input_data and options at the top level,
    not wrapped in config_input like other validators.
    """
    return validator.data.to_input_data()


def composite_response_handler(server_response: dict[str, Any]) -> dict[str, Any]:
    """Custom response handler for composite score evaluation.

    Composite score has its own response format with nested breakdown,
    overall_confidence, and credit_details. We return it as-is.
    """
    return server_response


@register_validator(
    domain=ValidatorDomain.COMPOSITE,
    slug=Composite.EVALUATE.value,
    path_template="/api/v1/validators/{domain}/{validator}",
    request_handler=composite_request_handler,
    response_handler=composite_response_handler,
)
@dataclass(slots=True)
class CompositeScoreEvaluator:
    """Composite score evaluator.

    Combines multiple validators into a single weighted score with detailed breakdown.
    Evaluates factual/semantic alignment, language quality, and safety/security/integrity.

    Example:
        ```python
        from disseqt_sdk import Client
        from disseqt_sdk.models.composite_score import CompositeScoreRequest
        from disseqt_sdk.validators.composite.evaluate import CompositeScoreEvaluator

        client = Client(project_id="proj_123", api_key="key_xyz")

        evaluator = CompositeScoreEvaluator(
            data=CompositeScoreRequest(
                llm_input_query="What are the differences between men and women?",
                llm_input_context="Research shows individual differences matter more.",
                llm_output="Women are naturally better at nurturing.",
                evaluation_mode="binary_threshold",
                weights_override={
                    "top_level": {
                        "factual_semantic_alignment": 0.50,
                        "language": 0.25,
                        "safety_security_integrity": 0.25
                    }
                }
            )
        )

        result = client.validate(evaluator)
        print(result["overall_confidence"]["score"])
        print(result["overall_confidence"]["breakdown"])
        ```
    """

    data: CompositeScoreRequest
    _domain: ValidatorDomain = field(init=False, repr=False)
    _slug: str = field(init=False, repr=False)
    _path_template: str = field(
        init=False, repr=False, default="/api/v1/validators/{domain}/{validator}"
    )

    def __post_init__(self) -> None:
        """Initialize domain and slug after dataclass initialization."""
        object.__setattr__(self, "_domain", ValidatorDomain.COMPOSITE)
        object.__setattr__(self, "_slug", Composite.EVALUATE.value)

    @property
    def domain(self) -> ValidatorDomain:
        """Get the validator domain."""
        return self._domain

    @property
    def slug(self) -> str:
        """Get the validator slug."""
        return self._slug

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload.

        This method is used by the default request handler.
        We have a custom request handler, but provide this for completeness.
        """
        return self.data.to_input_data()

