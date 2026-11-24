"""The model layer: what a judge model must provide, and the typed provider errors."""

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Protocol, runtime_checkable


class ProviderError(Exception):
    """Base of the typed provider errors the engine knows how to handle."""


class ProviderTransient(ProviderError):
    """429, 5xx, connection reset: retried with backoff. ``retry_after_s`` is honored."""

    def __init__(self, message: str, *, status: int | str = "", retry_after_s: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after_s = retry_after_s


class ProviderAuth(ProviderError):
    """401/403 or a missing key: never retried; opens a circuit for the model."""


class ProviderBadRequest(ProviderError):
    """A 400. One naming ``temperature`` or ``top_p`` triggers a single resend without them."""


class ExtraMissing(ProviderError):
    """The provider SDK is not installed (names the extra to install)."""

    def __init__(self, extra: str):
        super().__init__(extra)
        self.extra = extra


@dataclass(slots=True, frozen=True)
class Price:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


@dataclass(slots=True, frozen=True)
class JudgeRequest:
    messages: tuple[Mapping[str, str], ...]
    output_schema: Mapping[str, Any]
    schema_name: str
    params: Mapping[str, Any] = field(
        default_factory=dict
    )  # temperature, top_p, max_tokens, seed...
    sample_index: int = 0
    attempt_timeout_s: float = 60.0
    _: KW_ONLY
    kind: str = ""  # context for test doubles; providers ignore it
    order: int = 0


@dataclass(slots=True, frozen=True)
class JudgeResponse:
    text: str
    parsed: Mapping[str, Any] | None = None
    stop_reason: str = "stop"  # stop | max_tokens | refusal | content_filter | other
    usage: Mapping[str, int] | None = None
    cost_usd: float = 0.0
    model_id: str = ""
    system_fingerprint: str | None = None
    dropped_params: tuple[str, ...] = ()
    latency_ms: float = 0.0  # never cached, never in verdicts


@runtime_checkable
class JudgeModel(Protocol):
    id: str
    base_url: str | None
    networked: bool

    async def complete(self, req: JudgeRequest) -> JudgeResponse: ...

    def price(self) -> Price | None: ...


def safe_int(value: Any) -> int:
    """A non-negative int from whatever a provider sent; garbage counts as 0 (never raises)."""
    if isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(number, 0)


def total_tokens(usage: Mapping[str, Any] | None) -> int:
    if not usage:
        return 0
    return safe_int(usage.get("input_tokens", 0)) + safe_int(usage.get("output_tokens", 0))


_ = Sequence
