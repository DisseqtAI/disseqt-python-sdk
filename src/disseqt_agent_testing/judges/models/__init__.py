"""Judge models: the protocol, typed errors, test doubles and the provider registry."""

from .base import (
    ExtraMissing,
    JudgeModel,
    JudgeRequest,
    JudgeResponse,
    Price,
    ProviderAuth,
    ProviderBadRequest,
    ProviderError,
    ProviderTransient,
)
from .fake import (
    FakeError,
    FakeJudgeModel,
    FakeRaw,
    FakeRefusal,
    FakeScriptExhausted,
    FakeSlow,
    FakeTruncated,
    StaticJudge,
)
from .registry import register_provider, unregister_provider
from .service import FakeServiceTransport, HttpTransport, ServiceModel, ServiceTransport

__all__ = [
    "ExtraMissing", "FakeError", "FakeJudgeModel", "FakeRaw", "FakeRefusal", "FakeScriptExhausted",
    "FakeSlow", "FakeTruncated", "JudgeModel", "JudgeRequest", "JudgeResponse", "Price",
    "ProviderAuth", "ProviderBadRequest", "ProviderError", "ProviderTransient", "StaticJudge",
    "register_provider", "unregister_provider", "FakeServiceTransport", "HttpTransport",
    "ServiceModel", "ServiceTransport",
]  # fmt: skip
