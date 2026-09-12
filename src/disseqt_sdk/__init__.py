"""Disseqt SDK for Python.

Python SDK for Disseqt validators via the Dataset API.
Decorator-based dynamic registry. Enum-driven slugs.
Normalized responses with a dynamic `others` bag.
"""

from disseqt_logging import configure as configure_logging
from disseqt_logging import get_logger
from disseqt_logging import set_level as set_log_level

from ._version import SDK_VERSION as __version__
from .api_client import DisseqtAPIClient
from .client import Client, HTTPError, SDKVersionBlockedError
from .extensions import (
    BaseAttack,
    BaseGuard,
    BaseLLM,
    BaseMetric,
    BaseMultiTurnAttack,
    BaseSingleTurnAttack,
    BaseVulnerability,
)
from .guardrails import Guardrails, GuardResult
from .models.base import SDKConfigInput
from .policy import (
    DECISION_BLOCK,
    DECISION_BORDERLINE,
    DECISION_PASS,
    BlockedError,
    PolicyDecision,
    PolicyRule,
    PolicyRuleset,
    any_blocking,
    is_async,
    is_blocking,
)
from .policy import parse as parse_policy
from .red_team import RedTeamer, red_team
from .telemetry import cost_accumulator

__all__ = [
    "BaseAttack",
    "BaseGuard",
    "BaseLLM",
    "BaseMetric",
    "BaseMultiTurnAttack",
    "BaseSingleTurnAttack",
    "BaseVulnerability",
    "BlockedError",
    "Client",
    "DECISION_BLOCK",
    "DECISION_BORDERLINE",
    "DECISION_PASS",
    "DisseqtAPIClient",
    "GuardResult",
    "Guardrails",
    "HTTPError",
    "PolicyDecision",
    "PolicyRule",
    "PolicyRuleset",
    "RedTeamer",
    "SDKConfigInput",
    "SDKVersionBlockedError",
    "any_blocking",
    "configure_logging",
    "cost_accumulator",
    "get_logger",
    "is_async",
    "is_blocking",
    "parse_policy",
    "red_team",
    "set_log_level",
]
