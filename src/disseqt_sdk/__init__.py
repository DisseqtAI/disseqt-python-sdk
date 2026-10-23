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
    BaseLLM,
    BaseMetric,
    BaseMultiTurnAttack,
    BaseSingleTurnAttack,
    BaseVulnerability,
)
from .models.base import SDKConfigInput
from .policy import (  # deprecated shims, kept for 0.11 callers
    PolicyDecision,
    PolicyRule,
    PolicyRuleset,
    any_blocking,
    is_async,
    is_blocking,
)
from .policy import parse as parse_policy
from .resources import (
    ByovValidatorsResource,
    McpTargetsResource,
    OutputValidationsResource,
    PacksResource,
    RagTargetsResource,
    RunsResource,
    SessionsResource,
    TargetsResource,
    VulnerabilitiesResource,
)

__all__ = [
    "BaseAttack",
    "BaseLLM",
    "BaseMetric",
    "BaseMultiTurnAttack",
    "BaseSingleTurnAttack",
    "BaseVulnerability",
    "ByovValidatorsResource",
    "Client",
    "DisseqtAPIClient",
    "HTTPError",
    "McpTargetsResource",
    "OutputValidationsResource",
    "PacksResource",
    "PolicyDecision",
    "PolicyRule",
    "PolicyRuleset",
    "RagTargetsResource",
    "RunsResource",
    "SDKConfigInput",
    "SDKVersionBlockedError",
    "SessionsResource",
    "TargetsResource",
    "VulnerabilitiesResource",
    "any_blocking",
    "configure_logging",
    "get_logger",
    "is_async",
    "is_blocking",
    "parse_policy",
    "set_log_level",
]
