"""Assertions and metrics: score a recorded run against an ``Expect``.

    from disseqt_agent_testing.assertions import assert_agent
    assert_agent(result, Expect(decision="call", tools=(ToolExpect("issue_refund"),)))

Offline and deterministic: ``evaluate`` never runs an agent and never touches the network.
"""

from . import metrics as _metrics  # noqa: F401  (registers the built-in metrics)
from .api import (
    AgentAssertionError,
    AgentIncompleteError,
    aevaluate,
    assert_agent,
    check,
    evaluate,
    register_post_compose,
)
from .assignment import register_assignment_backend
from .classify import Classification, classify_turn
from .codes import register_code
from .config import EvalConfig, EvalOptions, ExpectConfigError, Lexicons
from .fluent import AgentAssertions
from .judge import Judge, JudgeTask, JudgeVerdict, interpret_verdict
from .lexicons import register_lexicon
from .matching import register_matcher_impl
from .metrics import (
    BaseMetric,
    EvalContext,
    MetricResult,
    get_metric,
    list_metrics,
    register_metric,
)
from .report import DecisionReport, RepeatReport, TriageResult

__all__ = [
    "AgentAssertionError",
    "AgentAssertions",
    "AgentIncompleteError",
    "BaseMetric",
    "Classification",
    "DecisionReport",
    "EvalConfig",
    "EvalContext",
    "EvalOptions",
    "ExpectConfigError",
    "Judge",
    "JudgeTask",
    "JudgeVerdict",
    "Lexicons",
    "MetricResult",
    "RepeatReport",
    "TriageResult",
    "aevaluate",
    "assert_agent",
    "check",
    "classify_turn",
    "evaluate",
    "get_metric",
    "interpret_verdict",
    "list_metrics",
    "register_assignment_backend",
    "register_code",
    "register_lexicon",
    "register_matcher_impl",
    "register_metric",
    "register_post_compose",
]
