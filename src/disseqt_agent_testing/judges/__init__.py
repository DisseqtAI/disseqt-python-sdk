"""LLM judges: an implementation of the assertions package's ``Judge`` protocol.

    judge = make_judge(FakeJudgeModel([...]))          # or make_judge("openai:gpt-4.1-2025-04-14")
    evaluate(result, expect, judge=judge, use_judge=True)

Phase 4a is the offline engine: caching, replay, voting, prechecks, five judge kinds and test
doubles. Live providers arrive in phase 4b, calibration reports in phase 4c.
"""

from .api import JudgeEngine, JudgeWarning, judge_from_env, make_judge
from .cache import CacheCorrupt, JsonDirStore, cache_key
from .codes import JudgeConfigError
from .config import JudgeConfig, JudgeOptions, KindConfig
from .identity import JudgeIdentity
from .kinds import JudgeKindSpec, Precheck, register_kind
from .ledger import BudgetExceeded, CostLedger
from .models import (
    FakeError,
    FakeJudgeModel,
    FakeRaw,
    FakeRefusal,
    FakeServiceTransport,
    FakeSlow,
    FakeTruncated,
    JudgeModel,
    JudgeRequest,
    JudgeResponse,
    Price,
    ServiceModel,
    StaticJudge,
    register_provider,
)
from .network import allow_judge_requests
from .templates import JudgeTemplate, register_template, verify_lock
from .templates import builtin as _builtin_templates  # noqa: F401  (registers the v1.0 templates)
from .templates import quality as _quality_templates  # noqa: F401
from .verdict import JudgeCallRecord

__all__ = [
    "BudgetExceeded", "CacheCorrupt", "CostLedger", "FakeError", "FakeJudgeModel", "FakeRaw",
    "FakeRefusal", "FakeServiceTransport", "ServiceModel", "FakeSlow", "FakeTruncated", "JsonDirStore", "JudgeCallRecord", "JudgeConfig",
    "JudgeConfigError", "JudgeEngine", "JudgeIdentity", "JudgeKindSpec", "JudgeModel",
    "JudgeOptions", "JudgeRequest", "JudgeResponse", "JudgeTemplate", "JudgeWarning", "KindConfig",
    "Precheck", "Price", "StaticJudge", "allow_judge_requests", "cache_key", "judge_from_env",
    "make_judge", "register_kind", "register_provider", "register_template", "verify_lock",
]  # fmt: skip
