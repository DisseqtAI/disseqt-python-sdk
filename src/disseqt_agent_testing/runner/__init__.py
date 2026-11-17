"""The runner: turn test cases into a deterministic, JSON-serializable ``SuiteResult``.

    result = run_suite("tests/agent_cases", agent_factory=make_agent, config=SuiteConfig(default_repeats=5))
    sys.exit(result.exit_code)

Never imports pytest. Each repeat is one harness run scored by one ``evaluate``; each case is
decided by the bounds rule over its repeats (see ``stats.decide``).
"""

from .aengine import arun_case, arun_suite
from .config import SuiteConfig, SuiteConfigError
from .engine import run_case, run_suite
from .errors import CODES, DisseqtWarning, format_code
from .events import CaseFinished, CaseStarted, RepeatFinished, SuiteFinished
from .filter import CaseFilter
from .limiter import JudgeLimiter, LimitedJudge, Limiter
from .loader import (
    LoadedSuite,
    format_errors,
    load_cases,
    register_case_format,
    register_dataset_format,
)
from .models import (
    CaseResult,
    CaseSource,
    CaseSpec,
    CollectionErrorRecord,
    RepeatRecord,
    RunBlock,
    SuiteResult,
)
from .pool import DaemonPool, Executor
from .seeds import RepeatContext, current_repeat, derive_seed
from .stats import decide, pass_at_1, pass_at_k, pass_hat_k, wilson_95
from .summarize import failure_codes_of, summarize_repeats

__all__ = [
    "CODES", "CaseFilter", "CaseFinished", "CaseResult", "CaseSource", "CaseSpec", "CaseStarted",
    "CollectionErrorRecord", "DaemonPool", "DisseqtWarning", "Executor", "JudgeLimiter",
    "LimitedJudge", "Limiter", "LoadedSuite", "RepeatContext", "RepeatFinished", "RepeatRecord",
    "RunBlock", "SuiteConfig", "SuiteConfigError", "SuiteFinished", "SuiteResult", "current_repeat",
    "decide", "derive_seed", "failure_codes_of", "format_code", "format_errors", "load_cases",
    "pass_at_1", "pass_at_k", "pass_hat_k", "register_case_format", "register_dataset_format",
    "arun_case", "arun_suite", "run_case", "run_suite", "summarize_repeats", "wilson_95",
]  # fmt: skip
