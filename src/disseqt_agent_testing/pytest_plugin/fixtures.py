"""Fixtures for Python-defined agent cases. Imports stay lazy: defining them costs nothing."""

import time
from typing import Any

import pytest


@pytest.fixture
def disseqt_config(request: pytest.FixtureRequest) -> Any:
    """The resolved ``SuiteConfig`` for this session."""
    from .plugin import get_impl

    return get_impl(request.config).suite_config()


@pytest.fixture
def disseqt_judge(request: pytest.FixtureRequest) -> Any:
    """The judge from ``pytest_disseqt_judge``, or ``None``."""
    from .plugin import get_impl

    return get_impl(request.config).judge(request.node)


@pytest.fixture
def disseqt_limiter(disseqt_config: Any) -> Any:
    from ..runner.limiter import JudgeLimiter

    return JudgeLimiter(disseqt_config.judge_concurrency or 0, disseqt_config.judge_rps)


@pytest.fixture
def disseqt_orchestrator() -> Any:
    from ..harness import build_test_orchestrator

    return build_test_orchestrator


def _apply_marker(request: pytest.FixtureRequest, spec: Any, config: Any) -> tuple[Any, Any]:
    import dataclasses

    marker = request.node.get_closest_marker("agent")
    if marker is None:
        return spec, config
    kw = marker.kwargs
    case = spec.case
    changes: dict[str, Any] = {}
    if kw.get("repeats") is not None:
        changes["repeats"] = kw["repeats"]
    if kw.get("k") is not None:
        changes["pass_k"] = kw["k"]
    if kw.get("threshold") is not None:
        changes["pass_k_threshold"] = kw["threshold"]
    if kw.get("tags") is not None:
        changes["tags"] = tuple(kw["tags"])
    if kw.get("use_judge") is not None:
        changes["use_judge"] = (
            tuple(kw["use_judge"])
            if not isinstance(kw["use_judge"], bool)
            else (("judge",) if kw["use_judge"] else ())
        )
    if changes:
        case = dataclasses.replace(case, **changes)
    run = spec.run
    if kw.get("timeout_s") is not None:
        run = dataclasses.replace(run, timeout_s=kw["timeout_s"])
    if kw.get("early_stop") is not None:
        config = dataclasses.replace(config, early_stop=bool(kw["early_stop"]))
    return dataclasses.replace(spec, case=case, run=run), config


def _deadline(request: pytest.FixtureRequest) -> float | None:
    from .plugin import get_impl

    return get_impl(request.config).suite_deadline()


def _raise_for(result: Any, config: Any, request: pytest.FixtureRequest) -> None:
    __tracebackhide__ = True
    from ..runner.models import CUT_OFF_REASONS
    from .nodes import AgentCaseFailed, AgentCaseIncomplete, properties_for
    from .plugin import get_impl

    category = {"incomplete": "incomplete", "error": "error"}.get(result.status)
    request.node.user_properties.extend(
        properties_for(result, category)
    )  # the controller aggregates it
    if result.status == "passed":
        return
    if result.status == "xfailed":
        pytest.xfail(result.reason)
    if result.status == "skipped":
        pytest.skip(result.reason)
    if result.status == "incomplete":
        if config.incomplete_policy == "skip" and result.reason_code not in CUT_OFF_REASONS:
            pytest.skip(result.reason)
        raise AgentCaseIncomplete(result.reason, result, "incomplete")
    text = result.reason
    if result.repeat_report is not None:
        from ..runner.render import case_failure_text

        text = case_failure_text(
            result, ascii_only=get_impl(request.config).ascii_only(), workers=config.max_workers
        )
    raise AgentCaseFailed(text, result, "error" if result.status == "error" else "failed")


@pytest.fixture
def disseqt_run_case(
    request: pytest.FixtureRequest, disseqt_config: Any, disseqt_judge: Any
) -> Any:
    """``disseqt_run_case(case, agent_factory=...)`` runs a case, raises on failure (or skips,
    per policy) and returns the ``CaseResult``."""

    def run(case: Any, *, agent_factory: Any = None, **kwargs: Any) -> Any:
        __tracebackhide__ = True
        from ..runner.engine import run_case
        from ..runner.models import CaseSpec
        from .plugin import cut_off_is_incomplete, timed_out_result

        spec = case if isinstance(case, CaseSpec) else CaseSpec(case)
        spec, config = _apply_marker(request, spec, kwargs.pop("config", disseqt_config))
        deadline = kwargs.setdefault("suite_deadline", _deadline(request))
        if deadline is not None and time.monotonic() > deadline:
            result = timed_out_result(spec, config)
        else:
            result = cut_off_is_incomplete(
                run_case(
                    spec,
                    agent_factory=agent_factory,
                    config=config,
                    judge=kwargs.pop("judge", disseqt_judge),
                    **kwargs,
                )
            )
        _raise_for(result, config, request)
        return result

    return run


@pytest.fixture
def disseqt_arun_case(
    request: pytest.FixtureRequest, disseqt_config: Any, disseqt_judge: Any
) -> Any:
    """The async twin of ``disseqt_run_case`` (the plugin does not depend on pytest-asyncio)."""

    async def run(case: Any, *, agent_factory: Any = None, **kwargs: Any) -> Any:
        __tracebackhide__ = True
        from ..runner.aengine import arun_case
        from ..runner.models import CaseSpec
        from .plugin import cut_off_is_incomplete, timed_out_result

        spec = case if isinstance(case, CaseSpec) else CaseSpec(case)
        spec, config = _apply_marker(request, spec, kwargs.pop("config", disseqt_config))
        deadline = kwargs.setdefault("suite_deadline", _deadline(request))
        if deadline is not None and time.monotonic() > deadline:
            result = timed_out_result(spec, config)
        else:
            result = cut_off_is_incomplete(
                await arun_case(
                    spec,
                    agent_factory=agent_factory,
                    config=config,
                    judge=kwargs.pop("judge", disseqt_judge),
                    **kwargs,
                )
            )
        _raise_for(result, config, request)
        return result

    return run
