"""Collection nodes and the exceptions that carry a case result through pytest."""

import json
from pathlib import Path
from typing import Any

import pytest

from .compat import node_path


class CaseFileError(Exception):
    """A case file could not be loaded; ``records`` are the problems (shown without a traceback)."""

    def __init__(self, records: list[Any]):
        super().__init__(f"{len(records)} problem(s) in the case file")
        self.records = records


class AgentSetupError(Exception):
    """Setup-time problem (no factory, bad import, harness configuration): a pytest ERROR."""


class AgentCaseFailed(AssertionError):
    """A case failed (or errored). ``.result`` is the ``CaseResult``; picklable for xdist."""

    def __init__(self, message: str, result: Any = None, category: str = "failed"):
        super().__init__(message)
        self.result = result
        self.category = category

    def __reduce__(self) -> tuple[Any, ...]:
        data = self.result.to_dict() if self.result is not None else None
        return (_rebuild, (type(self), str(self), data, self.category))


class AgentCaseIncomplete(AgentCaseFailed):
    """Nothing failed, but the verdict cannot be decided (shown as INCOMPLETE)."""


def _rebuild(cls: type, message: str, data: Any, category: str) -> AgentCaseFailed:
    result = None
    if data is not None:
        from ..runner.models import CaseResult

        result = CaseResult.from_dict(data)
    return cls(message, result, category)  # type: ignore[no-any-return]


CATEGORY_KEY = "disseqt.category"
RESULT_KEY = "disseqt.case_result"
XFAIL_KEY = "disseqt.xfail"


class AgentCaseFile(pytest.File):
    """A ``*.agent.yaml|yml|json`` file: one item per case (or per repeat in split mode)."""

    def collect(self) -> Any:
        from ..runner.errors import format_code
        from ..runner.loader import load_file
        from .plugin import get_impl

        impl = get_impl(self.config)
        path = node_path(self)
        try:
            shown = str(path.relative_to(Path(str(self.config.rootpath))))
        except ValueError:
            shown = str(path)
        specs, errors = load_file(path, impl.state.registry, display=shown)
        for e in errors:
            if e.code == "DSQ-W10":
                impl.warn(format_code("DSQ-W10", path=shown))
        real = [e for e in errors if e.code != "DSQ-W10"]
        if real:
            impl.state.collection_errors.extend(real)
            raise CaseFileError(real)
        yield from impl.make_items(self, specs)

    def repr_failure(self, excinfo: Any, style: Any = None) -> Any:
        if isinstance(excinfo.value, CaseFileError):
            from ..runner.loader import format_errors

            return "\n".join(format_errors(excinfo.value.records))
        return super().repr_failure(excinfo)


class AgentCaseItem(pytest.Item):
    """One agent case. The repeats run inside ``runtest`` (so pass^k is atomic in one item)."""

    def __init__(self, *, spec: Any, repeat: int | None = None, **kw: Any):
        super().__init__(**kw)
        self.spec = spec
        self.repeat = repeat
        self._factory: Any = None

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        from .plugin import get_impl

        get_impl(self.config).setup_item(self)

    def reportinfo(self) -> tuple[Any, int | None, str]:
        line = self.spec.source.line
        return node_path(self), (line - 1) if line else None, self.name

    # ------------------------------------------------------------------ run
    def runtest(self) -> None:
        from .plugin import get_impl

        get_impl(self.config).run_item(self)

    def repr_failure(self, excinfo: Any, style: Any = None) -> Any:
        exc = excinfo.value
        if isinstance(exc, AgentCaseFailed) and exc.result is not None:
            from .plugin import get_impl

            return get_impl(self.config).failure_text(self, exc.result)
        if isinstance(exc, AgentSetupError):
            return str(exc)
        return super().repr_failure(excinfo)


def properties_for(
    result: Any, category: str | None = None, *, xfail: str | None = None
) -> list[tuple[str, Any]]:
    """Compact JUnit properties plus the large transient ``disseqt.case_result`` the controller
    pops before the JUnit writer sees it. ``xfail`` is the case's expected-failure reason: the
    controller needs it to settle a case it merges from split repeats."""
    rr = result.repeat_report
    compact: dict[str, Any] = {
        "disseqt.case_id": result.case_id, "disseqt.status": result.status, "disseqt.n": result.n_planned,
        "disseqt.k": result.k, "disseqt.threshold": result.threshold,
    }  # fmt: skip
    if rr is not None:
        compact.update(
            {
                "disseqt.passed": rr.n_passed, "disseqt.failed": rr.n_failed, "disseqt.incomplete": rr.n_incomplete,
                "disseqt.pass_at_1": rr.pass_at_1, "disseqt.pass_hat_k": rr.pass_hat_k,
                "disseqt.wilson_low": rr.wilson_95[0] if rr.wilson_95 else None,
                "disseqt.wilson_high": rr.wilson_95[1] if rr.wilson_95 else None,
            }
        )  # fmt: skip
    if result.repeats:
        compact["disseqt.seed"] = result.repeats[0].seed
    props: list[tuple[str, Any]] = [(k, v) for k, v in compact.items() if v is not None]
    if category:
        props.append((CATEGORY_KEY, category))
    if xfail:
        props.append((XFAIL_KEY, xfail))
    props.append((RESULT_KEY, json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)))
    return props
