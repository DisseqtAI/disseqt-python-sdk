"""The plugin's hook implementations.

Module-level hooks stay tiny (they run in every pytest process where the SDK is installed). The
real work lives in ``DisseqtImpl``, registered in ``pytest_configure``, and imports the runner
lazily, so a run with no agent cases pays for nothing and prints nothing.
"""

import dataclasses
import json
import os
import re
import secrets
import sys
import time
import warnings
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import pytest

from .compat import has_xdist, is_worker, node_path
from .nodes import (
    CATEGORY_KEY,
    RESULT_KEY,
    AgentCaseFailed,
    AgentCaseFile,
    AgentCaseIncomplete,
    AgentCaseItem,
    AgentSetupError,
    properties_for,
)
from .options import (
    add_options,
    any_option_set,
    explicitly_configured,
    looks_like_case_file,
    resolve_options,
)

IMPL_KEY = pytest.StashKey[Any]()
DEFERRED_KEY = pytest.StashKey[pytest.UsageError]()  # a bad option value, reported only on use
DEFAULT_GLOBS = ["*.agent.yaml", "*.agent.yml", "*.agent.json"]
DEFERRED_USED_KEY = pytest.StashKey[bool]()
MARKERS = (
    "agent(repeats=None, k=None, threshold=None, tags=None, timeout_s=None, use_judge=None, early_stop=None): "
    "run a Python test as an agent case with these repeat settings",
    "disseqt_repeat: reserved by the disseqt plugin",
)


class DisseqtPytestWarning(pytest.PytestWarning):
    """A warning from the disseqt plugin."""


@dataclass
class State:
    options: dict[str, Any]
    seed: int
    seed_random: bool
    registry: dict[str, str] = field(default_factory=dict)
    collection_errors: list[Any] = field(default_factory=list)
    items: dict[str, Any] = field(default_factory=dict)  # nodeid -> CaseSpec
    results: dict[str, Any] = field(default_factory=dict)  # nodeid -> CaseResult
    suite_result: Any = None
    t0: float = field(default_factory=time.monotonic)
    warned: set[str] = field(default_factory=set)
    keyboard_interrupt: bool = False


def get_impl(config: pytest.Config) -> "DisseqtImpl":
    if IMPL_KEY not in config.stash and DEFERRED_KEY in config.stash:
        raise config.stash[DEFERRED_KEY]  # the plugin is being used: now the bad value matters
    impl: DisseqtImpl = config.stash[IMPL_KEY]
    return impl


# --------------------------------------------------------------------------- module-level hooks
def pytest_addhooks(pluginmanager: Any) -> None:
    from . import hookspecs

    pluginmanager.add_hookspecs(hookspecs)


def pytest_addoption(parser: Any) -> None:
    add_options(parser)


def pytest_configure(config: pytest.Config) -> None:
    if IMPL_KEY in config.stash:
        return
    try:
        options = resolve_options(config)  # DSQ-U01 -> pytest.UsageError
    except pytest.UsageError as exc:
        if explicitly_configured(config):
            raise
        config.stash[DEFERRED_KEY] = exc  # e.g. a stray DISSEQT_SEED must not break other tests
        return
    for line in MARKERS:
        config.addinivalue_line("markers", line)
    seed_opt = options["seed"]
    random_seed = seed_opt == "random"
    if random_seed:
        workerinput = getattr(config, "workerinput", None)
        seed = (
            int(workerinput["disseqt_seed"])
            if workerinput and "disseqt_seed" in workerinput
            else secrets.randbits(32)
        )
    else:
        seed = int(seed_opt)
    if options["early_stop"] and options["split_repeats"]:
        from ..runner.errors import format_code

        raise pytest.UsageError(format_code("DSQ-U03"))
    impl = DisseqtImpl(config, State(options, seed, random_seed))
    config.stash[IMPL_KEY] = impl
    config.pluginmanager.register(impl, "disseqt-impl")
    if has_xdist(config):
        from .xdist_support import XdistSupport

        config.pluginmanager.register(XdistSupport(impl), "disseqt-xdist")


def pytest_collection_modifyitems(config: pytest.Config) -> None:
    if config.stash.get(DEFERRED_USED_KEY, False):
        raise config.stash[DEFERRED_KEY]


def pytest_collect_file(file_path: Path, parent: Any) -> Any:
    config = parent.config
    if IMPL_KEY not in config.stash and DEFERRED_KEY not in config.stash:
        return None
    globs = (
        config.stash[IMPL_KEY].state.options["case_globs"]
        if IMPL_KEY in config.stash
        else DEFAULT_GLOBS
    )
    if not any(fnmatchcase(file_path.name, g) for g in globs):
        return None
    # Inert unless this is plausibly ours: an unrelated ``support.agent.json`` in a repo that merely
    # has the SDK installed must not break its test run.
    if not explicitly_configured(config) and not looks_like_case_file(file_path):
        return None
    if IMPL_KEY not in config.stash:
        # A bad option value was deferred and the plugin is now in use. A UsageError raised from
        # here would only become a collection error, so ``pytest_collection_modifyitems`` raises it.
        config.stash[DEFERRED_USED_KEY] = True
        return None
    return AgentCaseFile.from_parent(parent, path=file_path)


# --------------------------------------------------------------------------- the implementation
class DisseqtImpl:
    def __init__(self, config: pytest.Config, state: State):
        self.config = config
        self.state = state
        self._cfg: Any = None
        self._judge: Any = None
        self._judge_done = False

    # ------------------------------------------------------------------ config and helpers
    def warn(self, text: str) -> None:
        warnings.warn(text, DisseqtPytestWarning, stacklevel=2)

    def warn_once(self, key: str, text: str) -> None:
        if key not in self.state.warned:
            self.state.warned.add(key)
            self.warn(text)

    def workers_total(self) -> int:
        try:
            return max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1") or 1))
        except ValueError:
            return 1

    def suite_config(self) -> Any:
        if self._cfg is not None:
            return self._cfg
        from ..runner.config import SuiteConfig, SuiteConfigError
        from ..runner.errors import format_code

        o = self.state.options
        w = self.workers_total()
        jc = o["judge_concurrency"]
        if jc and w > jc:
            self.warn_once("w11", format_code("DSQ-W11", limit=jc, workers=w))
        if jc:
            jc = max(1, jc // w)
        rps = o["judge_rps"] / w if o["judge_rps"] else None
        try:
            self._cfg = SuiteConfig(
                repeats=o["repeats"], default_repeats=o["default_repeats"], max_workers=max(1, o["workers"] // w),
                run_timeout_s=o["run_timeout"], case_timeout_s=o["case_timeout"], suite_timeout_s=o["suite_timeout"],
                early_stop=bool(o["early_stop"]), seed=self.state.seed, incomplete_policy=o["incomplete"],
                judge_missing="fail" if o["require_judge"] else "skip", judge_concurrency=jc, judge_rps=rps,
                repeat_mode="split" if o["split_repeats"] else "internal", only_repeat=o["only_repeat"],
                run_label=o["run_label"] or "",
            )  # fmt: skip
        except SuiteConfigError as exc:
            raise pytest.UsageError(str(exc)) from None
        return self._cfg

    def ascii_only(self) -> bool:
        from ..runner.render import can_unicode

        explicit = self.state.options["ascii"]
        return bool(explicit) if explicit is not None else not can_unicode(sys.stdout)

    def judge(self, node: Any) -> Any:
        if not self._judge_done:
            self._judge_done = True
            self._judge = node.ihook.pytest_disseqt_judge(config=self.config)
        return self._judge

    # ------------------------------------------------------------------ collection
    def make_items(self, file_node: Any, specs: list[Any]) -> list[Any]:
        from ..runner.engine import resolve_n_k

        cfg = self.suite_config()
        items: list[Any] = []
        for spec in specs:
            repeats: list[int | None] = [None]
            if cfg.repeat_mode == "split":
                n, _, _ = resolve_n_k(spec, cfg)
                repeats = list(range(n))
            for r in repeats:
                name = spec.case.id if r is None else f"{spec.case.id}[r{r}]"
                item = AgentCaseItem.from_parent(file_node, name=name, spec=spec, repeat=r)
                self._mark(item, spec, split=r is not None)
                self.state.items[item.nodeid] = spec
                items.append(item)
        return items

    def _mark(self, item: Any, spec: Any, *, split: bool) -> None:
        item.add_marker(pytest.mark.agent)
        for tag in spec.case.tags:
            self.config.addinivalue_line(
                "markers", f"{tag}: agent case tag"
            )  # keeps --strict-markers green
            item.add_marker(getattr(pytest.mark, tag))
        if spec.run.skip:
            item.add_marker(pytest.mark.skip(reason=spec.run.skip))
        if spec.run.xfail:
            item.add_marker(pytest.mark.xfail(reason=spec.run.xfail, strict=True))
        if split:
            item.add_marker(pytest.mark.xdist_group(name=spec.case.id))

    def _case_filter(self) -> Any:
        from ..runner.filter import CaseFilter

        o = self.state.options
        return CaseFilter.build(ids=o["case"] or (), tags=o["tags"] or ())

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session: Any, config: Any, items: list[Any]) -> None:
        agent = [i for i in items if isinstance(i, AgentCaseItem)]
        if not agent:
            return
        flt = self._case_filter()
        if flt.active:
            gone = [i for i in agent if not flt.matches(i.spec.case.id, i.spec.case.tags)]
            if gone:
                config.hook.pytest_deselected(items=gone)
                items[:] = [i for i in items if i not in gone]
                agent = [i for i in agent if i not in gone]
        pm = config.pluginmanager
        from ..runner.errors import format_code

        if pm.hasplugin("rerunfailures"):
            self.warn_once("w01", format_code("DSQ-W01"))
            for i in agent:
                i.add_marker(pytest.mark.flaky(reruns=0))
        if pm.hasplugin("flaky"):
            self.warn_once("w02", format_code("DSQ-W02"))
        if pm.hasplugin("repeat"):
            self.warn_once("w03", format_code("DSQ-W03"))
        cfg = self.suite_config()
        if (
            cfg.repeat_mode == "split"
            and has_xdist(config)
            and getattr(config.option, "dist", "no") not in ("loadgroup", "no")
        ):
            self.warn_once("w06", format_code("DSQ-W06"))
        limit = _pytest_timeout(config)
        if limit:
            import math

            from ..runner.engine import resolve_n_k

            for i in agent:
                n, _, _ = resolve_n_k(i.spec, cfg)
                need = math.ceil(n / max(1, cfg.max_workers)) * cfg.run_timeout_s
                if need > limit:
                    self.warn_once(
                        f"w05:{i.spec.case.id}",
                        format_code("DSQ-W05", id=i.spec.case.id, need=need, limit=limit),
                    )

    def pytest_report_header(self, config: Any) -> Any:
        if not any_option_set(config):
            return None
        o = self.state.options
        return f"disseqt: seed={self.state.seed}{' (random)' if self.state.seed_random else ''} workers={o['workers']}"

    def pytest_report_collectionfinish(self, config: Any, items: list[Any]) -> Any:
        agent = [i for i in items if isinstance(i, AgentCaseItem)]
        if not agent:
            return None
        cases = len({i.spec.case.id for i in agent})
        seed = f"seed={self.state.seed}" + (" (random)" if self.state.seed_random else "")
        return f"disseqt: {cases} agent cases, {seed}"

    # ------------------------------------------------------------------ setup and run
    def setup_item(self, item: AgentCaseItem) -> None:
        from ..harness import FakeLLM, HarnessConfigError, build_test_orchestrator
        from ..models import Message, from_dict
        from ..runner.errors import format_code
        from ..runner.factory import FactoryImportError, import_factory, takes_seed

        spec, cfg = item.spec, self.suite_config()
        cid = spec.case.id
        factory: Any = None
        if spec.run.agent:
            try:
                factory = import_factory(spec.run.agent)
            except FactoryImportError as exc:
                raise AgentSetupError(
                    format_code("DSQ-E02", path=exc.path, id=cid, type=exc.exc_type, msg=exc.detail)
                ) from None
        else:
            factory = item.ihook.pytest_disseqt_agent_factory(case=spec.case, config=self.config)
            default = self.state.options["agent"]
            if factory is None and default:
                try:
                    factory = import_factory(default)
                except FactoryImportError as exc:
                    raise AgentSetupError(
                        format_code(
                            "DSQ-E02", path=exc.path, id=cid, type=exc.exc_type, msg=exc.detail
                        )
                    ) from None
        if factory is None:
            raise AgentSetupError(format_code("DSQ-E01", id=cid))
        if spec.case.use_judge and self.judge(item) is None and cfg.judge_missing == "fail":
            raise AgentSetupError(format_code("DSQ-E05", id=cid))
        if not takes_seed(factory):
            options: dict[str, Any] = {}
            if spec.run.llm is not None:
                script = tuple(
                    x if isinstance(x, str) else from_dict(Message, x) for x in spec.run.llm
                )
                options["llm"] = FakeLLM(script=script)
            try:
                build_test_orchestrator(factory, spec.case.fakes, **options)
            except HarnessConfigError as exc:
                raise AgentSetupError(format_code("DSQ-E03", id=cid, msg=str(exc))) from None
        item._factory = factory

    def run_item(self, item: AgentCaseItem) -> None:
        from ..runner.engine import run_case

        cfg = self.suite_config()
        if item.repeat is not None:
            cfg = dataclasses.replace(cfg, only_repeat=item.repeat, keep_reports="all")
        result = run_case(
            item.spec, agent_factory=item._factory, config=cfg, judge=self.judge(item)
        )
        self._finish(item, result)

    def _finish(self, item: AgentCaseItem, result: Any) -> None:
        cfg = self.suite_config()
        status = result.status
        if item.repeat is not None and result.repeats:
            # Split mode: this item stands for one repeat, so it passes or fails on that repeat
            # alone (the controller still aggregates one case verdict from all of them).
            own = next((r for r in result.repeats if r.index == item.repeat), None)
            if own is not None and status not in ("error", "skipped"):
                status = "failed" if own.status == "failed" else own.status
                if own.status == "not_run":
                    status = "skipped"
        category = {"incomplete": "incomplete", "error": "error"}.get(status)
        item.user_properties.extend(properties_for(result, category))
        if status == "passed":
            return
        if status == "skipped":
            pytest.skip(result.reason or "skipped")
        if status == "incomplete":
            if cfg.incomplete_policy == "skip":
                pytest.skip(result.reason)
            raise AgentCaseIncomplete(result.reason, result, "incomplete")
        raise AgentCaseFailed(
            result.reason or f"agent case '{result.case_id}' {status}", result, category or "failed"
        )

    def failure_text(self, item: AgentCaseItem, result: Any) -> str:
        from ..runner.render import case_failure_text

        cfg = self.suite_config()
        replay = f"pytest '{item.nodeid.split('[')[0]}' --disseqt-only-repeat {{index}} --disseqt-seed {self.state.seed}"
        return case_failure_text(
            result, ascii_only=self.ascii_only(), workers=cfg.max_workers, early_stop=cfg.early_stop,
            max_lines=cfg.eval_config.max_report_lines, replay=replay,
        )  # fmt: skip

    # ------------------------------------------------------------------ reporting
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report: Any) -> None:
        props = list(report.user_properties)
        entries = [p for p in props if p[0] == RESULT_KEY]
        if entries:
            if not is_worker(self.config):
                report.user_properties[:] = [
                    p for p in props if p[0] != RESULT_KEY
                ]  # before the JUnit writer
            from ..runner.models import CaseResult

            for n, entry in enumerate(entries):
                result = CaseResult.from_dict(json.loads(entry[1]))
                self._store(report.nodeid if n == 0 else f"{report.nodeid}#{n}", result)
            return
        spec = self.state.items.get(report.nodeid)
        if spec is None:  # under xdist the controller does not collect: fall back to the node id
            if ".agent." not in report.nodeid.split("::", 1)[0]:
                return
            spec = _spec_from_nodeid(report.nodeid)
        result = self._synthesize(report, spec)
        if result is not None:
            self._store(report.nodeid, result)

    def _store(self, nodeid: str, result: Any) -> None:
        self.state.results[nodeid] = result
        if not is_worker(self.config):
            self.config.hook.pytest_disseqt_case_result(result=result)

    def _synthesize(self, report: Any, spec: Any) -> Any:
        """A ``CaseResult`` for an item that never produced one: skipped by a marker, a setup
        error, or a crashed xdist worker."""
        from ..runner.models import CaseResult

        text = str(getattr(report, "longreprtext", "") or "")
        if report.failed and "crashed while running" in text:  # xdist reports it with when="???"
            return CaseResult(
                spec.case.id, "error", source=spec.source, tags=spec.case.tags, reason_code="worker_crashed",
                reason=f"worker crashed while running '{report.nodeid}'", n_planned=spec.case.repeats,
            )  # fmt: skip
        setup_bad = report.when == "setup" and (report.failed or report.skipped)
        call_bad = report.when == "call" and report.failed
        if not (setup_bad or call_bad):
            return None
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        if report.skipped:
            lr = getattr(report, "longrepr", None)
            reason = (
                lr[2].removeprefix("Skipped: ") if isinstance(lr, tuple) and len(lr) == 3 else first
            )
            return CaseResult(
                spec.case.id,
                "skipped",
                source=spec.source,
                tags=spec.case.tags,
                reason_code="skipped",
                reason=reason,
                n_planned=spec.case.repeats,
            )
        if "crashed" in text and report.when == "call":
            code, reason = "worker_crashed", f"worker crashed while running '{report.nodeid}'"
        else:
            code = _setup_code(text)
            reason = first
        return CaseResult(
            spec.case.id,
            "error",
            source=spec.source,
            tags=spec.case.tags,
            reason_code=code,
            reason=reason,
            n_planned=spec.case.repeats,
        )

    def pytest_report_teststatus(self, report: Any, config: Any) -> Any:
        if report.when != "call":
            return None
        category = dict(report.user_properties).get(CATEGORY_KEY)
        if category == "incomplete":
            return (
                ("incomplete", "I", "INCOMPLETE")
                if report.skipped
                else ("failed", "I", "INCOMPLETE")
            )
        if category == "error" and report.failed:
            return ("error", "E", "ERROR")
        return None

    def pytest_sessionfinish(self, session: Any, exitstatus: Any) -> None:
        if is_worker(self.config) or not (self.state.results or self.state.collection_errors):
            return
        from ..runner.engine import SuiteState, build_result

        cfg = self.suite_config()
        cases = self._cases_for_suite()
        suite = SuiteState(cfg, None)
        suite.interrupted = self.state.keyboard_interrupt
        result = build_result(
            cases, tuple(self.state.collection_errors), cfg, suite, self.state.t0, 0
        )
        self.state.suite_result = result
        out = self.state.options["output"]
        if out:
            result.write(out)
        self.config.hook.pytest_disseqt_suite_result(result=result)

    def _cases_for_suite(self) -> list[Any]:
        """One ``CaseResult`` per case. Only split-repeat parts (one repeat each) are merged; two
        whole results with the same id (two tests using one id) are a duplicate id: each stays its
        own case (later ones renamed ``id#2``) and a DSQ-C03 error is recorded, so a failure can
        never be merged away."""
        from ..runner.merge import is_split_part, merge_repeat_results
        from ..runner.models import CollectionErrorRecord

        by_case: dict[str, list[Any]] = {}
        for result in self.state.results.values():
            by_case.setdefault(result.case_id, []).append(result)
        cases: list[Any] = []
        for case_id, parts in by_case.items():
            split = [p for p in parts if is_split_part(p)]
            whole = [p for p in parts if not is_split_part(p)]
            if len(split) == 1:
                cases.append(split[0])
            elif split:
                key = lambda r: min((x.index for x in r.repeats), default=0)  # noqa: E731
                cases.append(merge_repeat_results(sorted(split, key=key)))
            cases.extend(whole[:1])
            for n, dup in enumerate(whole[1:], start=2):
                cases.append(dataclasses.replace(dup, case_id=f"{case_id}#{n}"))
            if len(whole) + (1 if split else 0) > 1:
                self.state.collection_errors.append(
                    CollectionErrorRecord(
                        None,
                        f"duplicate case id '{case_id}': {len(whole) + (1 if split else 0)} tests produced "
                        "a result for it; case ids must be unique across the run",
                        code="DSQ-C03",
                    )
                )
        return cases

    def pytest_keyboard_interrupt(self, excinfo: Any) -> None:
        # pytest raises Session.Interrupted (a KeyboardInterrupt) for collection errors too
        if not isinstance(excinfo.value, pytest.Session.Interrupted):
            self.state.keyboard_interrupt = True

    def pytest_terminal_summary(self, terminalreporter: Any, exitstatus: Any, config: Any) -> None:
        result = self.state.suite_result
        if result is None or not result.cases:
            return
        from ..runner.render import summary_table

        terminalreporter.section("disseqt agent results")
        for line in summary_table(result, self.ascii_only()):
            terminalreporter.write_line(line)
        out = self.state.options["output"]
        if out:
            terminalreporter.write_line(f"disseqt: wrote {out}")


def _spec_from_nodeid(nodeid: str) -> Any:
    """A stand-in ``CaseSpec`` (id, source path) for a node the controller did not collect."""
    from ..models import AgentTestCase
    from ..runner.models import CaseSource, CaseSpec

    path, _, name = nodeid.partition("::")
    case_id = re.sub(r"\[r\d+\]$", "", name) or "unknown"
    return CaseSpec(AgentTestCase(case_id, "-"), CaseSource(path, nodeid=nodeid))


def _pytest_timeout(config: Any) -> float | None:
    for name in ("timeout",):
        try:
            value = config.getini(name)
        except (ValueError, KeyError):
            value = None
        try:
            if value:
                return float(value)
        except (TypeError, ValueError):
            pass
    cli = getattr(config.option, "timeout", None)
    return float(cli) if cli else None


def _setup_code(text: str) -> str:
    if "has no agent factory" in text:
        return "no_agent_factory"
    if "cannot import agent factory" in text:
        return "agent_import"
    if "harness configuration error" in text:
        return "harness_config"
    if "no judge is configured" in text:
        return "judge_required"
    return "internal"


_ = (re, node_path)
