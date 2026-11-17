"""``disseqt test``: run agent case files with or without pytest and exit meaningfully.

Exit codes: 0 passed, 1 failed, 2 incomplete (including a timeout or cancellation), 3
configuration, usage or collection problem, 4 internal error, 5 no cases ran, 130 interrupted.
Both modes produce the same ``SuiteResult`` and exit code.
"""

import argparse
import contextlib
import importlib.metadata
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

from .exit import exit_code_for

USAGE_EXIT = 3  # argparse's own 2 would collide with "incomplete"

EPILOG = """\
exit codes: 0 passed; 1 failed; 2 incomplete (undecidable, timed out or cancelled); 3 usage,
configuration or collection problem; 4 internal error; 5 no cases ran (none selected, or every
case skipped); 130 interrupted.

Arguments after a bare '--' go to pytest unchanged (disseqt test cases -- -n 2).

Trust: a case file with `run: {agent: module:attr}` imports and executes that code, and
conftest.py files run too. Treat case files like source code and only run ones you trust.
A relative `dataset:` path must stay inside the case file's directory tree.
"""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(USAGE_EXIT, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="disseqt test",
        description="Run agent test cases (YAML/JSON case files).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("paths", nargs="*", default=["."], help="case files or directories (default: .)")
    p.add_argument("--repeats", type=int, help="force every case to run N repeats")
    p.add_argument("--workers", type=int, help="threads running repeats inside one case")
    p.add_argument("--case", action="append", default=[], help="id glob (repeat or comma-separate)")
    p.add_argument(
        "--tags",
        action="append",
        default=[],
        help="tag expression, e.g. 'smoke,fast' or '!slow' (repeat to AND)",
    )
    p.add_argument("--incomplete", choices=("fail", "skip"), help="what an undecidable case does")
    p.add_argument("--seed", help="suite seed, or 'random'")
    p.add_argument(
        "--early-stop", action="store_true", help="stop a case once its verdict is decided"
    )
    p.add_argument("--run-timeout", type=float, help="per-run timeout in seconds")
    p.add_argument("--case-timeout", type=float)
    p.add_argument("--suite-timeout", type=float)
    p.add_argument("--agent", help="default agent factory, module:attr")
    p.add_argument("--output", help="write the SuiteResult JSON here")
    p.add_argument("--only-repeat", type=int, help="run only repeat I of each selected case")
    p.add_argument("--run-label")
    p.add_argument("--require-judge", action="store_true")
    p.add_argument("--ascii", action="store_true")
    p.add_argument(
        "--no-pytest", action="store_true", help="run without pytest (no conftest, no xdist)"
    )
    return p


def _plugin_autoloaded() -> bool:
    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        return False
    try:
        return any(e.name == "disseqt" for e in importlib.metadata.entry_points(group="pytest11"))
    except Exception:
        return False


def pytest_args(ns: argparse.Namespace, output: str) -> list[str]:
    args = list(ns.paths)
    if not _plugin_autoloaded():
        args += ["-p", "disseqt_agent_testing.pytest_plugin"]
    for flag, value in (
        ("--disseqt-repeats", ns.repeats), ("--disseqt-workers", ns.workers), ("--disseqt-incomplete", ns.incomplete),
        ("--disseqt-seed", ns.seed), ("--disseqt-run-timeout", ns.run_timeout), ("--disseqt-case-timeout", ns.case_timeout),
        ("--disseqt-suite-timeout", ns.suite_timeout), ("--disseqt-agent", ns.agent),
        ("--disseqt-only-repeat", ns.only_repeat), ("--disseqt-run-label", ns.run_label),
    ):  # fmt: skip
        if value is not None:
            args += [flag, str(value)]
    for c in ns.case:
        args += ["--disseqt-case", c]
    for t in ns.tags:
        args += ["--disseqt-tags", t]
    for flag, on in (
        ("--disseqt-early-stop", ns.early_stop),
        ("--disseqt-require-judge", ns.require_judge),
        ("--disseqt-ascii", ns.ascii),
    ):
        if on:
            args.append(flag)
    args += ["--disseqt-output", output]
    extra = [a for a in ns.passthrough if a != "--"]
    return args + extra


class _Probe:
    """A pytest plugin for the CLI: sees a real keyboard interrupt (pytest also exits 2 for a
    collection error) and puts the rootdir on ``sys.path``, as ``python -m pytest`` would."""

    def __init__(self) -> None:
        self.interrupted = False

    def pytest_keyboard_interrupt(self, excinfo: Any) -> None:
        import pytest

        # pytest raises Session.Interrupted (a KeyboardInterrupt) for collection errors too
        if not isinstance(excinfo.value, pytest.Session.Interrupted):
            self.interrupted = True

    def pytest_configure(self, config: Any) -> None:
        root = str(config.rootpath)
        if root not in sys.path:
            sys.path.insert(0, root)


@contextlib.contextmanager
def _cwd_on_path() -> Iterator[None]:
    """Console scripts do not put the working directory on ``sys.path`` (``python -m`` does), so
    ``--agent agents.refund:make`` would not import. Restored afterwards."""
    saved = list(sys.path)
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        yield
    finally:
        sys.path[:] = saved


def _write_failed(path: str, exc: OSError) -> int:
    print(f"disseqt: cannot write --output '{path}': {exc.strerror or exc}", file=sys.stderr)
    return 3


def run_pytest_mode(ns: argparse.Namespace) -> int:
    import json

    import pytest

    from ..runner.models import SuiteResult

    dest = Path(ns.output).absolute() if ns.output else None
    if dest is not None and dest.is_file():
        with contextlib.suppress(OSError):
            dest.unlink()  # a stale file must never be mistaken for this run's result
    tmp = Path(tempfile.mkdtemp(prefix="disseqt-"))
    out = tmp / "suite.json"
    probe = _Probe()
    try:
        code = int(pytest.main(pytest_args(ns, str(out)), plugins=[probe]))
        result: Any = None
        if out.is_file():
            result = SuiteResult.from_dict(json.loads(out.read_text(encoding="utf-8")))
            if dest is not None:
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(out, dest)
                except OSError as exc:
                    return _write_failed(str(ns.output), exc)
        return exit_code_for(result, pytest_code=code, interrupted=probe.interrupted)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_direct_mode(ns: argparse.Namespace) -> int:
    from ..runner import CaseFilter, SuiteConfig, SuiteConfigError, load_cases, run_suite
    from ..runner.loader import format_errors
    from ..runner.render import can_unicode, case_failure_text, summary_table

    seed: int = 0
    if ns.seed is not None:
        import secrets

        try:
            seed = secrets.randbits(32) if ns.seed == "random" else int(ns.seed)
        except ValueError:
            print(
                f"disseqt: --seed must be an integer or 'random', got '{ns.seed}'", file=sys.stderr
            )
            return 3
    kwargs: dict[str, Any] = {"seed": seed, "incomplete_policy": ns.incomplete or "fail", "early_stop": ns.early_stop,
                              "judge_missing": "fail" if ns.require_judge else "skip", "run_label": ns.run_label or "",
                              "only_repeat": ns.only_repeat}  # fmt: skip
    for field, value in (("repeats", ns.repeats), ("max_workers", ns.workers), ("run_timeout_s", ns.run_timeout),
                         ("case_timeout_s", ns.case_timeout), ("suite_timeout_s", ns.suite_timeout)):  # fmt: skip
        if value is not None:
            kwargs[field] = value
    try:
        config = SuiteConfig(**kwargs)
    except SuiteConfigError as exc:
        print(f"disseqt: {exc}", file=sys.stderr)
        return 3
    factory = None
    if ns.agent:
        from ..runner.factory import FactoryImportError, import_factory

        try:
            factory = import_factory(ns.agent)
        except FactoryImportError as exc:
            print(
                f"disseqt: cannot import agent factory '{exc.path}': {exc.exc_type}: {exc.detail}",
                file=sys.stderr,
            )
            return 3
    flt = CaseFilter.build(ids=ns.case, tags=ns.tags)
    loaded = load_cases(list(ns.paths), filter=flt)
    for line in format_errors([e for e in loaded.errors if e.code != "DSQ-W10"]):
        print(line, file=sys.stderr)
    try:
        result = run_suite(loaded, agent_factory=factory, config=config)
    except KeyboardInterrupt as exc:
        partial = getattr(exc, "suite_result", None)
        if partial is not None and ns.output:
            with contextlib.suppress(OSError):
                partial.write(ns.output)
        return 130
    except SuiteConfigError as exc:
        print(f"disseqt: {exc}", file=sys.stderr)
        return 3
    ascii_only = ns.ascii or not can_unicode(sys.stdout)
    for case in result.cases:
        if case.status in ("failed", "incomplete", "error"):
            print(
                case_failure_text(
                    case,
                    ascii_only=ascii_only,
                    workers=config.max_workers,
                    early_stop=config.early_stop,
                ),
                end="\n\n",
            )
    for line in summary_table(result, ascii_only):
        print(line)
    if ns.output:
        try:
            Path(ns.output).absolute().parent.mkdir(parents=True, exist_ok=True)
            result.write(ns.output)
        except OSError as exc:
            return _write_failed(ns.output, exc)
        print(f"disseqt: wrote {ns.output}")
    return exit_code_for(result)


def run_test_command(argv: list[str]) -> int:
    """Arguments after a bare ``--`` are passed to pytest unchanged (``-- -n 2``)."""
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1 :]
    else:
        passthrough = []
    try:
        ns = build_parser().parse_args(argv)
    except SystemExit as exc:  # --help (0) or a usage error (3)
        return exc.code if isinstance(exc.code, int) else 0
    ns.passthrough = passthrough
    with _cwd_on_path():
        return run_direct_mode(ns) if ns.no_pytest else run_pytest_mode(ns)
