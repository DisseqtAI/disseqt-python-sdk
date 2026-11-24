"""The option table: one row per option generates the CLI flag, the ini key, the env var, the
parser and the ``SuiteConfig`` field. Precedence: CLI > env > ini > default."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

TRUE = {"1", "true", "yes", "on"}
FALSE = {"0", "false", "no", "off"}


def _bool(text: str) -> bool:
    low = text.strip().lower()
    if low in TRUE:
        return True
    if low in FALSE:
        return False
    raise ValueError("expected one of 1/true/yes/on or 0/false/no/off")


def _int_min(minimum: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        value = int(text)
        if value < minimum:
            raise ValueError(f"expected an integer >= {minimum}")
        return value

    return parse


_int = _int_min(0)
_pos_int = _int_min(1)


def _seed(text: str) -> "int | str":
    if text.strip().lower() == "random":
        return "random"
    return int(text)


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise ValueError("expected a number > 0")
    return value


def _choice(*names: str) -> Callable[[str], str]:
    def parse(text: str) -> str:
        low = text.strip().lower()
        if low not in names:
            raise ValueError(f"expected one of {', '.join(names)}")
        return low

    return parse


def _list(sep: str) -> Callable[[str], list[str]]:
    return lambda text: [p.strip() for p in text.split(sep) if p.strip()]


@dataclass(frozen=True)
class Option:
    name: str
    flags: tuple[str, ...] = ()
    ini: str | None = None
    env: str | None = None
    parse: Callable[[str], Any] = str
    expected: str = "a value"
    action: str = "store"  # store | store_true | append
    default: Any = None
    help: str = ""
    field: str | None = None  # the SuiteConfig field this feeds (None: a plugin-only option)
    ini_type: str | None = (
        None  # a typed ini declaration (int/float/bool/args) where pytest supports it
    )
    flatten: bool = False  # an append option whose every entry is itself a parsed list

    @property
    def dest(self) -> str:
        return f"disseqt_{self.name}"


OPTIONS: tuple[Option, ...] = (
    Option(
        "repeats",
        ("--disseqt-repeats",),
        None,
        "DISSEQT_REPEATS",
        _pos_int,
        "an integer >= 1",
        help="force every case to run N repeats",
        field="repeats",
    ),
    Option(
        "default_repeats",
        (),
        "disseqt_default_repeats",
        None,
        _pos_int,
        "an integer >= 1",
        default=1,
        help="repeats for cases that do not set them",
        field="default_repeats",
        ini_type="int",
    ),
    Option(
        "workers",
        ("--disseqt-workers",),
        "disseqt_workers",
        "DISSEQT_WORKERS",
        _pos_int,
        "an integer >= 1",
        default=8,
        help="threads running repeats inside one case",
        field="max_workers",
        ini_type="int",
    ),
    Option(
        "case",
        ("--disseqt-case",),
        None,
        "DISSEQT_CASE",
        _list(","),
        "comma-separated id globs",
        action="append",
        help="run only cases whose id matches a glob (repeat or comma-separate)",
        flatten=True,
    ),
    Option(
        "tags",
        ("--disseqt-tags",),
        None,
        "DISSEQT_TAGS",
        _list(";"),
        "tag expressions (';' separates entries in the env var)",
        action="append",
        help="select cases by tag expression, e.g. 'smoke,fast' or '!slow' (repeat to AND)",
    ),
    Option(
        "incomplete",
        ("--disseqt-incomplete",),
        "disseqt_incomplete",
        "DISSEQT_INCOMPLETE",
        _choice("fail", "skip"),
        "fail or skip",
        default="fail",
        help="what an undecidable case does: fail (default) or skip",
        field="incomplete_policy",
    ),
    Option(
        "require_judge",
        ("--disseqt-require-judge",),
        "disseqt_require_judge",
        "DISSEQT_REQUIRE_JUDGE",
        _bool,
        "a boolean",
        action="store_true",
        default=False,
        help="error (not skip) when a case needs a judge and none is configured",
        ini_type="bool",
    ),
    Option(
        "output",
        ("--disseqt-output",),
        "disseqt_output",
        "DISSEQT_OUTPUT",
        str,
        "a file path",
        help="write the SuiteResult JSON here",
    ),
    Option(
        "seed",
        ("--disseqt-seed",),
        "disseqt_seed",
        "DISSEQT_SEED",
        _seed,
        "an integer or 'random'",
        default=0,
        help="suite seed for per-repeat seeds ('random' draws one and prints it)",
        field="seed",
    ),
    Option(
        "early_stop",
        ("--disseqt-early-stop",),
        "disseqt_early_stop",
        "DISSEQT_EARLY_STOP",
        _bool,
        "a boolean",
        action="store_true",
        default=False,
        help="stop a case once the verdict is decided (marks it not comparable)",
        field="early_stop",
        ini_type="bool",
    ),
    Option(
        "run_timeout",
        ("--disseqt-run-timeout",),
        "disseqt_run_timeout",
        "DISSEQT_RUN_TIMEOUT",
        _positive_float,
        "seconds > 0",
        default=60.0,
        help="per-run timeout in seconds",
        field="run_timeout_s",
        ini_type="float",
    ),
    Option(
        "case_timeout",
        ("--disseqt-case-timeout",),
        "disseqt_case_timeout",
        "DISSEQT_CASE_TIMEOUT",
        _positive_float,
        "seconds > 0",
        help="per-case timeout in seconds",
        field="case_timeout_s",
        ini_type="float",
    ),
    Option(
        "suite_timeout",
        ("--disseqt-suite-timeout",),
        "disseqt_suite_timeout",
        "DISSEQT_SUITE_TIMEOUT",
        _positive_float,
        "seconds > 0",
        help="whole-suite timeout in seconds",
        field="suite_timeout_s",
        ini_type="float",
    ),
    Option(
        "judge_concurrency",
        ("--disseqt-judge-concurrency",),
        "disseqt_judge_concurrency",
        "DISSEQT_JUDGE_CONCURRENCY",
        _int,
        "an integer >= 0",
        default=4,
        help="runner-side judge concurrency (0 = no limit)",
        field="judge_concurrency",
        ini_type="int",
    ),
    Option(
        "judge_rps",
        ("--disseqt-judge-rps",),
        "disseqt_judge_rps",
        "DISSEQT_JUDGE_RPS",
        _positive_float,
        "requests per second > 0",
        help="runner-side judge rate limit",
        field="judge_rps",
        ini_type="float",
    ),
    Option(
        "agent",
        ("--disseqt-agent",),
        "disseqt_agent_factory",
        "DISSEQT_AGENT",
        str,
        "module:attr",
        help="default agent factory, as module:attr",
    ),
    Option(
        "split_repeats",
        ("--disseqt-split-repeats",),
        "disseqt_split_repeats",
        "DISSEQT_SPLIT_REPEATS",
        _bool,
        "a boolean",
        action="store_true",
        default=False,
        help="one pytest item per repeat (for debugging single repeats)",
        ini_type="bool",
    ),
    Option(
        "only_repeat",
        ("--disseqt-only-repeat",),
        None,
        None,
        _int,
        "an integer >= 0",
        help="run only repeat I of each selected case",
    ),
    Option(
        "run_label",
        ("--disseqt-run-label",),
        None,
        "DISSEQT_RUN_LABEL",
        str,
        "text",
        help="a label stored in the SuiteResult",
        field="run_label",
    ),
    Option(
        "ascii",
        ("--disseqt-ascii",),
        "disseqt_ascii",
        "DISSEQT_ASCII",
        _bool,
        "a boolean",
        action="store_true",
        help="use ASCII symbols in the repeat strip",
        ini_type="bool",
    ),
    Option(
        "case_globs",
        (),
        "disseqt_case_globs",
        None,
        _list(" "),
        "space-separated globs",
        default=["*.agent.yaml", "*.agent.yml", "*.agent.json"],
        ini_type="args",
    ),
)


def _declare_ini(parser: Any, opt: Option) -> None:
    """Typed where pytest supports it (so pytest 9's native TOML ``disseqt_workers = 2`` works);
    plain strings on pytest 8.0, which has no ``int`` / ``float`` ini types."""
    assert opt.ini is not None
    if opt.ini_type:
        try:
            parser.addini(opt.ini, help=opt.help or opt.name, type=opt.ini_type, default=None)
            return
        except (AssertionError, ValueError, TypeError):
            pass
    parser.addini(opt.ini, help=opt.help or opt.name, default=None)


def add_options(parser: Any) -> None:
    group = parser.getgroup("disseqt", "disseqt agent testing")
    for opt in OPTIONS:
        if opt.flags:
            kwargs: dict[str, Any] = {"dest": opt.dest, "default": None, "help": opt.help}
            if opt.action == "store_true":
                kwargs["action"] = "store_true"
                kwargs["default"] = None
            elif opt.action == "append":
                kwargs["action"] = "append"
            group.addoption(*opt.flags, **kwargs)
        if opt.ini:
            _declare_ini(parser, opt)


def _error(opt: Option, value: str, source: str, why: str) -> pytest.UsageError:
    from ..runner.errors import format_code

    flag = opt.flags[0] if opt.flags else (opt.ini or opt.name)
    return pytest.UsageError(
        format_code(
            "DSQ-U01", value=value, option=flag, source=source, expected=opt.expected or why
        )
    )


def ini_text_from_raw(raw: Any) -> str | None:
    """A configured value as text, or ``None`` if unset. TOML configuration (pytest 9) hands us
    ints, floats, bools and lists: every one is stringified and then parsed by the option's own
    parser, so ``disseqt_seed = 11`` and ``disseqt_seed = "11"`` mean the same thing."""
    raw = getattr(raw, "value", raw)  # pytest 9.2 wraps values in a ConfigValue
    if raw is None or raw == "" or raw == []:
        return None
    if isinstance(raw, (list, tuple)):
        return " ".join(str(x) for x in raw)
    if isinstance(raw, bool):
        return "true" if raw else "false"
    return str(raw)


def _ini_text(config: pytest.Config, name: str) -> str | None:
    """The configured value as text, or ``None`` if unset. Typed declarations come back typed
    (int / float / bool / list). When pytest refuses a value for its declared type (a TOML int
    for a string option: ``TypeError`` before pytest 9.2, ``UsageError`` from it), read the raw
    configuration instead of failing, so every option accepts any scalar."""
    try:
        raw: Any = config.getini(name)
    except (TypeError, pytest.UsageError):
        import warnings

        with warnings.catch_warnings():  # config.inicfg is deprecated in pytest 9.1, removed in 10
            warnings.simplefilter("ignore")
            try:
                raw = getattr(config, "inicfg", {}).get(name)
            except Exception:
                raw = getattr(config, "_inicfg", {}).get(name)
    return ini_text_from_raw(raw)


def resolve_options(config: pytest.Config) -> dict[str, Any]:
    """The option values by name, from CLI > env > ini > default. Invalid values raise
    ``pytest.UsageError`` with DSQ-U01."""
    out: dict[str, Any] = {}
    for opt in OPTIONS:
        value: Any = None
        found = False
        if opt.flags:
            raw = config.getoption(opt.dest, None)
            if raw is not None:
                found = True
                if opt.action == "store_true":
                    value = bool(raw)
                elif opt.action == "append":
                    items: list[str] = []
                    for r in raw:
                        if opt.flatten:
                            items.extend(opt.parse(str(r)))
                        else:
                            items.append(r)
                    value = items
                else:
                    try:
                        value = opt.parse(str(raw))
                    except (ValueError, TypeError) as exc:
                        raise _error(opt, str(raw), "CLI", str(exc)) from None
        if not found and opt.env and os.environ.get(opt.env, "") != "":
            text = os.environ[opt.env]
            try:
                value = opt.parse(text)
            except (ValueError, TypeError) as exc:
                raise _error(opt, text, f"env {opt.env}", str(exc)) from None
            found = True
        if not found and opt.ini:
            ini_text = _ini_text(config, opt.ini)
            if ini_text not in (None, ""):
                try:
                    value = opt.parse(str(ini_text))
                except (ValueError, TypeError) as exc:
                    raise _error(opt, str(ini_text), f"ini {opt.ini}", str(exc)) from None
                found = True
        out[opt.name] = value if found else opt.default
    if isinstance(out["tags"], list):  # repeated --disseqt-tags entries stay separate
        out["tags"] = [t for t in out["tags"] if t]
    return out


def any_option_set(config: pytest.Config) -> bool:
    """True if the user passed a ``--disseqt-*`` flag or set a DISSEQT_* variable."""
    for opt in OPTIONS:
        if opt.flags and config.getoption(opt.dest, None) is not None:
            return True
        if opt.env and os.environ.get(opt.env, "") != "":
            return True
    return False


def explicitly_configured(config: pytest.Config) -> bool:
    """True if the user passed a ``--disseqt-*`` flag or set a ``disseqt_*`` ini key (env
    variables alone do not count: they may be set for an unrelated project)."""
    for opt in OPTIONS:
        if opt.flags and config.getoption(opt.dest, None) is not None:
            return True
        if opt.ini and _ini_text(config, opt.ini) is not None:
            return True
    return False


def looks_like_case_file(path: Any) -> bool:
    """Cheap, safe check that a ``*.agent.*`` file is a disseqt case file: a mapping holding both
    ``version`` and ``cases``. A file that does not parse but names both keys is still treated as
    a case file, so its syntax error is reported instead of swallowed."""
    import json
    import re

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    data: Any = None
    parsed = False
    try:
        if str(path).lower().endswith(".json"):
            data, parsed = json.loads(text), True
        else:
            import importlib

            data, parsed = importlib.import_module("yaml").safe_load(text), True
    except Exception:  # syntax error, or PyYAML missing for a YAML file
        parsed = False
    if parsed:
        return isinstance(data, dict) and "version" in data and "cases" in data
    return bool(re.search(r"(?m)^[\"']?version[\"']?\s*:", text) or '"version"' in text) and bool(
        re.search(r"(?m)^[\"']?cases[\"']?\s*:", text) or '"cases"' in text
    )
